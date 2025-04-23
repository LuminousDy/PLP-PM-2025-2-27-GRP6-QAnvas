"""
PDF analysis implementation
"""
import os
from typing import Dict, List, Any, Optional, Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from config.settings import PDF_BASE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from models.data_models import PDFAnalysisResult

class PDFAnalyzer:
    """PDF document analysis tool"""
    def __init__(self):
        """Initialize PDF analyzer with text splitter and embedding model"""
        # Text chunking parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"PDF Analyzer using device: {self.device}")
        
        # Initialize embedding model for semantic search
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
            print(f"Initialized embedding model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            print(f"Error initializing embedding model: {str(e)}")
            self.embedding_model = None
    
    def analyze(self, file_path: str, query: str) -> Dict[str, Any]:
        """
        Analyze PDF document and extract content relevant to the query
        
        Args:
            file_path: Path to the PDF file (relative to PDF_BASE_PATH)
            query: User query to find relevant information
            
        Returns:
            Dictionary with analysis results
        """
        # Construct full path
        full_path = os.path.join(PDF_BASE_PATH, file_path)
        if not os.path.exists(full_path):
            return PDFAnalysisResult(
                source=file_path,
                content=[],
                page_numbers=[],
                error=f"File not found: {file_path}"
            ).dict()
        
        try:
            # Load PDF document
            loader = PyMuPDFLoader(full_path)
            pages = loader.load()
            
            # Get document metadata
            total_pages = len(pages)
            metadata = self._extract_metadata(pages)
            
            # Split document into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Find relevant chunks using semantic search
            if self.embedding_model and query:
                relevant_chunks, chunk_scores = self._find_relevant_chunks(chunks, query)
                relevant_pages = self._get_page_numbers(relevant_chunks)
            else:
                # If no embedding model or no query, return all chunks
                relevant_chunks = chunks[:10]  # Limit to first 10 chunks for manageability
                chunk_scores = [1.0] * len(relevant_chunks)  # Default score
                relevant_pages = list(range(min(10, total_pages)))
            
            # Format chunks for output
            formatted_chunks = self._format_chunks(relevant_chunks, chunk_scores)
            
            return {
                "source": file_path,
                "total_pages": total_pages,
                "metadata": metadata,
                "content": formatted_chunks,
                "page_numbers": relevant_pages,
                "error": None
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "source": file_path,
                "content": [],
                "page_numbers": [],
                "error": f"Error analyzing PDF: {str(e)}\n{error_details}"
            }
    
    def _extract_metadata(self, pages: List[Any]) -> Dict[str, Any]:
        """Extract metadata from PDF document"""
        if not pages:
            return {}
        
        # Get metadata from first page
        metadata = pages[0].metadata.copy() if hasattr(pages[0], 'metadata') else {}
        
        # Remove large or unnecessary fields
        keys_to_remove = ['page_content', 'source', 'pdf_img']
        for key in keys_to_remove:
            if key in metadata:
                del metadata[key]
        
        return metadata
    
    def _find_relevant_chunks(self, chunks: List[Any], query: str) -> Tuple[List[Any], List[float]]:
        """Find chunks most relevant to the query using semantic search"""
        if not chunks:
            return [], []
        
        try:
            # Extract text from chunks
            texts = [chunk.page_content for chunk in chunks]
            
            # Create query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            
            # Create document embeddings (in batches)
            batch_size = 16
            chunk_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_tensor=True)
                chunk_embeddings.append(batch_embeddings)
            
            # Combine batch embeddings
            if len(chunk_embeddings) == 1:
                all_embeddings = chunk_embeddings[0]
            else:
                all_embeddings = torch.cat(chunk_embeddings, dim=0)
            
            # Calculate cosine similarity
            similarities = self._cosine_similarity(query_embedding, all_embeddings)
            
            # Get indices of top 5 most similar chunks
            top_k = min(5, len(chunks))
            top_indices = similarities.argsort(descending=True)[:top_k].cpu().numpy()
            
            # Get similarity scores
            top_scores = [similarities[idx].item() for idx in top_indices]
            
            # Get top chunks
            top_chunks = [chunks[idx] for idx in top_indices]
            
            return top_chunks, top_scores
        
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            # Fallback to returning first few chunks
            return chunks[:5], [1.0, 0.9, 0.8, 0.7, 0.6][:len(chunks[:5])]
    
    def _cosine_similarity(self, query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate cosine similarity between query and documents"""
        # Normalize embeddings
        query_embedding = query_embedding / query_embedding.norm()
        document_embeddings = document_embeddings / document_embeddings.norm(dim=1, keepdim=True)
        
        # Calculate similarity (dot product of normalized vectors)
        return torch.matmul(document_embeddings, query_embedding)
    
    def _get_page_numbers(self, chunks: List[Any]) -> List[int]:
        """Extract page numbers from chunks"""
        page_numbers = set()
        
        for chunk in chunks:
            if hasattr(chunk, 'metadata') and 'page' in chunk.metadata:
                page_number = chunk.metadata['page']
                # Convert to int if it's a string
                if isinstance(page_number, str) and page_number.isdigit():
                    page_number = int(page_number)
                page_numbers.add(page_number)
        
        return sorted(list(page_numbers))
    
    def _format_chunks(self, chunks: List[Any], scores: List[float]) -> List[Dict[str, Any]]:
        """Format chunks for output"""
        formatted_chunks = []
        
        for chunk, score in zip(chunks, scores):
            chunk_dict = {
                "content": chunk.page_content,
                "relevance_score": round(score, 3)
            }
            
            # Add metadata
            if hasattr(chunk, 'metadata'):
                # Filter metadata to include only useful fields
                metadata = {}
                for key, value in chunk.metadata.items():
                    if key not in ['source', 'pdf_img']:  # Skip large fields
                        metadata[key] = value
                chunk_dict["metadata"] = metadata
            
            formatted_chunks.append(chunk_dict)
        
        return formatted_chunks
