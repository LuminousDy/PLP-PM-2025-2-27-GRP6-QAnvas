"""
Canvas search implementation
"""
import os
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from Agent_code.models.data_models import SearchPath, SearchResult
from Agent_code.config.settings import (
    EMBEDDING_MODEL_NAME, 
    MONGODB_URI, 
    MONGODB_DB_NAME, 
    AVAILABLE_MODULES,
    AVAILABLE_COURSES,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    COLLECTION_NAMES,
    COLLECTION_FIELDS
)
from pymongo import MongoClient, ASCENDING, TEXT
import json
import re
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from datetime import datetime
from functools import lru_cache
from Agent_code.tools.multi_intents_decomposition import prompt_analyze
# Download NLTK data for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CanvasSearcher:
    """Tool for searching Canvas content"""
    
    def __init__(self):
        """Initialize MongoDB connection and models"""
        try:
            # Initialize MongoDB connection
            self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
            self.db = self.client[MONGODB_DB_NAME]
            
            # Verify database connection and collections
            db_names = self.client.list_database_names()
            if MONGODB_DB_NAME not in db_names:
                raise Exception(f"Database {MONGODB_DB_NAME} not found. Available databases: {db_names}")
            
            collection_names = self.db.list_collection_names()
            print(f"Available collections: {collection_names}")
            
            # Check for GPU availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device} for embedding model")
            
            # Initialize embeddings model using SentenceTransformer with GPU if available
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            
            # Initialize LLM for path decision
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            
            # Ensure text indexes exist for all collections
            for collection_name in COLLECTION_NAMES.values():
                if collection_name in self.db.list_collection_names():
                    collection = self.db[collection_name]
                    try:
                        collection.create_index([("$**", TEXT)], background=True)
                        collection.create_index([("course_id", ASCENDING)])
                        print(f"Created indexes for collection: {collection_name}")
                    except Exception as e:
                        print(f"Error creating indexes for {collection_name}: {str(e)}")
            
            print("Successfully connected to MongoDB and initialized models")
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            # self.client = None
            # self.db = None
            # self.model = None
            # self.llm = None
            raise

    def search(self, query: str) -> Dict[str, Any]:
        """
        Search Canvas content based on query using hybrid search (keyword + vector)
        """
        try:
            # Determine which collections to search based on query
            search_path = self._determine_search_path(query)
            if not search_path:
                return {"error": "Could not determine search path", "search_path": None, "results": []}

            results = []
            collections_to_search = [COLLECTION_NAMES[m] for m in search_path.get("modules", [])]
            courses_to_search = search_path.get("courses", [])
            
            # Debug print
            print(f"Searching in collections: {collections_to_search}")
            print(f"Searching for courses: {courses_to_search}")
            
            # Get course IDs if courses are specified
            course_ids = []
            if courses_to_search:
                courses_collection = self.db[COLLECTION_NAMES["courses"]]
                for course_name in courses_to_search:
                    # Try exact match first
                    course_doc = courses_collection.find_one({"course_name": course_name})
                    if course_doc:
                        course_ids.append(course_doc["_id"])
                        print(f"Found exact match for course: {course_name}")
                        continue
                    
                    # If no exact match, try partial match
                    base_name = course_name.split('(')[0].strip()
                    pattern = re.compile(f".*{re.escape(base_name)}.*", re.IGNORECASE)
                    course_docs = courses_collection.find({"course_name": pattern})
                    
                    for doc in course_docs:
                        course_ids.append(doc["_id"])
                        print(f"Found partial match: {doc['course_name']} for {course_name}")
            
            print(f"Found course IDs: {course_ids}")
            
            for collection_name in collections_to_search:
                if collection_name not in self.db.list_collection_names():
                    print(f"Collection {collection_name} not found in database")
                    continue
                    
                collection = self.db[collection_name]
                try:
                    # Build query filter for course_id
                    query_filter = {"course_id": {"$in": course_ids}} if course_ids else {}
                    
                    # Get documents matching course filter
                    matching_docs = list(collection.find(query_filter))
                    print(f"Found {len(matching_docs)} documents in {collection_name} matching course filter")
                    
                    if not matching_docs:
                        print(f"No documents found in {collection_name} for the specified courses")
                        if not course_ids:  # If no course filter, try searching all documents
                            matching_docs = list(collection.find())
                            print(f"Searching all documents in {collection_name}, found {len(matching_docs)}")
                        else:
                            continue
                        
                    # Perform hybrid search on documents
                    hybrid_results = self._hybrid_search(query, matching_docs, collection_name)
                    results.extend(hybrid_results)
                        
                except Exception as e:
                    print(f"Error processing collection {collection_name}: {str(e)}")
                    continue

            print(f"Final results count: {len(results)}")
            return {
                "error": None,
                "search_path": search_path,
                "results": results
            }
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return {"error": str(e), "search_path": None, "results": []}

    def _hybrid_search(self, query: str, documents: List[Dict], collection_name: str, top_k: int = 2) -> List[Dict]:
        """
        Perform hybrid search combining BM25 keyword search with vector semantic search
        
        Args:
            query: User query
            documents: List of documents to search
            collection_name: Name of collection being searched
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Prepare texts for embedding and BM25 search
            texts_to_embed = []
            doc_map = {}
            
            for doc in documents:
                if collection_name == COLLECTION_NAMES["announcements"]:
                    text = f"{doc.get('title', '')} {doc.get('message', '')}"
                    fields_to_keep = COLLECTION_FIELDS["announcements"]
                elif collection_name == COLLECTION_NAMES["assignments"]:
                    text = f"{doc.get('name', '')} {doc.get('description', '')}"
                    fields_to_keep = COLLECTION_FIELDS["assignments"]
                elif collection_name == COLLECTION_NAMES["files"]:
                    text = f"{doc.get('filename', '')} {doc.get('folder_path', '')}"
                    fields_to_keep = COLLECTION_FIELDS["files"]
                else:
                    continue
                    
                if text.strip():  # Only process non-empty texts
                    texts_to_embed.append(text)
                    # Create a filtered document with only the required fields
                    filtered_doc = {k: doc.get(k) for k in fields_to_keep if k in doc}
                    filtered_doc["source"] = collection_name
                    doc_map[text] = filtered_doc
            
            if not texts_to_embed:
                print(f"No valid texts to embed found in {collection_name}")
                return []
                
            print(f"Processing {len(texts_to_embed)} texts for hybrid search in {collection_name}")
            
            # 1. Perform BM25 keyword search
            bm25_results = self._perform_bm25_search(query, texts_to_embed, top_k=top_k*2)
            
            # 2. Perform vector similarity search
            vector_results = self._perform_vector_search(query, texts_to_embed, top_k=top_k*2)
            
            # 3. Combine and rerank results
            hybrid_results = self._combine_search_results(
                bm25_results, 
                vector_results, 
                texts_to_embed, 
                doc_map, 
                top_k=top_k
            )
            
            return hybrid_results
                
        except Exception as e:
            print(f"Error during hybrid search: {str(e)}")
            return []
    
    def _perform_bm25_search(self, query: str, texts: List[str], top_k: int = 4) -> List[Dict]:
        """
        Perform BM25 keyword search on texts
        
        Args:
            query: User query
            texts: List of text documents to search
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples for top results
        """
        try:
            # Tokenize corpus for BM25
            tokenized_corpus = [word_tokenize(text.lower()) for text in texts]
            tokenized_query = word_tokenize(query.lower())
            
            # Create BM25 model
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(-bm25_scores)[:top_k]
            
            # Return indices and scores
            return [(idx, bm25_scores[idx]) for idx in top_indices if bm25_scores[idx] > 0]
        
        except Exception as e:
            print(f"Error during BM25 search: {str(e)}")
            return []
    
    def _perform_vector_search(self, query: str, texts: List[str], top_k: int = 4) -> List[Dict]:
        """
        Perform vector similarity search on texts
        
        Args:
            query: User query
            texts: List of text documents to search
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples for top results
        """
        try:
            # Encode query with query prompt
            query_embedding = self.model.encode([query], prompt_name="query", 
                                              show_progress_bar=False)[0]
            
            # Encode documents in batches
            batch_size = 32  # Adjust based on GPU memory
            doc_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                doc_embeddings.extend(batch_embeddings)
                
            doc_embeddings = np.array(doc_embeddings)
            
            # Normalize embeddings for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1)[:, np.newaxis]
            
            # Calculate similarities using vectorized operations
            similarities = np.dot(doc_embeddings, query_embedding)
            
            # Sort and get top results
            top_indices = np.argsort(-similarities)[:top_k]
            
            # Return indices and scores
            return [(idx, float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]
            
        except Exception as e:
            print(f"Error during vector search: {str(e)}")
            return []
    
    def _combine_search_results(self, bm25_results: List[tuple], vector_results: List[tuple], 
                               texts: List[str], doc_map: Dict, top_k: int = 2) -> List[Dict]:
        """
        Combine and rerank BM25 and vector search results
        
        Args:
            bm25_results: List of (index, score) tuples from BM25 search
            vector_results: List of (index, score) tuples from vector search
            texts: Original texts used for search
            doc_map: Mapping from text to document data
            top_k: Number of top results to return
            
        Returns:
            Combined and reranked search results
        """
        try:
            # Create a combined score map
            combined_scores = {}
            
            # Add BM25 scores (normalized between 0 and 1)
            bm25_max = max([score for _, score in bm25_results]) if bm25_results else 1.0
            for idx, score in bm25_results:
                normalized_score = score / bm25_max
                combined_scores[idx] = {"bm25_score": normalized_score, "vector_score": 0, "index": idx}
            
            # Add vector scores (already between 0 and 1)
            for idx, score in vector_results:
                if idx in combined_scores:
                    combined_scores[idx]["vector_score"] = score
                else:
                    combined_scores[idx] = {"bm25_score": 0, "vector_score": score, "index": idx}
            
            # Calculate combined score (weighted average)
            # Here we weight vector search higher (0.7) than keyword search (0.3)
            for idx in combined_scores:
                combined_scores[idx]["combined_score"] = (
                    0.3 * combined_scores[idx]["bm25_score"] + 
                    0.7 * combined_scores[idx]["vector_score"]
                )
            
            # Sort by combined score
            sorted_results = sorted(
                combined_scores.values(), 
                key=lambda x: x["combined_score"], 
                reverse=True
            )[:top_k]
            
            # Convert to final results format
            final_results = []
            for result in sorted_results:
                idx = result["index"]
                text = texts[idx]
                if text in doc_map:
                    result_doc = doc_map[text].copy()
                    result_doc["score"] = result["combined_score"]
                    result_doc["vector_score"] = result["vector_score"]
                    result_doc["keyword_score"] = result["bm25_score"]
                    result_doc["matched_text"] = text
                    
                    # Convert ObjectId to string
                    if "_id" in result_doc:
                        result_doc["_id"] = str(result_doc["_id"])
                    if "course_id" in result_doc:
                        result_doc["course_id"] = str(result_doc["course_id"])
                        
                    final_results.append(result_doc)
            
            return final_results
            
        except Exception as e:
            print(f"Error combining search results: {str(e)}")
            return []

    def _determine_search_path(self, query: str) -> Dict[str, List[str]]:
        """Determine which modules and courses to search based on the query using LLM"""
        
        try:
            # Construct prompt for LLM
            prompt = f"""Given a user query about Canvas content, determine which modules and courses to search in.
            

Available Modules:
{', '.join(AVAILABLE_MODULES)}

Available Courses:
{', '.join(AVAILABLE_COURSES)}

User Query: {query}

Analyze the query and determine:
1. Which modules would contain relevant information? (can be multiple or none)
2. Which specific courses should be searched? (can be multiple or none)

Return your decision as a valid JSON object with this format:
{{
    "modules": ["module1", "module2"],
    "courses": ["course1", "course2"],
    "reasoning": "explanation for your choices"
}}

Only return the JSON object, no other text."""


            # Get LLM response
            response = self.llm.predict(prompt)
            
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Further cleaning to handle potential formatting issues
            response = response.strip()
            
            # Try to extract JSON object if wrapped in other text
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = response[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    data = json.loads(response)
            except json.JSONDecodeError:
                print(f"JSON decode error: {response}")
                # Fallback to a basic search across all modules if parsing fails
                return {
                    "modules": AVAILABLE_MODULES,
                    "courses": [],
                    "reasoning": "Failed to parse search path, using all modules as fallback"
                }
            
            # Validate expected fields
            if "modules" not in data or not isinstance(data["modules"], list):
                data["modules"] = []
            if "courses" not in data or not isinstance(data["courses"], list):
                data["courses"] = []
            if "reasoning" not in data or not isinstance(data["reasoning"], str):
                data["reasoning"] = "No reasoning provided"
                
            # Ensure module and course names are valid
            data["modules"] = [m for m in data["modules"] if m in AVAILABLE_MODULES]
            data["courses"] = [c for c in data["courses"] if c in AVAILABLE_COURSES]
            
            # Log search path decision
            print(f"Determined search path: {data}")
            
            return data
            
        except Exception as e:
            print(f"Error determining search path: {str(e)}")
            # Fallback to searching all modules
            return {
                "modules": AVAILABLE_MODULES,
                "courses": [],
                "reasoning": f"Error determining path: {str(e)}"
            }