"""
PDF analysis implementation
"""
import os
from typing import Dict
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import PDF_BASE_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from models.data_models import PDFAnalysisResult

class PDFAnalyzer:
    """PDF document analysis tool"""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    
    def analyze(self, file_path: str, query: str) -> PDFAnalysisResult:
        full_path = os.path.join(PDF_BASE_PATH, file_path)
        if not os.path.exists(full_path):
            return PDFAnalysisResult(
                source=file_path,
                content=[],
                page_numbers=[],
                error="File not found"
            )
        
        try:
            loader = PyMuPDFLoader(full_path)
            pages = loader.load()
            splits = self.text_splitter.split_documents(pages)
            
            return PDFAnalysisResult(
                source=file_path,
                content=splits,
                page_numbers=list(range(len(pages))),
                error=None
            )
        except Exception as e:
            return PDFAnalysisResult(
                source=file_path,
                content=[],
                page_numbers=[],
                error=str(e)
            )
