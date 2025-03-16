"""
Data models for the Canvas QA Agent
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class SearchPath(BaseModel):
    """Model for search path configuration"""
    modules: List[str]
    courses: List[str]
    reasoning: Optional[str]

class SearchResult(BaseModel):
    """Model for search results"""
    module: str
    matches: List[Any]

class PDFAnalysisResult(BaseModel):
    """Model for PDF analysis results"""
    source: str
    content: Any
    page_numbers: List[int]
    error: Optional[str]
