"""
Data models for the Canvas QA Agent
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID, uuid4

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

class Message(BaseModel):
    """Model for conversation messages"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class Conversation(BaseModel):
    """Model for storing conversation history"""
    id: UUID = uuid4()
    messages: List[Message] = []
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation"""
        self.messages.append(
            Message(
                role=role,
                content=content,
                timestamp=datetime.now()
            )
        )
        self.updated_at = datetime.now()
    
    def get_last_n_messages(self, n: int = 5) -> List[Message]:
        """Get the last n messages in the conversation"""
        return self.messages[-n:] if len(self.messages) >= n else self.messages[:]
    
    def get_context_for_query(self) -> str:
        """Format conversation history as context for LLM queries"""
        formatted_messages = []
        for msg in self.messages:
            role_prefix = "User" if msg.role == "user" else "Assistant"
            formatted_messages.append(f"{role_prefix}: {msg.content}")
        return "\n".join(formatted_messages)
