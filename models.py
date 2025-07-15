from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SearchQuery(BaseModel):
    """Search Request Model"""
    query: str
    top_k: int = 5

class QuestionRequest(BaseModel):
    """Question Request Model"""
    query: str

class ChatRequest(BaseModel):
    """Chat Request Model"""
    query: str
    history: List[Dict[str, str]] = []

class FileInfo(BaseModel):
    """File Information"""
    filename: str
    size_bytes: int
    size_mb: float
    num_pages: int
    has_text: bool
    extracted_text: str

class SearchResult(BaseModel):
    """Search Result"""
    id: str
    score: float
    text: str
    filename: str
    chunk_index: int

class Source(BaseModel):
    """Source of Information"""
    filename: str
    chunk_index: int
    relevance_score: float

class StandardResponse(BaseModel):
    """Standard Response"""
    status: str
    message: str

class ValidationResponse(StandardResponse):
    """Validation Response"""
    file_info: Optional[FileInfo] = None

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    chunks_stored: int
    total_vectors: int
    upload_success: Optional[bool] = True
    storage_method: Optional[str] = "pinecone"

class SearchResponse(StandardResponse):
    """Search Response"""
    query: str
    results: List[SearchResult]
    total_found: int
    search_method: str

class AnswerResponse(BaseModel):
    """Question Response"""
    status: str
    query: str
    answer: str
    confidence: str
    model_used: str
    sources_used: int
    sources: List[Source]

class ChatResponse(BaseModel):
    """Chat Response"""
    query: str 
    answer: str 
    model_used: str 
    context_used: int 
    relevance_score : float

class SearchStatusResponse(BaseModel): 
    """Search System Status""" 
    pinecone_available: bool 
    backup_file_exists: bool 
    backup_file: Optional[str] 
    search_method: str 
    vectors_count: Optional[int] = None