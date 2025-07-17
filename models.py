from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Chat Request Model"""
    query: str

class ChatResponse(BaseModel):
    """Chat Response"""
    query: str 
    answer: str 
    model_used: str 
    context_used: int 
    relevance_score : float