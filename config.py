import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """System Settings"""
    
    # API Keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # File settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = [".pdf"]
    
    # Vector Database settings
    PINECONE_INDEX_NAME = "pdf-rag-index"
    VECTOR_DIMENSION = 384
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    
    # Text processing settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # AI model settings
    AI_MODEL = "qwen/qwen-2.5-72b-instruct"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.2
    TOP_P = 0.9
    
    # Search settings
    DEFAULT_TOP_K = 5
    MAX_CONTEXT_CHUNKS = 3
    CONVERSATION_HISTORY_LIMIT = 4