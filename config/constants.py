import os

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LANGUAGE OPTIONS
# ============================================================
NONE = "None"
ENGLISH = "English"
VIETNAMESE = "Vietnamese"
EN = "en"
VI = "vi"

# ============================================================
# EMBEDDING PROVIDERS
# ============================================================
# Online Embedding Providers
OPENAI_EMBEDDING_PROVIDER = "openai"
OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_DEFAULT_EMBEDDING_DIMENSION = 1536

# Local Embedding Providers (Placeholder for future)
LOCAL_EMBEDDING_PROVIDER = "local"
LOCAL_EMBEDDING_MODELS = {
    "English Small": "all-MiniLM-L6-v2",  # 384 dimensions
    "English Large": "all-mpnet-base-v2",  # 768 dimensions
    "Vietnamese": "keepitreal/vietnamese-sbert",  # 768 dimensions
}

# Deprecated - Keep for backward compatibility
ENGLISH_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VIETNAMESE_EMBEDDING_MODEL = 'keepitreal/vietnamese-sbert'

# ============================================================
# LLM CONFIGURATION
# ============================================================
# Default LLM Model
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Available LLM Models
OPENAI_LLM_MODELS = {
    "GPT-4o Mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-4 Turbo": "gpt-4-turbo",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
}

# ============================================================
# CHUNKING OPTIONS
# ============================================================
NO_CHUNKING = "No Chunking"
RECURSIVE_TOKEN_CHUNKER = "RecursiveTokenChunker"
SEMANTIC_CHUNKER = "SemanticChunker"
AGENTIC_CHUNKER = "AgenticChunker"

# ============================================================
# CHAT ROLES
# ============================================================
USER = "user"
ASSISTANT = "assistant"

# ============================================================
# LLM TYPES & PROVIDERS
# ============================================================
ONLINE_LLM = "online_llm"
LOCAL_LLM = "local_llm"

GEMINI = "Gemini"
OPENAI = "OpenAI"
OLLAMA = "Ollama"

# Default values
DEFAULT_LOCAL_LLM = "llama3.2:3b"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_NUM_RETRIEVAL = 3
DEFAULT_SCORE_THRESHOLD = 0.5

# ============================================================
# DATA SOURCES
# ============================================================
UPLOAD = "UPLOAD"
DB = "DB"

# ============================================================
# VECTOR DATABASE (QDRANT)
# ============================================================
QDRANT_HOST = os.getenv('QDRANT_SERVER', 'http://localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'rag_chatbot_collection')

# Search options
VECTOR_SEARCH = "Vector Search"
KEYWORDS_SEARCH = "Keywords Search"
HYDE_SEARCH = "Hyde Search"

# ============================================================
# FILE TYPES
# ============================================================
SUPPORTED_FILE_TYPES = ["csv", "pdf", "docx"]

# ============================================================
# UI MESSAGES
# ============================================================
MSG_SELECT_LANGUAGE = "Please select a language first"
MSG_UPLOAD_DATA = "Please upload and save data first"
MSG_SELECT_LLM = "Please select and configure LLM first"
MSG_DATA_SAVED_SUCCESS = "Data saved successfully!"
MSG_EXPORT_SUCCESS = "Chatbot exported successfully!"

# ============================================================
# PAGE NAVIGATION
# ============================================================
PAGE_CHAT = "chat"
PAGE_UPLOAD = "upload"
PAGE_DATA_MANAGEMENT = "data_management"

# ============================================================
# API KEYS
# ============================================================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
