import os

from dotenv import load_dotenv

load_dotenv()

# Language options
NONE = "None"
ENGLISH = "English"
VIETNAMESE = "Vietnamese"
EN = "en"
VI = "vi"

# Chunking options
NO_CHUNKING = "No Chunking"
RECURSIVE_TOKEN_CHUNKER = "RecursiveTokenChunker"
SEMANTIC_CHUNKER = "SemanticChunker"
AGENTIC_CHUNKER = "AgenticChunker"

# Chat roles
USER = "user"
ASSISTANT = "assistant"

# LLM types
ONLINE_LLM = "online_llm"
LOCAL_LLM = "local_llm"

# LLM providers
GEMINI = "Gemini"
OPENAI = "OpenAI"
OLLAMA = "Ollama"

# Default values
DEFAULT_LOCAL_LLM = "llama3.2:3b"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_NUM_RETRIEVAL = 3

# Data sources
UPLOAD = "UPLOAD"
DB = "DB"

# Embedding models
ENGLISH_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VIETNAMESE_EMBEDDING_MODEL = 'keepitreal/vietnamese-sbert'

# Vector database
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_PREFIX = "rag_collection"

# Search options
VECTOR_SEARCH = "Vector Search"
KEYWORDS_SEARCH = "Keywords Search"
HYDE_SEARCH = "Hyde Search"

# File types
SUPPORTED_FILE_TYPES = ["csv", "json", "pdf", "docx", "doc", "xlsx"]

# UI Messages
MSG_SELECT_LANGUAGE = "Please select a language first"
MSG_UPLOAD_DATA = "Please upload and save data first"
MSG_SELECT_LLM = "Please select and configure LLM first"
MSG_DATA_SAVED_SUCCESS = "Data saved successfully!"
MSG_EXPORT_SUCCESS = "Chatbot exported successfully!"

QDRANT_SERVER = os.getenv('QDRANT_SERVER', 'http://localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
COLLECTION_NAME = os.getenv('QDRANT_SERVER', 'rag_chatbot_collection')

# OPENAI_API_KEY
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
