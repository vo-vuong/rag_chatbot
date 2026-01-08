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
AGENTIC_CHUNKER = "AgenticChunker"

# ============================================================
# DOCLING PDF PROCESSING
# ============================================================

DOCLING_CONFIG = {
    "ocr": {
        "enabled": True,
        "engine": "easyocr",
        "languages": ["en", "vi"],
        "confidence_threshold": 0.5,
    },
    "table": {
        "mode": "accurate",  # "accurate" or "fast"
        "do_cell_matching": True,
    },
    "chunking": {
        "tokenizer_type": "openai",  # "openai" or "huggingface"
        "tokenizer_model": "text-embedding-3-small",
        "max_tokens": 1024,  # Optimal chunk size for RAG retrieval
        "merge_peers": True,
        "overlap_tokens": 50,  # Post-processing overlap for context continuity
    },
    "acceleration": {
        "device": "auto",  # "auto", "cpu", "cuda", "mps"
        "num_threads": 4,
    },
    "images_scale": 2.0,
}

# ============================================================
# CHAT ROLES
# ============================================================
USER = "user"
ASSISTANT = "assistant"

# ============================================================
# LLM TYPES & PROVIDERS
# ============================================================
ONLINE_LLM = "online_llm"

OPENAI = "OpenAI"

# Default values
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
QDRANT_TEXT_COLLECTION = os.getenv('QDRANT_TEXT_COLLECTION', 'rag_chatbot_text')

# Search options
VECTOR_SEARCH = "Vector Search"
KEYWORDS_SEARCH = "Keywords Search"
HYDE_SEARCH = "Hyde Search"

# ============================================================
# FILE TYPES
# ============================================================
SUPPORTED_FILE_TYPES = ["csv", "pdf"]

# ============================================================
# PDF PROCESSING (Docling)
# ============================================================
# Docling OCR modes - used in UI for user selection
PDF_PROCESSING_MODES = {
    "no_ocr": "Skip OCR (Recommended)",
    "ocr": "Enable OCR (For scanned PDFs)",
}

PDF_SIZE_WARNING_MB = 10  # Warning threshold for PDF file sizes
PDF_SIZE_LIMIT_MB = 50  # Hard limit for PDF file sizes

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
# CSV PROCESSING
# ============================================================

DEFAULT_CSV_CONFIG = {
    "max_rows_per_chunk": 10,
    "max_chunk_size": 2000,
    "include_headers": True,
    "null_value_handling": "skip",
    "encoding": "utf-8",
    "delimiter": ",",
}

# CSV UI Messages
CSV_UI_MESSAGES = {
    "upload_help": "Upload CSV files to process with column-based chunking. Select "
    "grouping columns to combine related rows into meaningful chunks.",
    "column_selection_help": "Choose columns to group related rows. Rows with the same "
    "values in selected columns will be combined into chunks.",
    "preview_help": "Preview how your data will be chunked based on your column selection.",
    "processing_success": "✅ CSV file processed successfully with {chunk_count} chunks",
    "processing_error": "❌ Error processing CSV file: {error}",
    "large_file_warning": "⚠️ Large CSV file detected. Processing may take some time.",
}

# ============================================================
# API KEYS
# ============================================================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
