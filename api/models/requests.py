"""
Request Models - Pydantic models for API requests.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request model."""

    query: str
    session_id: str
    mode: Literal["rag", "llm_only"] = "rag"
    top_k: int = 3
    score_threshold: float = 0.5


class RAGSearchRequest(BaseModel):
    """RAG search request model."""

    query: str
    collection_type: Literal["text", "image"] = "text"
    top_k: int = 3
    score_threshold: float = 0.5


class UploadConfig(BaseModel):
    """Upload configuration (form fields)."""

    collection_name: str = Field(..., min_length=1)
    language: Literal["en", "vi"] = "en"
    processing_mode: Literal["fast", "ocr"] = "fast"
    csv_columns: Optional[str] = None  # Comma-separated
    vision_failure_mode: Literal["graceful", "strict", "skip"] = "graceful"


class SaveChunkData(BaseModel):
    """Chunk data for save request."""

    text: str
    source_file: str
    page_number: Optional[int] = None
    element_type: str = "text"
    chunk_index: int
    file_type: str


class SaveImageData(BaseModel):
    """Image data for save request."""

    caption: str
    image_path: str
    page_number: Optional[int] = None
    source_file: str
    image_hash: str
    image_metadata: Dict[str, Any]


class SaveUploadRequest(BaseModel):
    """Request to save processed upload data to Qdrant."""

    file_name: str
    file_type: str
    language: str = "en"
    chunks: List[SaveChunkData]
    images: List[SaveImageData] = []
