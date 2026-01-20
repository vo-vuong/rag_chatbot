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
    """Chunk data for save request.

    Matches FullChunkData fields for complete metadata preservation.
    """

    text: str
    source_file: str
    page_number: Optional[int] = None
    element_type: str = "text"
    chunk_index: int
    file_type: str
    # Extended metadata fields (flattened from DoclingChunker)
    headings: List[str] = []
    source: str = "docling"  # Processing source identifier
    bbox: Optional[Dict[str, float]] = None  # Bounding box: left, top, right, bottom
    chunk_type: str = "hybrid"  # Chunker type used
    token_count: Optional[int] = None
    processing_strategy: Optional[str] = None  # e.g., "docling", "fallback_pypdf2"
    ocr_used: bool = False


class SaveImageData(BaseModel):
    """Image data for save request.

    Matches FullImageData fields for complete metadata preservation.
    """

    caption: str
    image_path: str
    page_number: Optional[int] = None
    source_file: str
    image_hash: str
    image_metadata: Dict[str, Any]  # width, height, format, optimized_size_bytes
    # Extended metadata fields
    bbox: Optional[Dict[str, float]] = None  # Bounding box: left, top, right, bottom
    docling_caption: Optional[str] = None  # Caption from Docling extraction
    surrounding_context: Optional[str] = None  # Text context around the image
    headings: List[str] = []  # Document headings hierarchy for this image
    caption_cost: float = 0.0  # Vision API cost for captioning
    file_type: str = ""  # Source file type (pdf, docx)
    language: str = "en"  # Document language
    processing_strategy: str = "docling"  # Processing source


class SaveUploadRequest(BaseModel):
    """Request to save processed upload data to Qdrant."""

    file_name: str
    file_type: str
    language: str = "en"
    chunks: List[SaveChunkData]
    images: List[SaveImageData] = []
