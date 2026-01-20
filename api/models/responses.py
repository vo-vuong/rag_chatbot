"""
Response Models - Pydantic models for API responses.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from backend.models import ChunkElement, ImageElement


class RetrievedChunk(BaseModel):
    """Retrieved text chunk model."""

    text: str  # API uses 'text' for backward compat (internally 'content')
    score: float
    source_file: str = "Unknown"
    page_number: Optional[int] = None
    element_type: str = "text"
    point_id: Optional[str] = None  # Qdrant point ID

    @classmethod
    def from_chunk_element(cls, chunk: "ChunkElement") -> "RetrievedChunk":
        """Create from internal ChunkElement model."""
        from backend.models import ChunkElement

        return cls(
            text=chunk.content,
            score=chunk.score,
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            element_type=chunk.element_type,
            point_id=chunk.point_id,
        )


class ImageResult(BaseModel):
    """Retrieved image result model."""

    caption: str  # API uses 'caption' for backward compat (internally 'content')
    image_path: str
    score: float
    page_number: Optional[int] = None
    source_file: str = "Unknown"

    @classmethod
    def from_image_element(cls, image: "ImageElement") -> "ImageResult":
        """Create from internal ImageElement model."""
        from backend.models import ImageElement

        return cls(
            caption=image.content,
            image_path=image.image_path,
            score=image.score,
            page_number=image.page_number,
            source_file=image.source_file,
        )


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    route: Optional[str] = None
    route_reasoning: Optional[str] = None
    retrieved_chunks: List[RetrievedChunk] = []
    images: List[ImageResult] = []
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    qdrant_connected: bool


class PreviewChunk(BaseModel):
    """Chunk data for preview display with metadata."""

    text: str
    source_file: str
    page_number: Optional[int] = None
    element_type: str = "text"
    chunk_index: int
    file_type: str
    # Extended metadata fields for preview
    headings: List[str] = []
    source: str = "docling"
    bbox: Optional[Dict[str, float]] = None
    chunk_type: str = "hybrid"
    token_count: Optional[int] = None
    processing_strategy: Optional[str] = None
    ocr_used: bool = False


class PreviewImage(BaseModel):
    """Image data for preview display with metadata."""

    caption: str
    image_path: str
    page_number: Optional[int] = None
    source_file: str
    image_hash: str
    # Extended metadata fields for preview
    image_metadata: Dict[str, Any] = {}
    bbox: Optional[Dict[str, float]] = None
    docling_caption: Optional[str] = None
    surrounding_context: Optional[str] = None
    headings: List[str] = []  # Document headings hierarchy for this image
    caption_cost: float = 0.0
    file_type: str = ""
    language: str = "en"
    processing_strategy: str = "docling"


class FullChunkData(BaseModel):
    """Complete chunk data for save operation.

    Includes all metadata fields from DoclingChunker for full traceability.
    """

    text: str
    source_file: str
    page_number: Optional[int] = None
    element_type: str = "text"
    chunk_index: int
    file_type: str
    language: str = "en"
    # Extended metadata fields (flattened from DoclingChunker)
    headings: List[str] = []
    source: str = "docling"  # Processing source identifier
    bbox: Optional[Dict[str, float]] = None  # Bounding box: left, top, right, bottom
    chunk_type: str = "hybrid"  # Chunker type used
    token_count: Optional[int] = None
    processing_strategy: Optional[str] = None  # e.g., "docling", "fallback_pypdf2"
    ocr_used: bool = False


class FullImageData(BaseModel):
    """Complete image data for save operation.

    Includes all metadata fields from image extraction for full traceability.
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


class UploadPreviewResponse(BaseModel):
    """Upload preview response - returns processed data without saving."""

    status: str
    file_name: str
    file_type: str
    chunks: List[PreviewChunk]
    images: List[PreviewImage]
    total_chunks_count: int
    total_images_count: int
    processing_time_seconds: float
    full_chunks_data: List[FullChunkData]
    full_images_data: List[FullImageData]


class SaveUploadResponse(BaseModel):
    """Save upload response - confirms data saved to Qdrant."""

    status: str
    file_name: str
    chunks_count: int
    images_count: int
    message: str
    text_collection: str
    image_collection: str
