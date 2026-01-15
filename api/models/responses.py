"""
Response Models - Pydantic models for API responses.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """Retrieved text chunk model."""

    text: str
    score: float
    source_file: str = "Unknown"


class ImageResult(BaseModel):
    """Retrieved image result model."""

    caption: str
    image_path: str
    score: float
    page_number: Optional[int] = None
    source_file: str = "Unknown"


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
