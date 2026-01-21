"""
Shared domain models for chunks and retrieval results.

This module provides unified data models for RAG retrieval, ensuring consistent
field naming across backend, API, and UI layers. Uses frozen dataclasses for
immutability and type safety.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ChunkElement:
    """Unified text chunk model for RAG retrieval.

    Standardizes field naming across layers:
    - 'content' unifies 'text', 'chunk' fields
    - 'source_file' is the standard source document field
    """

    content: str  # Unified: was text/chunk
    score: float = 0.0
    source_file: str = "Unknown"  # Standardized name
    page_number: Optional[int] = None
    element_type: str = "text"  # text, table, list, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    point_id: Optional[int] = None  # Qdrant point ID (auto-incrementing integer)

    @classmethod
    def from_qdrant_payload(
        cls, payload: Dict[str, Any], score: float = 0.0, point_id: Optional[int] = None
    ) -> "ChunkElement":
        """Create from Qdrant search result payload.

        Handles legacy field names:
        - 'chunk' -> 'content'
        - 'source_file' already standard

        Args:
            payload: Qdrant payload dictionary
            score: Similarity score
            point_id: Qdrant point ID
        """
        return cls(
            content=payload.get("chunk", payload.get("content", "")),
            score=score,
            source_file=payload.get("source_file", "Unknown"),
            page_number=payload.get("page_number"),
            element_type=payload.get("element_type", "text"),
            metadata={
                k: v
                for k, v in payload.items()
                if k
                not in ("chunk", "content", "source_file", "page_number", "element_type")
            },
            point_id=point_id,
        )

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {
            "text": self.content,  # API uses 'text' for backward compat
            "score": self.score,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "element_type": self.element_type,
        }
        if self.point_id is not None:
            result["point_id"] = self.point_id
        return result


@dataclass(frozen=True)
class ImageElement:
    """Unified image result model for RAG retrieval.

    Standardizes field naming:
    - 'content' holds caption text (unified naming)
    - 'source_file' is the standard source document field
    """

    content: str  # Caption - unified naming
    image_path: str
    score: float = 0.0
    source_file: str = "Unknown"  # Standardized: was source_document
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_qdrant_payload(
        cls, payload: Dict[str, Any], score: float = 0.0
    ) -> "ImageElement":
        """Create from Qdrant search result payload.

        Handles legacy field names:
        - 'chunk' -> 'content' (caption stored in chunk field)
        - 'source_file' already standard
        """
        return cls(
            content=payload.get("chunk", payload.get("content", "")),
            image_path=payload.get("image_path", ""),
            score=score,
            source_file=payload.get("source_file", "Unknown"),
            page_number=payload.get("page_number"),
            metadata={
                k: v
                for k, v in payload.items()
                if k
                not in ("chunk", "content", "source_file", "page_number", "image_path")
            },
        )

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "caption": self.content,  # API uses 'caption' for backward compat
            "image_path": self.image_path,
            "score": self.score,
            "source_file": self.source_file,
            "page_number": self.page_number,
        }
