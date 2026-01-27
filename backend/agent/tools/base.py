"""
Base utilities for agent tools.

Provides serialization helpers and common tool patterns.
"""

from typing import Any, Dict

from backend.models import ChunkElement, ImageElement


def serialize_chunk(chunk: ChunkElement) -> Dict[str, Any]:
    """Convert ChunkElement to serializable dict for tool output."""
    return {
        "content": chunk.content,
        "score": chunk.score,
        "source_file": chunk.source_file,
        "page_number": chunk.page_number,
        "element_type": chunk.element_type,
        "point_id": chunk.point_id,
    }


def serialize_image(image: ImageElement) -> Dict[str, Any]:
    """Convert ImageElement to serializable dict for tool output."""
    return {
        "content": image.content,
        "image_path": image.image_path,
        "score": image.score,
        "source_file": image.source_file,
        "page_number": image.page_number,
    }


def deserialize_chunk(data: Dict[str, Any]) -> ChunkElement:
    """Convert dict back to ChunkElement."""
    return ChunkElement(
        content=data.get("content", ""),
        score=data.get("score", 0.0),
        source_file=data.get("source_file", "Unknown"),
        page_number=data.get("page_number"),
        element_type=data.get("element_type", "text"),
        point_id=data.get("point_id"),
    )


def deserialize_image(data: Dict[str, Any]) -> ImageElement:
    """Convert dict back to ImageElement."""
    return ImageElement(
        content=data.get("content", ""),
        image_path=data.get("image_path", ""),
        score=data.get("score", 0.0),
        source_file=data.get("source_file", "Unknown"),
        page_number=data.get("page_number"),
    )
