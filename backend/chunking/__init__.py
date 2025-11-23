"""
Document chunking strategies package.

This package provides various chunking strategies for processing documents
into manageable pieces for vector storage and retrieval.
"""

from .semantic_chunker import SemanticChunker
from .chunk_result import ChunkResult, EmptyChunkResult

__all__ = [
    "SemanticChunker",
    "ChunkResult",
    "EmptyChunkResult",
]