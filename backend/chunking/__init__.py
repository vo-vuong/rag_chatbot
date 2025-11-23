"""
Document chunking strategies package.

This package provides various chunking strategies for processing documents
into manageable pieces for vector storage and retrieval.
"""

from .semantic_chunker import SemanticChunker

__all__ = [
    "SemanticChunker",
]