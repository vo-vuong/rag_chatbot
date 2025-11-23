"""
Result classes for document chunking operations.

This module provides standardized result classes for different chunking strategies,
ensuring consistent data structures and interfaces across the chunking system.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChunkResult:
    """
    Result object for chunking operations.

    This class provides a standardized container for chunking results,
    including the chunked elements, metadata, and statistics about the operation.
    """

    def __init__(
        self,
        chunks: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize chunk result.

        Args:
            chunks: List of chunked elements
            metadata: Additional metadata about the chunking process
            stats: Statistics about the chunking operation
        """
        self.chunks = chunks
        self.metadata = metadata or {}
        self.stats = stats or {}

        # Validate input data
        self._validate_result()

    def _validate_result(self) -> None:
        """Validate the chunk result data."""
        if not isinstance(self.chunks, list):
            raise TypeError("chunks must be a list")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

        if not isinstance(self.stats, dict):
            raise TypeError("stats must be a dictionary")

    @property
    def chunk_count(self) -> int:
        """Get the number of chunks."""
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        """Check if the result contains any chunks."""
        return len(self.chunks) == 0

    @property
    def total_characters(self) -> int:
        """Calculate total characters across all chunks."""
        total = 0
        for chunk in self.chunks:
            if hasattr(chunk, 'text') and chunk.text:
                total += len(chunk.text)
        return total

    @property
    def average_chunk_size(self) -> float:
        """Calculate average chunk size in characters."""
        if self.is_empty:
            return 0.0
        return self.total_characters / self.chunk_count

    def get_chunk_by_index(self, index: int) -> Any:
        """
        Get chunk by index.

        Args:
            index: Index of the chunk to retrieve

        Returns:
            Chunk element at the specified index

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= index < len(self.chunks):
            raise IndexError(
                f"Chunk index {index} out of range (0-{len(self.chunks) - 1})"
            )
        return self.chunks[index]

    def get_chunks_with_text(self) -> List[Any]:
        """
        Get chunks that contain text content.

        Returns:
            List of chunks with non-empty text content
        """
        return [
            chunk
            for chunk in self.chunks
            if hasattr(chunk, 'text') and chunk.text and chunk.text.strip()
        ]

    def update_metadata(self, **kwargs) -> None:
        """
        Update metadata with additional key-value pairs.

        Args:
            **kwargs: Additional metadata to add
        """
        self.metadata.update(kwargs)

    def update_stats(self, **kwargs) -> None:
        """
        Update statistics with additional key-value pairs.

        Args:
            **kwargs: Additional statistics to add
        """
        self.stats.update(kwargs)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the chunking result.

        Returns:
            Dictionary containing result summary
        """
        return {
            "chunk_count": self.chunk_count,
            "total_characters": self.total_characters,
            "average_chunk_size": self.average_chunk_size,
            "chunks_with_text": len(self.get_chunks_with_text()),
            "is_empty": self.is_empty,
            "metadata_keys": list(self.metadata.keys()),
            "stats_keys": list(self.stats.keys()),
        }

    def __repr__(self) -> str:
        """String representation of the chunk result."""
        return f"ChunkResult(chunks={len(self.chunks)}, metadata={len(self.metadata)} keys)"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ChunkResult: {self.chunk_count} chunks, "
            f"{self.total_characters} total characters, "
            f"avg size: {self.average_chunk_size:.1f} chars"
        )

    def __len__(self) -> int:
        """Get the number of chunks using len()."""
        return len(self.chunks)

    def __iter__(self):
        """Iterate over chunks."""
        return iter(self.chunks)

    def __getitem__(self, index: int) -> Any:
        """Get chunk by index using bracket notation."""
        return self.get_chunk_by_index(index)


class EmptyChunkResult(ChunkResult):
    """
    Specialized ChunkResult for empty or failed chunking operations.

    This class provides a convenient way to create empty results with
    appropriate metadata for error conditions.
    """

    def __init__(self, error_message: Optional[str] = None, **metadata):
        """
        Initialize empty chunk result.

        Args:
            error_message: Error message describing why chunking failed
            **metadata: Additional metadata about the failure
        """
        chunks = []
        stats = {"failed": True, "error_message": error_message}

        # Add error message to metadata if provided
        if error_message:
            metadata["error"] = error_message

        super().__init__(chunks=chunks, metadata=metadata, stats=stats)

    @property
    def error_message(self) -> Optional[str]:
        """Get the error message."""
        return self.stats.get("error_message")

    def __repr__(self) -> str:
        """String representation of the empty result."""
        error = self.error_message or "No error"
        return f"EmptyChunkResult(error='{error}')"
