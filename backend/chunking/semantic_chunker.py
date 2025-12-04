"""
Semantic chunking implementation using unstructured's chunk_by_title.

This module provides intelligent document chunking that preserves document structure
by using title-based segmentation with configurable parameters.
"""

import logging
from typing import Any, Dict, List, Optional

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Text

from backend.chunking import ChunkResult, EmptyChunkResult

# Configure logging
logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_MAX_CHARACTERS = 1500
DEFAULT_COMBINE_TEXT_UNDER_N_CHARS = 1000
DEFAULT_NEW_AFTER_N_CHARS = 3000


class SemanticChunker:
    """
    Semantic chunking implementation that preserves document structure.

    This class provides intelligent document chunking using unstructured's
    chunk_by_title functionality with configurable parameters and fallback
    strategies.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_characters: int = DEFAULT_MAX_CHARACTERS,
        combine_text_under_n_chars: int = DEFAULT_COMBINE_TEXT_UNDER_N_CHARS,
        new_after_n_chars: int = DEFAULT_NEW_AFTER_N_CHARS,
        multipage_sections: bool = True,
        enforce_strict: bool = False,
    ):
        """
        Initialize semantic chunker with configuration parameters.

        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            max_characters: Maximum characters per chunk
            combine_text_under_n_chars: Combine consecutive text elements under this size
            new_after_n_chars: Start new chunk after this many characters
            multipage_sections: Whether to combine sections across multiple pages
            enforce_strict: Whether to enforce strict chunking rules
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_characters = max_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_chars = new_after_n_chars
        self.multipage_sections = multipage_sections
        self.enforce_strict = enforce_strict

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate chunking parameters."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.max_characters <= 0:
            raise ValueError("max_characters must be positive")
        if self.combine_text_under_n_chars <= 0:
            raise ValueError("combine_text_under_n_chars must be positive")
        if self.new_after_n_chars <= 0:
            raise ValueError("new_after_n_chars must be positive")

    def chunk_elements(
        self, elements: List[Element], language: Optional[str] = None, **kwargs
    ) -> ChunkResult:
        """
        Chunk document elements using semantic chunking.

        Args:
            elements: List of document elements to chunk
            language: Document language (for logging/metadata)
            **kwargs: Additional parameters for chunking

        Returns:
            ChunkResult containing chunked elements and metadata
        """
        if not elements:
            self.logger.warning("No elements provided for chunking")
            return EmptyChunkResult("No elements to chunk", chunker_type="semantic")

        start_time = time.time() if 'time' in globals() else None
        original_count = len(elements)

        try:
            self.logger.info(
                f"Starting semantic chunking for {original_count} elements"
            )

            # Apply semantic chunking using unstructured
            chunks = chunk_by_title(
                elements=elements,
                max_characters=self.max_characters,
                combine_text_under_n_chars=self.combine_text_under_n_chars,
                new_after_n_chars=self.new_after_n_chars,
                multipage_sections=self.multipage_sections,
            )

            # Post-process chunks if needed
            processed_chunks = self._post_process_chunks(chunks)

            # Generate statistics
            stats = self._generate_stats(elements, processed_chunks, start_time)

            # Prepare metadata
            metadata = {
                "chunker": "semantic",
                "language": language,
                "original_elements": original_count,
                "chunked_elements": len(processed_chunks),
                "parameters": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "max_characters": self.max_characters,
                    "combine_text_under_n_chars": self.combine_text_under_n_chars,
                    "new_after_n_chars": self.new_after_n_chars,
                    "multipage_sections": self.multipage_sections,
                },
            }

            self.logger.info(
                f"Semantic chunking completed: {original_count} -> {len(processed_chunks)} chunks"
            )

            return ChunkResult(processed_chunks, metadata, stats)

        except Exception as e:
            self.logger.error(f"Semantic chunking failed: {e}")

            # Fallback to simple chunking
            self.logger.info("Falling back to simple chunking strategy")
            return self._fallback_simple_chunking(elements, language)

    def _post_process_chunks(self, chunks: List[Element]) -> List[Element]:
        """
        Post-process chunks to ensure quality and consistency.

        Args:
            chunks: List of chunks from semantic chunking

        Returns:
            Post-processed chunks
        """
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not hasattr(chunk, 'text') or not chunk.text.strip():
                continue

            # Apply additional processing if needed
            processed_chunk = self._process_single_chunk(chunk, i)
            processed_chunks.append(processed_chunk)

        return processed_chunks

    def _process_single_chunk(self, chunk: Element, index: int) -> Element:
        """
        Process a single chunk to ensure quality.

        Args:
            chunk: The chunk element to process
            index: Index of the chunk in the list

        Returns:
            Processed chunk element
        """
        # Clean up text
        if hasattr(chunk, 'text'):
            # Remove excessive whitespace
            text = ' '.join(chunk.text.split())
            chunk.text = text

        # Add chunk metadata
        if hasattr(chunk, 'metadata'):
            chunk.metadata['chunk_index'] = index
            chunk.metadata['chunker_type'] = 'semantic'
            chunk.metadata['chunking_strategy'] = 'title_based'

        return chunk

    def _generate_stats(
        self,
        original_elements: List[Element],
        chunks: List[Element],
        start_time: Optional[float],
    ) -> Dict[str, Any]:
        """
        Generate statistics about the chunking operation.

        Args:
            original_elements: Original document elements
            chunks: Chunked elements
            start_time: Start time of the operation

        Returns:
            Dictionary containing chunking statistics
        """
        stats = {
            "original_element_count": len(original_elements),
            "chunk_count": len(chunks),
            "compression_ratio": (
                len(chunks) / len(original_elements) if original_elements else 0
            ),
        }

        # Calculate character statistics
        original_chars = sum(
            len(elem.text) for elem in original_elements if hasattr(elem, 'text')
        )
        chunked_chars = sum(
            len(chunk.text) for chunk in chunks if hasattr(chunk, 'text')
        )

        stats.update(
            {
                "original_characters": original_chars,
                "chunked_characters": chunked_chars,
                "character_compression_ratio": (
                    chunked_chars / original_chars if original_chars else 0
                ),
            }
        )

        # Calculate average chunk sizes
        if chunks:
            chunk_sizes = [
                len(chunk.text) for chunk in chunks if hasattr(chunk, 'text')
            ]
            stats.update(
                {
                    "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
                    "min_chunk_size": min(chunk_sizes),
                    "max_chunk_size": max(chunk_sizes),
                }
            )

        # Add timing if available
        if start_time and 'time' in globals():
            stats["processing_time"] = time.time() - start_time

        return stats

    def _fallback_simple_chunking(
        self, elements: List[Element], language: Optional[str] = None
    ) -> ChunkResult:
        """
        Fallback simple chunking strategy.

        Args:
            elements: List of elements to chunk
            language: Document language

        Returns:
            ChunkResult with simple chunks
        """
        self.logger.info("Using simple fallback chunking strategy")

        chunks = []
        current_chunk_text = ""
        current_chunk_metadata = {}
        chunk_index = 0

        for element in elements:
            if not hasattr(element, 'text') or not element.text.strip():
                continue

            element_text = element.text.strip()

            # Check if adding this element would exceed chunk size
            if (
                len(current_chunk_text) + len(element_text) > self.chunk_size
                and current_chunk_text
            ):
                # Create current chunk
                chunk = Text(text=current_chunk_text.strip())
                chunk.metadata = {
                    "chunk_index": chunk_index,
                    "chunker_type": "simple_fallback",
                    "language": language,
                    **current_chunk_metadata,
                }
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap if configured
                if self.chunk_overlap > 0 and current_chunk_text:
                    overlap_start = max(0, len(current_chunk_text) - self.chunk_overlap)
                    current_chunk_text = current_chunk_text[overlap_start:] + "\n\n"
                else:
                    current_chunk_text = ""

                current_chunk_metadata = {}

            # Add element to current chunk
            current_chunk_text += element_text + "\n\n"

            # Merge element metadata
            if hasattr(element, 'metadata'):
                try:
                    # Convert ElementMetadata to dictionary if needed
                    element_metadata = element.metadata
                    if hasattr(element_metadata, '__dict__'):
                        # ElementMetadata object - convert to dict
                        metadata_dict = {
                            'filename': getattr(element_metadata, 'filename', None),
                            'file_directory': getattr(
                                element_metadata, 'file_directory', None
                            ),
                            'filetype': getattr(element_metadata, 'filetype', None),
                            'page_number': getattr(
                                element_metadata, 'page_number', None
                            ),
                            'last_modified': getattr(
                                element_metadata, 'last_modified', None
                            ),
                            'coordinates': getattr(
                                element_metadata, 'coordinates', None
                            ),
                        }
                        # Only add non-None values
                        metadata_dict = {
                            k: v for k, v in metadata_dict.items() if v is not None
                        }
                        current_chunk_metadata.update(metadata_dict)
                    elif isinstance(element_metadata, dict):
                        # Already a dictionary
                        current_chunk_metadata.update(element_metadata)
                except Exception as e:
                    logger.debug(f"Could not process element metadata: {e}")

        # Add final chunk if there's remaining text
        if current_chunk_text.strip():
            chunk = Text(text=current_chunk_text.strip())
            chunk.metadata = {
                "chunk_index": chunk_index,
                "chunker_type": "simple_fallback",
                "language": language,
                **current_chunk_metadata,
            }
            chunks.append(chunk)

        # Prepare metadata
        metadata = {
            "chunker": "simple_fallback",
            "language": language,
            "original_elements": len(elements),
            "chunked_elements": len(chunks),
            "fallback_reason": "semantic_chunking_failed",
        }

        self.logger.info(
            f"Fallback chunking completed: {len(elements)} -> {len(chunks)} chunks"
        )

        return ChunkResult(chunks, metadata)

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information for the chunker.

        Returns:
            Dictionary containing chunker configuration
        """
        return {
            "chunker_type": "semantic",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_characters": self.max_characters,
            "combine_text_under_n_chars": self.combine_text_under_n_chars,
            "new_after_n_chars": self.new_after_n_chars,
            "multipage_sections": self.multipage_sections,
            "enforce_strict": self.enforce_strict,
        }


# Import time for performance measurement
try:
    import time
except ImportError:
    # Fallback for systems where time module might not be available
    time = None
