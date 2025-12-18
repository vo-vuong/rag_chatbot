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
        self, elements: List[Element], language: Optional[str] = None, image_paths: Optional[List[str]] = None, **kwargs
    ) -> ChunkResult:
        """
        Chunk document elements using semantic chunking.

        Args:
            elements: List of document elements to chunk
            language: Document language (for logging/metadata)
            image_paths: List of image paths associated with the document
            **kwargs: Additional parameters for chunking

        Returns:
            ChunkResult containing chunked elements and metadata
        """
        if not elements:
            self.logger.warning("No elements provided for chunking")
            return EmptyChunkResult("No elements to chunk", chunker_type="semantic")

        start_time = time.time() if 'time' in globals() else None
        original_count = len(elements)
        image_paths = image_paths or []

        try:
            self.logger.info(
                f"Starting semantic chunking for {original_count} elements with {len(image_paths)} images"
            )

            # Apply semantic chunking using unstructured
            chunks = chunk_by_title(
                elements=elements,
                max_characters=self.max_characters,
                combine_text_under_n_chars=self.combine_text_under_n_chars,
                new_after_n_chars=self.new_after_n_chars,
                multipage_sections=self.multipage_sections,
            )

            # Post-process chunks and associate images
            processed_chunks = self._post_process_chunks(chunks, image_paths)

            # Generate statistics
            stats = self._generate_stats(elements, processed_chunks, start_time, image_paths)

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
                f"Semantic chunking completed: {original_count} -> {len(processed_chunks)} chunks with {len(image_paths)} images"
            )

            return ChunkResult(processed_chunks, metadata, stats, image_paths)

        except Exception as e:
            self.logger.error(f"Semantic chunking failed: {e}")

            # Fallback to simple chunking
            self.logger.info("Falling back to simple chunking strategy")
            return self._fallback_simple_chunking(elements, language, image_paths)

    def _post_process_chunks(self, chunks: List[Element], image_paths: Optional[List[str]] = None) -> List[Element]:
        """
        Post-process chunks to ensure quality and consistency.

        Args:
            chunks: List of chunks from semantic chunking
            image_paths: List of image paths to associate with chunks

        Returns:
            Post-processed chunks
        """
        processed_chunks = []
        image_paths = image_paths or []

        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not hasattr(chunk, 'text') or not chunk.text.strip():
                continue

            # Apply additional processing if needed
            processed_chunk = self._process_single_chunk(chunk, i, image_paths)
            processed_chunks.append(processed_chunk)

        return processed_chunks

    def _process_single_chunk(self, chunk: Element, index: int, image_paths: Optional[List[str]] = None) -> Element:
        """
        Process a single chunk to ensure quality.

        Args:
            chunk: The chunk element to process
            index: Index of the chunk in the list
            image_paths: List of image paths to associate with the chunk

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
            # Check if metadata is a dict or ElementMetadata object
            if hasattr(chunk.metadata, '__dict__'):
                # ElementMetadata object - add attributes
                chunk.metadata.chunk_index = index
                chunk.metadata.chunker_type = 'semantic'
                chunk.metadata.chunking_strategy = 'title_based'

                # Add image metadata
                if image_paths:
                    relevant_images = self._get_images_for_chunk(chunk, image_paths)
                    chunk.metadata.has_images = bool(relevant_images)
                    chunk.metadata.image_paths = relevant_images
                    chunk.metadata.image_count = len(relevant_images)
                else:
                    chunk.metadata.has_images = False
                    chunk.metadata.image_paths = []
                    chunk.metadata.image_count = 0
            else:
                # Dictionary - can assign directly
                chunk.metadata['chunk_index'] = index
                chunk.metadata['chunker_type'] = 'semantic'
                chunk.metadata['chunking_strategy'] = 'title_based'

                # Add image metadata
                if image_paths:
                    relevant_images = self._get_images_for_chunk(chunk, image_paths)
                    chunk.metadata['has_images'] = bool(relevant_images)
                    chunk.metadata['image_paths'] = relevant_images
                    chunk.metadata['image_count'] = len(relevant_images)
                else:
                    chunk.metadata['has_images'] = False
                    chunk.metadata['image_paths'] = []
                    chunk.metadata['image_count'] = 0

        return chunk

    def _get_images_for_chunk(self, chunk: Element, all_image_paths: List[str]) -> List[str]:
        """
        Get images relevant to specific chunk based on page correlation.

        Args:
            chunk: The chunk element to find images for
            all_image_paths: List of all image paths from the document

        Returns:
            List of image paths relevant to the chunk
        """
        if not all_image_paths:
            return []

        # Get page number from chunk metadata
        chunk_page = None
        if hasattr(chunk, 'metadata'):
            # Handle different metadata formats
            if hasattr(chunk.metadata, 'page_number'):
                chunk_page = chunk.metadata.page_number
            elif isinstance(chunk.metadata, dict):
                chunk_page = chunk.metadata.get('page_number')
            # Also check for page info in coordinates
            if chunk_page is None and isinstance(chunk.metadata, dict):
                coordinates = chunk.metadata.get('coordinates')
                if coordinates and hasattr(coordinates, 'points'):
                    # Extract page from coordinates if available
                    chunk_page = getattr(coordinates, 'page_number', None)

        if chunk_page is None:
            return []

        # Filter images by page number
        relevant_images = []
        for img_path in all_image_paths:
            # Check if image path contains page reference
            # Support different naming patterns: page_1_, _page1_, p1_, etc.
            page_patterns = [
                f"page_{chunk_page}_",
                f"_page{chunk_page}_",
                f"page{chunk_page}_",
                f"-page{chunk_page}-",
                f"p{chunk_page}_",
                f"_p{chunk_page}_",
            ]

            # Also check for patterns without underscores or separators
            if f"page{chunk_page}" in img_path.lower():
                relevant_images.append(img_path)
                continue
            if f"p{chunk_page}" in img_path.lower():
                relevant_images.append(img_path)
                continue

            if any(pattern in img_path for pattern in page_patterns):
                relevant_images.append(img_path)

        return relevant_images

    def _generate_stats(
        self,
        original_elements: List[Element],
        chunks: List[Element],
        start_time: Optional[float],
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate statistics about the chunking operation.

        Args:
            original_elements: Original document elements
            chunks: Chunked elements
            start_time: Start time of the operation
            image_paths: List of image paths associated with the document

        Returns:
            Dictionary containing chunking statistics
        """
        image_paths = image_paths or []
        stats = {
            "original_element_count": len(original_elements),
            "chunk_count": len(chunks),
            "compression_ratio": (
                len(chunks) / len(original_elements) if original_elements else 0
            ),
            "image_count": len(image_paths),
            "chunks_with_images": 0,
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

        # Count chunks with images
        if image_paths:
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and chunk.metadata.get('has_images', False):
                    stats["chunks_with_images"] += 1

        # Add timing if available
        if start_time and 'time' in globals():
            stats["processing_time"] = time.time() - start_time

        return stats

    def _fallback_simple_chunking(
        self, elements: List[Element], language: Optional[str] = None, image_paths: Optional[List[str]] = None
    ) -> ChunkResult:
        """
        Fallback simple chunking strategy.

        Args:
            elements: List of elements to chunk
            language: Document language
            image_paths: List of image paths associated with the document

        Returns:
            ChunkResult with simple chunks
        """
        self.logger.info("Using simple fallback chunking strategy")
        image_paths = image_paths or []

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

                # Prepare chunk metadata with image information
                chunk_metadata = {
                    "chunk_index": chunk_index,
                    "chunker_type": "simple_fallback",
                    "language": language,
                    **current_chunk_metadata,
                }

                # Add image metadata
                if image_paths:
                    relevant_images = self._get_images_for_chunk_from_metadata(chunk_metadata, image_paths)
                    chunk_metadata.update({
                        'has_images': bool(relevant_images),
                        'image_paths': relevant_images,
                        'image_count': len(relevant_images),
                    })
                else:
                    chunk_metadata.update({
                        'has_images': False,
                        'image_paths': [],
                        'image_count': 0,
                    })

                chunk.metadata = chunk_metadata
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

            # Prepare final chunk metadata with image information
            chunk_metadata = {
                "chunk_index": chunk_index,
                "chunker_type": "simple_fallback",
                "language": language,
                **current_chunk_metadata,
            }

            # Add image metadata
            if image_paths:
                relevant_images = self._get_images_for_chunk_from_metadata(chunk_metadata, image_paths)
                chunk_metadata.update({
                    'has_images': bool(relevant_images),
                    'image_paths': relevant_images,
                    'image_count': len(relevant_images),
                })
            else:
                chunk_metadata.update({
                    'has_images': False,
                    'image_paths': [],
                    'image_count': 0,
                })

            chunk.metadata = chunk_metadata
            chunks.append(chunk)

        # Prepare metadata
        metadata = {
            "chunker": "simple_fallback",
            "language": language,
            "original_elements": len(elements),
            "chunked_elements": len(chunks),
            "fallback_reason": "semantic_chunking_failed",
            "image_count": len(image_paths),
        }

        self.logger.info(
            f"Fallback chunking completed: {len(elements)} -> {len(chunks)} chunks with {len(image_paths)} images"
        )

        return ChunkResult(chunks, metadata, image_paths=image_paths)

    def _get_images_for_chunk_from_metadata(self, chunk_metadata: Dict[str, Any], all_image_paths: List[str]) -> List[str]:
        """
        Get images relevant to a chunk based on its metadata.

        Args:
            chunk_metadata: Metadata dictionary for the chunk
            all_image_paths: List of all image paths from the document

        Returns:
            List of image paths relevant to the chunk
        """
        if not all_image_paths:
            return []

        # Get page number from chunk metadata
        chunk_page = chunk_metadata.get('page_number')
        if not chunk_page:
            return []

        # Filter images by page number
        relevant_images = []
        for img_path in all_image_paths:
            # Check if image path contains page reference
            page_patterns = [
                f"page_{chunk_page}_",
                f"_page{chunk_page}_",
                f"page{chunk_page}_",
                f"-page{chunk_page}-",
                f"p{chunk_page}_",
                f"_p{chunk_page}_",
            ]

            # Also check for patterns without underscores or separators
            if f"page{chunk_page}" in img_path.lower():
                relevant_images.append(img_path)
                continue
            if f"p{chunk_page}" in img_path.lower():
                relevant_images.append(img_path)
                continue

            if any(pattern in img_path for pattern in page_patterns):
                relevant_images.append(img_path)

        return relevant_images

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
