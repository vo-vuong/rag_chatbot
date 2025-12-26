"""
Layout-aware chunking using unstructured chunk_by_title.

This module provides layout-aware chunking that preserves PDF structure
(titles, sections, tables) while applying overlap for context preservation.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Table

from backend.chunking import ChunkResult, EmptyChunkResult

logger = logging.getLogger(__name__)


class LayoutChunker:
    """
    Layout-aware chunking using unstructured chunk_by_title.

    Preserves PDF structure (titles, sections, tables) while chunking.
    Implements 15% overlap for context preservation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize layout chunker.

        Args:
            config: Layout chunking configuration
        """
        self.max_characters = config.get("max_characters", 1500)
        self.new_after_n_chars = config.get("new_after_n_chars", 1000)
        self.combine_text_under_n_chars = config.get("combine_text_under_n_chars", 500)
        self.multipage_sections = config.get("multipage_sections", False)
        self.overlap = config.get("overlap", 150)
        self.overlap_all = config.get("overlap_all", False)

        logger.info(
            f"LayoutChunker initialized: max_chars={self.max_characters}, "
            f"overlap={self.overlap}, multipage={self.multipage_sections}"
        )

    def chunk_elements(
        self,
        elements: List[Element],
        language: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        **kwargs,
    ) -> ChunkResult:
        """
        Chunk elements using layout-aware strategy.

        Args:
            elements: List of unstructured elements
            language: Document language
            image_paths: Image paths (preserved in metadata)
            **kwargs: Additional parameters (ignored)

        Returns:
            ChunkResult with layout-aware chunks
        """
        if not elements:
            logger.warning("No elements provided for layout chunking")
            return EmptyChunkResult(
                "No elements", chunker="layout", chunker_type="layout"
            )

        start_time = time.time()
        original_count = len(elements)
        image_paths = image_paths or []

        try:
            logger.info(f"Starting layout chunking for {original_count} elements")

            # Separate tables from other elements
            tables = [e for e in elements if isinstance(e, Table)]
            non_tables = [e for e in elements if not isinstance(e, Table)]

            logger.info(
                f"Found {len(tables)} tables, {len(non_tables)} non-table elements"
            )

            # Chunk non-table elements with chunk_by_title
            if non_tables:
                text_chunks = chunk_by_title(
                    non_tables,
                    max_characters=self.max_characters,
                    new_after_n_chars=self.new_after_n_chars,
                    combine_text_under_n_chars=self.combine_text_under_n_chars,
                    multipage_sections=self.multipage_sections,
                    overlap=self.overlap,
                    overlap_all=self.overlap_all,
                )
            else:
                text_chunks = []

            # Tables remain as individual chunks (no merging)
            table_chunks = tables

            # Combine and sort by page number (if available)
            all_chunks = text_chunks + table_chunks
            all_chunks = self._sort_by_page(all_chunks)

            # Enrich metadata
            all_chunks = self._enrich_metadata(all_chunks, language)

            # Generate statistics
            stats = self._generate_stats(elements, all_chunks, start_time, image_paths)

            # Prepare metadata
            metadata = {
                "chunker": "layout",
                "chunking_strategy": "chunk_by_title",
                "language": language,
                "original_elements": original_count,
                "chunked_elements": len(all_chunks),
                "max_characters": self.max_characters,
                "overlap": self.overlap,
                "tables_count": len(tables),
            }

            logger.info(
                f"Layout chunking completed: {original_count} → "
                f"{len(all_chunks)} chunks ({stats['processing_time']:.2f}s)"
            )

            return ChunkResult(all_chunks, metadata, stats, image_paths)

        except Exception as e:
            logger.error(f"Layout chunking failed: {e}", exc_info=True)
            raise

    def _sort_by_page(self, chunks: List[Element]) -> List[Element]:
        """Sort chunks by page number (if available)."""

        def get_page(elem):
            if hasattr(elem, 'metadata') and hasattr(elem.metadata, 'page_number'):
                return elem.metadata.page_number or 0
            return 0

        return sorted(chunks, key=get_page)

    def _enrich_metadata(
        self, chunks: List[Element], language: Optional[str]
    ) -> List[Element]:
        """Enrich chunks with additional metadata."""
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}

            # Add chunk index
            chunk.metadata.chunk_index = i
            chunk.metadata.chunker_type = "layout"
            chunk.metadata.language = language

            # Preserve element category
            chunk.metadata.element_category = (
                chunk.category if hasattr(chunk, 'category') else "Unknown"
            )

            # Track if element is a table
            chunk.metadata.is_table = isinstance(chunk, Table)

        return chunks

    def _generate_stats(
        self,
        original_elements: List[Element],
        chunks: List[Element],
        start_time: float,
        image_paths: List[str],
    ) -> Dict[str, Any]:
        """Generate chunking statistics."""
        stats = {
            "original_element_count": len(original_elements),
            "chunk_count": len(chunks),
            "compression_ratio": (
                len(chunks) / len(original_elements) if original_elements else 0
            ),
            "image_count": len(image_paths),
            "processing_time": time.time() - start_time,
        }

        # Character statistics
        original_chars = sum(
            len(e.text) for e in original_elements if hasattr(e, 'text')
        )
        chunked_chars = sum(len(c.text) for c in chunks if hasattr(c, 'text'))

        stats.update(
            {
                "original_characters": original_chars,
                "chunked_characters": chunked_chars,
                "character_compression_ratio": (
                    chunked_chars / original_chars if original_chars else 0
                ),
            }
        )

        # Chunk size statistics
        if chunks:
            chunk_sizes = [len(c.text) for c in chunks if hasattr(c, 'text')]
            if chunk_sizes:
                stats.update(
                    {
                        "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
                        "min_chunk_size": min(chunk_sizes),
                        "max_chunk_size": max(chunk_sizes),
                    }
                )

        return stats
