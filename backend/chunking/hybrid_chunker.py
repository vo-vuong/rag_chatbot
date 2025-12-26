"""
Hybrid chunking: Layout-aware + Semantic.

Combines unstructured chunk_by_title with LangChain SemanticChunker
for optimal PDF chunking.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from unstructured.documents.elements import Element, Table

from backend.chunking import ChunkResult, EmptyChunkResult
from backend.chunking.layout_chunker import LayoutChunker
from backend.chunking.semantic_chunker import SemanticChunker

logger = logging.getLogger(__name__)


class HybridChunker:
    """
    Hybrid chunking combining layout awareness and semantic similarity.

    5-stage pipeline:
    1. Partition (handled by PDFStrategy)
    2. Preprocess (spacing fixes)
    3. Layout chunking (section grouping)
    4. Semantic chunking (within sections)
    5. Metadata enrichment
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hybrid chunker.

        Args:
            config: Hybrid chunking configuration from session_manager
                {
                    "preprocessing": {...},
                    "layout": {...},
                    "semantic": {...},
                    "metadata": {...},
                    "openai_api_key": "..."
                }
        """
        self.config = config

        # Stage 2: Preprocessing config
        self.fix_spacing = config.get("preprocessing", {}).get("fix_spacing", True)
        self.spacing_pattern = config.get("preprocessing", {}).get(
            "spacing_pattern", r'([.!?])([A-ZẮẰẲẴẶĂÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƠỜỚỞỠỢƯỪỨỬỮỰ])'
        )

        # Stage 3: Layout chunker
        layout_config = config.get("layout", {})
        self.layout_chunker = LayoutChunker(layout_config)

        # Stage 4: Semantic chunker
        semantic_config = config.get("semantic", {})
        self.semantic_chunker = SemanticChunker(
            openai_api_key=config.get("openai_api_key"),
            breakpoint_percentile=semantic_config.get("breakpoint_percentile", 95),
            buffer_size=semantic_config.get("buffer_size", 500),
            embedding_model=semantic_config.get(
                "embedding_model", "text-embedding-3-small"
            ),
        )

        # Stage 5: Metadata config
        self.preserve_coordinates = config.get("metadata", {}).get(
            "preserve_coordinates", True
        )
        self.image_association_mode = config.get("metadata", {}).get(
            "image_association_mode", "page_based"
        )

        logger.info("HybridChunker initialized with 5-stage pipeline")

    def chunk_elements(
        self,
        elements: List[Element],
        language: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        **kwargs,
    ) -> ChunkResult:
        """
        Chunk elements using hybrid approach.

        Args:
            elements: Unstructured elements from partition
            language: Document language
            image_paths: Extracted image paths

        Returns:
            ChunkResult with hybrid chunks
        """
        if not elements:
            logger.warning("No elements provided for hybrid chunking")
            return EmptyChunkResult("No elements", chunker_type="hybrid")

        start_time = time.time()
        original_count = len(elements)
        image_paths = image_paths or []

        try:
            logger.info(f"Starting hybrid chunking for {original_count} elements")

            # Stage 2: Preprocess text (spacing fixes)
            if self.fix_spacing:
                elements = self._preprocess_text(elements)

            # Stage 3: Layout chunking (section grouping with overlap)
            logger.info("Stage 3: Layout chunking")
            layout_result = self.layout_chunker.chunk_elements(
                elements, language=language, image_paths=image_paths
            )
            layout_chunks = layout_result.chunks

            logger.info(
                f"Layout chunking: {original_count} → {len(layout_chunks)} chunks"
            )

            # Stage 4: Semantic chunking within layout chunks
            logger.info("Stage 4: Semantic sub-chunking")
            final_chunks = []
            for i, section_chunk in enumerate(layout_chunks):
                # Skip tables (already optimally sized)
                if isinstance(section_chunk, Table):
                    final_chunks.append(section_chunk)
                    continue

                # Apply semantic chunking to large sections
                section_text_length = (
                    len(section_chunk.text) if hasattr(section_chunk, 'text') else 0
                )

                if section_text_length > 1500:  # Threshold for sub-chunking
                    # Semantic split within section
                    semantic_result = self.semantic_chunker.chunk_elements(
                        [section_chunk], language=language
                    )
                    sub_chunks = semantic_result.chunks

                    # Preserve section metadata in sub-chunks
                    for sub_chunk in sub_chunks:
                        if hasattr(sub_chunk, 'metadata'):
                            sub_chunk.metadata.section_index = i
                            if hasattr(section_chunk, 'metadata') and hasattr(
                                section_chunk.metadata, 'section'
                            ):
                                parent_section = getattr(
                                    section_chunk.metadata, 'section', None
                                )
                                if parent_section:
                                    sub_chunk.metadata.parent_section = parent_section

                    final_chunks.extend(sub_chunks)
                else:
                    # Keep section as-is
                    final_chunks.append(section_chunk)

            logger.info(
                f"Semantic sub-chunking: {len(layout_chunks)} → {len(final_chunks)} final chunks"
            )

            # Stage 5: Metadata enrichment + image association
            logger.info("Stage 5: Metadata enrichment")
            final_chunks = self._enrich_metadata(final_chunks, language, elements)

            if image_paths and self.image_association_mode == "page_based":
                final_chunks = self._associate_images_page_based(
                    final_chunks, image_paths
                )

            # Generate statistics
            stats = self._generate_stats(
                elements, final_chunks, start_time, image_paths
            )

            # Prepare metadata
            metadata = {
                "chunker": "hybrid",
                "chunking_strategy": "layout_semantic",
                "language": language,
                "original_elements": original_count,
                "chunked_elements": len(final_chunks),
                "layout_chunks": len(layout_chunks),
                "final_chunks": len(final_chunks),
                "pipeline_stages": 5,
                "spacing_fixed": self.fix_spacing,
                "image_association_mode": self.image_association_mode,
            }

            logger.info(
                f"Hybrid chunking completed: {original_count} → {len(final_chunks)} chunks "
                f"({stats['processing_time']:.2f}s)"
            )

            return ChunkResult(final_chunks, metadata, stats, image_paths)

        except Exception as e:
            logger.error(f"Hybrid chunking failed: {e}", exc_info=True)
            raise

    def _preprocess_text(self, elements: List[Element]) -> List[Element]:
        """Stage 2: Fix spacing issues in text."""
        logger.info("Stage 2: Preprocessing text (spacing fixes)")

        for elem in elements:
            if hasattr(elem, 'text') and elem.text:
                # Fix missing spaces after sentence boundaries
                elem.text = re.sub(self.spacing_pattern, r'\1 \2', elem.text)

        return elements

    def _enrich_metadata(
        self,
        chunks: List[Element],
        language: Optional[str],
        original_elements: List[Element],
    ) -> List[Element]:
        """Stage 5: Enrich chunk metadata."""
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = type('obj', (object,), {})()

            # Basic metadata
            chunk.metadata.chunk_index = i
            chunk.metadata.chunker_type = "hybrid"
            chunk.metadata.chunking_strategy = "layout_semantic"
            chunk.metadata.language = language

            # Preserve coordinates if available
            if self.preserve_coordinates and hasattr(chunk.metadata, 'coordinates'):
                chunk.metadata.has_coordinates = True

            # Track element type
            chunk.metadata.is_table = isinstance(chunk, Table)
            chunk.metadata.element_category = (
                chunk.category if hasattr(chunk, 'category') else "Unknown"
            )

        return chunks

    def _associate_images_page_based(
        self, chunks: List[Element], image_paths: List[str]
    ) -> List[Element]:
        """Associate images with chunks based on page number."""
        logger.info(f"Associating {len(image_paths)} images with chunks (page-based)")

        # Build image -> page mapping
        # Assume image_paths include page metadata (e.g., "page_5_img_1.png")
        # For now, simple approach: distribute images across chunks

        for chunk in chunks:
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = type('obj', (object,), {})()

            chunk.metadata.has_images = False
            chunk.metadata.image_paths = []
            chunk.metadata.image_count = 0

        # TODO: Implement smart page-based association
        # Requires image metadata with page_number
        # For Phase 03, stub implementation

        return chunks

    def _generate_stats(
        self,
        original_elements: List[Element],
        chunks: List[Element],
        start_time: float,
        image_paths: List[str],
    ) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
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

        # Type distribution
        table_count = sum(1 for c in chunks if isinstance(c, Table))
        stats["table_chunks"] = table_count
        stats["text_chunks"] = len(chunks) - table_count

        return stats
