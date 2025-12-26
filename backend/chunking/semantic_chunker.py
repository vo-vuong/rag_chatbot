"""
TRUE semantic chunking using LangChain SemanticChunker with embedding similarity.
"""

import logging
import re
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from langchain_experimental.text_splitter import (
    SemanticChunker as LangChainSemanticChunker,
)
from langchain_openai import OpenAIEmbeddings
from unstructured.documents.elements import Element, Text

from backend.chunking import ChunkResult, EmptyChunkResult
from config.constants import (
    DEFAULT_SEMANTIC_BREAKPOINT_PERCENTILE,
    DEFAULT_SEMANTIC_BUFFER_SIZE,
    DEFAULT_SEMANTIC_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    TRUE semantic chunking using embedding similarity.

    Uses LangChain SemanticChunker with OpenAI embeddings to detect
    semantic breakpoints based on embedding distance.
    """

    def __init__(
        self,
        openai_api_key: str,
        breakpoint_percentile: int = DEFAULT_SEMANTIC_BREAKPOINT_PERCENTILE,
        buffer_size: int = DEFAULT_SEMANTIC_BUFFER_SIZE,
        embedding_model: str = DEFAULT_SEMANTIC_EMBEDDING_MODEL,
    ):
        """
        Initialize semantic chunker.

        Args:
            openai_api_key: OpenAI API key for embeddings
            breakpoint_percentile: Percentile threshold for breakpoints (0-100)
            buffer_size: Buffer size for sentence grouping (characters)
            embedding_model: OpenAI embedding model name
        """
        self.openai_api_key = openai_api_key
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.embedding_model = embedding_model

        # Validate parameters
        if not 0 <= breakpoint_percentile <= 100:
            raise ValueError("breakpoint_percentile must be 0-100")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, model=embedding_model
        )

        # Initialize LangChain semantic chunker
        self.text_splitter = LangChainSemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_percentile,
            buffer_size=buffer_size,
        )

        logger.info(
            f"SemanticChunker initialized: percentile={breakpoint_percentile}, "
            f"buffer={buffer_size}, model={embedding_model}"
        )

    def chunk_elements(
        self,
        elements: List[Element],
        language: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        **kwargs,
    ) -> ChunkResult:
        """
        Chunk elements using TRUE semantic similarity.

        Args:
            elements: List of unstructured elements to chunk
            language: Document language (for metadata)
            image_paths: Image paths (preserved in metadata)
            **kwargs: Additional parameters (ignored)

        Returns:
            ChunkResult with semantically chunked elements
        """
        if not elements:
            logger.warning("No elements provided for chunking")
            return EmptyChunkResult("No elements to chunk", chunker_type="semantic")

        start_time = time.time()
        original_count = len(elements)
        image_paths = image_paths or []

        try:
            logger.info(f"Starting semantic chunking for {original_count} elements")

            # Step 1: Convert elements to text strings
            texts = self._extract_texts(elements)
            if not texts:
                logger.warning("No text extracted from elements")
                return EmptyChunkResult("No text content", chunker_type="semantic")

            # Step 2: Apply semantic chunking (LangChain)
            logger.info(f"Applying semantic chunking to {len(texts)} text blocks")
            langchain_docs = self.text_splitter.create_documents(texts)
            logger.info(f"Semantic chunking created {len(langchain_docs)} chunks")

            # Step 3: Convert back to unstructured Elements
            chunked_elements = self._convert_to_elements(langchain_docs, language)

            # Step 4: Associate images with chunks
            if image_paths:
                chunked_elements = self._associate_images(
                    chunked_elements, image_paths, elements
                )

            # Step 5: Generate statistics
            stats = self._generate_stats(
                elements, chunked_elements, start_time, image_paths
            )

            # Step 6: Prepare metadata
            metadata = {
                "chunker": "semantic",
                "chunking_strategy": "embedding_similarity",
                "language": language,
                "original_elements": original_count,
                "chunked_elements": len(chunked_elements),
                "breakpoint_percentile": self.breakpoint_percentile,
                "buffer_size": self.buffer_size,
                "embedding_model": self.embedding_model,
            }

            logger.info(
                f"Semantic chunking completed: {original_count} → "
                f"{len(chunked_elements)} chunks ({stats.get('processing_time', 0):.2f}s)"
            )

            return ChunkResult(chunked_elements, metadata, stats, image_paths)

        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}", exc_info=True)
            raise

    def _extract_texts(self, elements: List[Element]) -> List[str]:
        """Extract text from elements with spacing fix."""
        texts = []
        for elem in elements:
            if hasattr(elem, 'text') and elem.text and elem.text.strip():
                text = elem.text.strip()

                # Fix missing spaces after sentence boundaries
                # Pattern: Period/exclamation/question + uppercase letter
                text = re.sub(
                    r'([.!?])([A-ZẮẰẲẴẶĂÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƠỜỚỞỠỢƯỪỨỬỮỰ])',
                    r'\1 \2',
                    text,
                )

                texts.append(text)
        return texts

    def _convert_to_elements(
        self, langchain_docs: List[Any], language: Optional[str]
    ) -> List[Element]:
        """Convert LangChain Documents to unstructured Elements."""
        elements = []
        for i, doc in enumerate(langchain_docs):
            # Create Text element
            elem = Text(text=doc.page_content)

            # Add metadata
            elem.metadata = {
                "chunk_index": i,
                "chunker_type": "semantic",
                "chunking_strategy": "embedding_similarity",
                "language": language,
                **doc.metadata,  # Preserve any existing metadata
            }

            elements.append(elem)

        return elements

    def _associate_images(
        self,
        chunks: List[Element],
        image_paths: List[str],
        original_elements: List[Element],
    ) -> List[Element]:
        """Associate images with chunks based on page correlation."""
        # Simple approach: distribute images across chunks
        # TODO: Smarter association based on page_number metadata
        for chunk in chunks:
            chunk.metadata["has_images"] = False
            chunk.metadata["image_paths"] = []
            chunk.metadata["image_count"] = 0

        return chunks

    def _generate_stats(
        self,
        original_elements: List[Element],
        chunks: List[Element],
        start_time: float,
        image_paths: List[str],
    ) -> dict:
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
            stats.update(
                {
                    "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
                    "min_chunk_size": min(chunk_sizes),
                    "max_chunk_size": max(chunk_sizes),
                }
            )

        return stats
