"""
Docling HybridChunker wrapper for token-aware chunking.

Replaces layout_chunker + semantic_chunker with single Docling HybridChunker.
Supports OpenAI tokenizer (tiktoken) for text-embedding-3-small alignment.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.chunking import ChunkResult, EmptyChunkResult

logger = logging.getLogger(__name__)


@dataclass
class DoclingChunkElement:
    """Chunk element compatible with existing pipeline."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number")


class DoclingChunker:
    """
    Wrapper for Docling HybridChunker with ChunkResult output.

    Token-aware chunking aligned with embedding model context windows.
    Supports both OpenAI (tiktoken) and HuggingFace tokenizers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Docling chunker.

        Args:
            config: Chunking configuration
                {
                    "tokenizer_type": "openai",  # "openai" or "huggingface"
                    "tokenizer_model": "text-embedding-3-small",
                    "max_tokens": 8191,  # OpenAI embedding context limit
                    "merge_peers": True,
                }
        """
        self.config = config or {}
        self.tokenizer = None
        self.chunker = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialize tokenizer and chunker."""
        if self._initialized:
            return

        try:
            import tiktoken
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.openai import (
                OpenAITokenizer,
            )
        except ImportError as e:
            logger.error(f"Docling imports failed: {e}")
            raise ImportError(
                "Install docling-core[chunking-openai]: "
                "pip install docling-core[chunking-openai]"
            ) from e

        tokenizer_type = self.config.get("tokenizer_type", "openai")
        tokenizer_model = self.config.get("tokenizer_model", "text-embedding-3-small")
        max_tokens = self.config.get("max_tokens", 8191)
        merge_peers = self.config.get("merge_peers", True)

        if tokenizer_type == "openai":
            try:
                enc = tiktoken.encoding_for_model(tokenizer_model)
            except KeyError:
                # Fallback for embedding models not in tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                logger.warning(
                    f"Tokenizer for {tokenizer_model} not found, using cl100k_base"
                )

            self.tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=max_tokens)
            logger.info(f"Using OpenAI tokenizer: {tokenizer_model}")
        else:
            # HuggingFace tokenizer
            try:
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )
                from transformers import AutoTokenizer

                hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
                self.tokenizer = HuggingFaceTokenizer(
                    tokenizer=hf_tokenizer, max_tokens=max_tokens
                )
                logger.info(f"Using HuggingFace tokenizer: {tokenizer_model}")
            except ImportError as e:
                raise ImportError(
                    "Install transformers for HuggingFace tokenizer: "
                    "pip install transformers"
                ) from e

        self.chunker = HybridChunker(tokenizer=self.tokenizer, merge_peers=merge_peers)

        self._initialized = True
        logger.info(
            f"DoclingChunker initialized: type={tokenizer_type}, "
            f"model={tokenizer_model}, max_tokens={max_tokens}"
        )

    def chunk_document(
        self,
        doc: Any,
        image_paths: Optional[List[str]] = None,
    ) -> ChunkResult:
        """
        Chunk Docling document using HybridChunker.

        Args:
            doc: Docling DoclingDocument
            image_paths: Associated image paths

        Returns:
            ChunkResult with chunked elements
        """
        image_paths = image_paths or []

        if doc is None:
            return EmptyChunkResult("No document provided", chunker_type="docling")

        try:
            self._ensure_initialized()

            # Chunk document (dl_doc= is the required keyword argument)
            chunk_result = self.chunker.chunk(dl_doc=doc)
            if chunk_result is None:
                return EmptyChunkResult(
                    "HybridChunker returned None", chunker_type="docling"
                )
            chunks = list(chunk_result)

            if not chunks:
                return EmptyChunkResult("No chunks generated", chunker_type="docling")

            # Convert to compatible format
            converted_chunks = self._convert_chunks(chunks, doc)

            # Build metadata
            metadata = {
                "chunker_type": "docling_hybrid",
                "tokenizer_type": self.config.get("tokenizer_type", "openai"),
                "tokenizer_model": self.config.get(
                    "tokenizer_model", "text-embedding-3-small"
                ),
                "max_tokens": self.config.get("max_tokens", 8191),
                "original_chunk_count": len(chunks),
            }

            # Build stats
            total_tokens = 0
            for c in converted_chunks:
                if hasattr(c, "metadata") and "token_count" in c.metadata:
                    total_tokens += c.metadata["token_count"]

            stats = {
                "total_chunks": len(converted_chunks),
                "total_tokens": total_tokens,
            }

            return ChunkResult(
                chunks=converted_chunks,
                metadata=metadata,
                stats=stats,
                image_paths=image_paths,
            )

        except Exception as e:
            logger.error(f"Docling chunking failed: {e}")
            return EmptyChunkResult(str(e), chunker_type="docling")

    def _convert_chunks(self, chunks: List, doc: Any) -> List[DoclingChunkElement]:
        """
        Convert Docling chunks to compatible element format.

        Args:
            chunks: Raw Docling chunks
            doc: Original document for contextualization

        Returns:
            List of converted chunk elements
        """
        converted = []

        for i, chunk in enumerate(chunks):
            # Get contextualized text (includes heading context)
            try:
                text = self.chunker.contextualize(chunk=chunk)
            except Exception as e:
                # Fallback to raw text
                logger.debug(f"Contextualize failed for chunk {i}: {e}")
                text = chunk.text if hasattr(chunk, "text") else str(chunk)

            # Extract metadata
            metadata = {
                "chunk_index": i,
                "headings": (
                    list(chunk.meta.headings) if hasattr(chunk.meta, "headings") else []
                ),
                "source": "docling",
            }

            # Extract page number and bbox from doc_items provenance
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                doc_items = chunk.meta.doc_items
                if doc_items:
                    first_item = doc_items[0]
                    # Extract element type/label from doc_items
                    if hasattr(first_item, "label"):
                        metadata["element_type"] = str(first_item.label)
                    if hasattr(first_item, "prov") and first_item.prov:
                        prov = (
                            first_item.prov[0]
                            if isinstance(first_item.prov, list)
                            else first_item.prov
                        )
                        # Docling uses 'page_no' not 'page'
                        if hasattr(prov, "page_no"):
                            metadata["page_number"] = prov.page_no
                        elif hasattr(prov, "page"):
                            metadata["page_number"] = prov.page
                        if hasattr(prov, "bbox"):
                            metadata["bbox"] = {
                                "left": prov.bbox.l,
                                "top": prov.bbox.t,
                                "right": prov.bbox.r,
                                "bottom": prov.bbox.b,
                            }

            # Set chunk_type based on content
            metadata["chunk_type"] = "hybrid"  # Docling HybridChunker output

            # Token count
            try:
                metadata["token_count"] = self.tokenizer.count_tokens(text)
            except Exception as e:
                logger.debug(f"Token counting failed for chunk {i}: {e}")
                metadata["token_count"] = len(text.split())  # Fallback word count

            converted.append(DoclingChunkElement(text=text, metadata=metadata))

        return converted

    def chunk_elements(
        self,
        elements: List[Any],
        language: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        **kwargs,
    ) -> ChunkResult:
        """
        Fallback method for compatibility with existing interface.

        NOTE: Docling chunker works best with DoclingDocument directly.
        This method creates simple chunks from elements.

        Args:
            elements: List of elements (from Docling or other sources)
            language: Document language
            image_paths: Image paths

        Returns:
            ChunkResult
        """
        if not elements:
            return EmptyChunkResult("No elements provided", chunker_type="docling")

        # Extract text from elements
        chunks = []
        for i, elem in enumerate(elements):
            text = elem.text if hasattr(elem, "text") else str(elem)
            if text and text.strip():
                elem_metadata = {
                    "chunk_index": i,
                    "source": "docling_fallback",
                    "language": language,
                }
                # Preserve existing metadata
                if hasattr(elem, "metadata"):
                    if isinstance(elem.metadata, dict):
                        elem_metadata.update(elem.metadata)
                    elif hasattr(elem.metadata, "__dict__"):
                        elem_metadata.update(elem.metadata.__dict__)

                chunks.append(DoclingChunkElement(text=text, metadata=elem_metadata))

        return ChunkResult(
            chunks=chunks,
            metadata={"chunker_type": "docling_fallback"},
            stats={"total_chunks": len(chunks)},
            image_paths=image_paths or [],
        )
