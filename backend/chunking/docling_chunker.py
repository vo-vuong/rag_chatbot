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
    """Chunk element compatible with existing pipeline.

    Provides unified naming aliases for consistency with ChunkElement:
    - content: alias for text
    - source_file: from metadata
    - element_type: from metadata (default 'text')
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Unified content field (alias for text)."""
        return self.text

    @property
    def source_file(self) -> Optional[str]:
        """Source document filename."""
        return self.metadata.get("source_file")

    @property
    def element_type(self) -> str:
        """Element type: text, table, list, etc."""
        return self.metadata.get("element_type", "text")

    @property
    def page_number(self) -> Optional[int]:
        """Page number from metadata."""
        return self.metadata.get("page_number")

    def to_chunk_element(self) -> "ChunkElement":
        """Convert to unified ChunkElement model.

        Returns:
            ChunkElement with unified field naming
        """
        from backend.models import ChunkElement

        return ChunkElement(
            content=self.text,
            source_file=self.source_file or "Unknown",
            page_number=self.page_number,
            element_type=self.element_type,
            metadata=self.metadata,
        )


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
                    "max_tokens": 1024,  # Optimal chunk size for RAG retrieval
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
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
        except ImportError as e:
            logger.error(f"Docling imports failed: {e}")
            raise ImportError(
                "Install docling-core[chunking-openai]: "
                "pip install docling-core[chunking-openai]"
            ) from e

        tokenizer_type = self.config.get("tokenizer_type", "openai")
        tokenizer_model = self.config.get("tokenizer_model", "text-embedding-3-small")
        max_tokens = self.config.get("max_tokens", 1024)
        merge_peers = self.config.get("merge_peers", True)

        if tokenizer_type == "openai":
            try:
                enc = tiktoken.encoding_for_model(tokenizer_model)
            except KeyError as e:
                raise ValueError(
                    f"Tokenizer for model '{tokenizer_model}' not found in tiktoken. "
                    f"Ensure the model name is correct. Error: {e}"
                ) from e

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
        source_file: str = "Unknown",
    ) -> ChunkResult:
        """
        Chunk Docling document using HybridChunker.

        Args:
            doc: Docling DoclingDocument
            image_paths: Associated image paths
            source_file: Source document filename for metadata

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

            # Convert to compatible format with source_file
            converted_chunks = self._convert_chunks(chunks, doc, source_file)

            # Apply post-processing overlap
            overlap_tokens = self.config.get("overlap_tokens", 0)
            if overlap_tokens > 0:
                converted_chunks = self._add_overlap(converted_chunks, overlap_tokens)

            # Build metadata
            metadata = {
                "chunker_type": "docling_hybrid",
                "tokenizer_type": self.config.get("tokenizer_type", "openai"),
                "tokenizer_model": self.config.get(
                    "tokenizer_model", "text-embedding-3-small"
                ),
                "max_tokens": self.config.get("max_tokens", 1024),
                "overlap_tokens": overlap_tokens,
                "original_chunk_count": len(chunks),
            }

            # Build stats
            total_tokens = 0
            chunks_with_overlap = 0
            for c in converted_chunks:
                if hasattr(c, "metadata") and "token_count" in c.metadata:
                    total_tokens += c.metadata["token_count"]
                if hasattr(c, "metadata") and c.metadata.get("has_overlap"):
                    chunks_with_overlap += 1

            stats = {
                "total_chunks": len(converted_chunks),
                "total_tokens": total_tokens,
                "chunks_with_overlap": chunks_with_overlap,
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

    def _format_headings(self, headings: List[str]) -> str:
        """Format headings with bold highlighting for consistent display.

        Args:
            headings: List of heading strings in hierarchical order

        Returns:
            Formatted string: "Section: **[Heading 1]** > **[Heading 2]** > ..."
        """
        if not headings:
            return ""
        headings_str = " > ".join(f"**[{h}]**" for h in headings)
        return f"Section: {headings_str}"

    def _convert_chunks(
        self, chunks: List, doc: Any, source_file: str = "Unknown"
    ) -> List[DoclingChunkElement]:
        """
        Convert Docling chunks to compatible element format.

        Args:
            chunks: Raw Docling chunks
            doc: Original document for contextualization
            source_file: Source document filename

        Returns:
            List of converted chunk elements
        """
        converted = []

        for i, chunk in enumerate(chunks):
            # Extract headings first (needed for custom contextualization)
            headings = []
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "headings"):
                if chunk.meta.headings is not None:
                    headings = list(chunk.meta.headings)

            # Get raw chunk text
            raw_text = chunk.text if hasattr(chunk, "text") else str(chunk)

            # Build contextualized text with highlighted headings
            # Format: "Section: [Heading 1] > [Heading 2]\n\n<chunk text>"
            headings_prefix = self._format_headings(headings)
            if headings_prefix:
                text = f"{headings_prefix}\n\n{raw_text}"
            else:
                text = raw_text

            metadata = {
                "document_chunk_index": i,
                "headings": headings,
                "source": "docling",
                "source_file": source_file,
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

    def _add_overlap(
        self, chunks: List[DoclingChunkElement], overlap_tokens: int
    ) -> List[DoclingChunkElement]:
        """Add overlap from previous chunk to current chunk for better context.

        Post-processing overlap since Docling HybridChunker lacks native support.

        Args:
            chunks: List of DoclingChunkElement from HybridChunker
            overlap_tokens: Number of tokens to overlap from previous chunk

        Returns:
            List of chunks with overlap text prepended (except first chunk)
        """
        if overlap_tokens <= 0 or len(chunks) < 2:
            return chunks

        result = [chunks[0]]  # First chunk unchanged

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            curr_chunk = chunks[i]

            # Get last N tokens from previous chunk
            try:
                prev_token_count = self.tokenizer.count_tokens(prev_text)
                if prev_token_count <= overlap_tokens:
                    # Previous chunk smaller than overlap, use entire text
                    overlap_text = prev_text
                else:
                    # Extract approximate overlap by character ratio
                    char_ratio = overlap_tokens / prev_token_count
                    overlap_chars = int(len(prev_text) * char_ratio)
                    overlap_text = prev_text[-overlap_chars:]
            except Exception as e:
                logger.warning(f"Overlap extraction failed for chunk {i}: {e}")
                result.append(curr_chunk)
                continue

            # Prepend overlap with separator
            new_text = f"[...] {overlap_text.strip()} [...]\n\n{curr_chunk.text}"

            # Update metadata
            new_metadata = {**curr_chunk.metadata}
            new_metadata["has_overlap"] = True
            new_metadata["overlap_tokens"] = overlap_tokens

            # Re-count tokens
            try:
                new_metadata["token_count"] = self.tokenizer.count_tokens(new_text)
            except Exception:
                new_metadata["token_count"] = len(new_text.split())

            result.append(DoclingChunkElement(text=new_text, metadata=new_metadata))

        return result
