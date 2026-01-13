"""
RAG Service - Retrieval-Augmented Generation orchestration.

Extracted from ChatMainUI to enable API-based RAG operations.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from backend.embeddings.embedding_strategy import EmbeddingStrategy
from backend.routing import QueryRouter
from backend.vector_db.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Text search result container."""

    chunks: List[Tuple[str, float]]  # (text, score)
    route: str  # "text_only" or "image_only"
    reasoning: str


@dataclass
class ImageSearchResult:
    """Image search result container."""

    caption: str
    image_path: str
    score: float
    page_number: Optional[int]
    source_document: str


class RAGService:
    """
    RAG retrieval service - extracted from ChatMainUI.

    Handles query routing and vector search operations.
    """

    def __init__(
        self,
        router: QueryRouter,
        text_manager: QdrantManager,
        image_manager: Optional[QdrantManager],
        embedding: EmbeddingStrategy,
    ):
        self._router = router
        self._text_manager = text_manager
        self._image_manager = image_manager
        self._embedding = embedding

    def route_query(self, query: str) -> Tuple[str, str]:
        """
        Classify query route.

        Returns:
            Tuple of (route, reasoning)
        """
        classification = self._router.classify(query)
        return classification.route, classification.reasoning

    def search_text(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Search text collection.

        Returns:
            List of (chunk_text, score) tuples
        """
        query_embedding = self._embedding.embed_query(query)
        results = self._text_manager.search(
            query_vector=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return [
            (r["payload"].get("chunk", ""), r["score"])
            for r in results
        ]

    def search_images(
        self,
        query: str,
        top_k: int = 1,
        score_threshold: float = 0.6,
    ) -> List[ImageSearchResult]:
        """
        Search image collection.

        Returns:
            List of ImageSearchResult objects
        """
        if not self._image_manager:
            return []

        query_embedding = self._embedding.embed_query(query)
        results = self._image_manager.search(
            query_vector=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return [
            ImageSearchResult(
                caption=r["payload"].get("chunk", ""),
                image_path=r["payload"].get("image_path", ""),
                score=r["score"],
                page_number=r["payload"].get("page_number"),
                source_document=r["payload"].get("source_document", ""),
            )
            for r in results
        ]

    def search_with_routing(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.5,
        image_score_threshold: float = 0.6,
    ) -> SearchResult:
        """
        Search with automatic routing.

        Routes query to text or image collection based on content.
        """
        route, reasoning = self.route_query(query)
        logger.info(f"Query routed to: {route} (reason: {reasoning})")

        if route == "text_only":
            chunks = self.search_text(query, top_k, score_threshold)
            return SearchResult(chunks=chunks, route=route, reasoning=reasoning)
        else:
            images = self.search_images(query, top_k=1, score_threshold=image_score_threshold)
            chunks = [(img.caption, img.score) for img in images]
            return SearchResult(chunks=chunks, route=route, reasoning=reasoning)
