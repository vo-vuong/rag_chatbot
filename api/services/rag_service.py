"""
RAG Service - Retrieval-Augmented Generation orchestration.

Extracted from ChatMainUI to enable API-based RAG operations.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from backend.embeddings.embedding_strategy import EmbeddingStrategy
from backend.models import ChunkElement, ImageElement
from backend.reranking import CohereReranker
from backend.routing import QueryRouter
from backend.vector_db.qdrant_manager import QdrantManager
from config.constants import (
    COHERE_API_KEY,
    COHERE_MODEL,
    DEFAULT_IMAGE_NUM_RETRIEVAL,
    DEFAULT_IMAGE_SCORE_THRESHOLD,
    RERANK_TOP_K,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Text search result container."""

    chunks: List[ChunkElement]  # Typed ChunkElement objects
    route: str  # "text_only" or "image_only"
    reasoning: str


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

        # Initialize Reranker
        self._reranker = None
        if COHERE_API_KEY:
            try:
                self._reranker = CohereReranker(
                    api_key=COHERE_API_KEY, model=COHERE_MODEL
                )
                logger.info(f"Cohere Reranker initialized with model: {COHERE_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere Reranker: {e}")

    def route_query(self, query: str) -> Tuple[str, str]:
        """
        Classify query route.

        Returns:
            Tuple of (route, reasoning)
        """
        # classification = self._router.classify(query)
        # return classification.route, classification.reasoning
        # Sử dụng tạm để đánh giá trên tập text dataset.
        return "text_only", "default"

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> List[ChunkElement]:
        """
        Search text collection.

        Returns:
            List of ChunkElement objects with content, score, source_file, metadata, point_id
        """
        # Determine retrieval top_k (increase recall for reranking)
        retrieval_k = max(top_k, RERANK_TOP_K) if self._reranker else top_k

        query_embedding = self._embedding.embed_query(query)
        results = self._text_manager.search(
            query_vector=query_embedding,
            top_k=retrieval_k,
            score_threshold=score_threshold,
        )

        chunks = [
            ChunkElement.from_qdrant_payload(
                r["payload"],
                r["score"],
                point_id=r["id"],  # Already an integer from Qdrant
            )
            for r in results
        ]

        # Apply Reranking if enabled and available
        if self._reranker and chunks:
            try:
                chunks = self._reranker.rerank(query, chunks, top_k=top_k)
            except Exception as e:
                logger.error(f"Reranking failed, falling back to original results: {e}")
                # Fallback to original top_k if reranking fails
                # Note: User requested to raise error, but in a production service
                # passing the error up might crash the request completely.
                # However, following the user's specific instruction:
                # "Fallback Strategy: Báo lỗi cho người dùng"
                # The reranker itself raises RuntimeError, so we let it propagate if we want to stop.
                # But here I caught it to log. Re-raising it now.
                raise e

        # Ensure we don't return more than requested if reranker was skipped/not used
        return chunks[:top_k]

    def search_images(
        self,
        query: str,
        top_k: int = DEFAULT_IMAGE_NUM_RETRIEVAL,
        score_threshold: float = DEFAULT_IMAGE_SCORE_THRESHOLD,
    ) -> List[ImageElement]:
        """
        Search image collection.

        Returns:
            List of ImageElement objects
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
            ImageElement.from_qdrant_payload(r["payload"], r["score"]) for r in results
        ]

    def search_with_routing(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        image_score_threshold: float = DEFAULT_IMAGE_SCORE_THRESHOLD,
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
            images = self.search_images(
                query,
                top_k=DEFAULT_IMAGE_NUM_RETRIEVAL,
                score_threshold=image_score_threshold,
            )
            # Convert ImageElement to ChunkElement for unified interface
            chunks = [
                ChunkElement(
                    content=img.content,
                    score=img.score,
                    source_file=img.source_file,
                    page_number=img.page_number,
                    element_type="image",
                )
                for img in images
            ]
            return SearchResult(chunks=chunks, route=route, reasoning=reasoning)
