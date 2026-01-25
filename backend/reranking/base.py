from abc import ABC, abstractmethod
from typing import List

from backend.models import ChunkElement


class RerankerStrategy(ABC):
    """
    Abstract base class for reranking strategies.
    """

    @abstractmethod
    def rerank(
        self, query: str, documents: List[ChunkElement], top_k: int
    ) -> List[ChunkElement]:
        """
        Rerank a list of documents based on the query.

        Args:
            query: The search query.
            documents: List of ChunkElement objects to rerank.
            top_k: Number of documents to return after reranking.

        Returns:
            List of reranked ChunkElement objects (sorted by new score).
        """
        pass
