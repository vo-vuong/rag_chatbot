"""
Embedding Strategy Pattern for RAG Chatbot.

This module implements the Strategy pattern for different embedding providers,
allowing easy switching between online (OpenAI) and local (Sentence Transformers)
embedding models.

Usage:
    from backend.embeddings.embedding_strategy import EmbeddingFactory

    # Create OpenAI embedding strategy
    strategy = EmbeddingFactory.create_online_embedding("openai", api_key)
    embeddings = strategy.embed_texts(["Hello world", "Test document"])

    # Future: Create local embedding strategy
    strategy = EmbeddingFactory.create_local_embedding("all-MiniLM-L6-v2")
"""

import logging
from abc import ABC, abstractmethod
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingStrategy(ABC):
    """
    Abstract base class for embedding strategies.

    This class defines the interface that all embedding providers must implement,
    ensuring consistent behavior across different embedding models.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            Exception: If embedding generation fails
        """

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector (list of floats)

        Raises:
            Exception: If embedding generation fails
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimensionality of embeddings produced by this strategy.

        Returns:
            Integer dimension size (e.g., 1536 for OpenAI, 384 for MiniLM)
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this embedding strategy is properly configured and available.

        Returns:
            True if the strategy can be used, False otherwise
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name/identifier of the embedding model.

        Returns:
            Model name string
        """
