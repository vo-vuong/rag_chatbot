"""
OpenAI Embeddings Strategy Implementation.

This module provides the OpenAI embedding implementation using the official
OpenAI Python SDK. It supports all OpenAI embedding models including
text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.

Usage:
    from backend.embeddings.openai_embeddings import OpenAIEmbeddingStrategy

    strategy = OpenAIEmbeddingStrategy(
        api_key="your-api-key",
        model="text-embedding-3-small"
    )

    embeddings = strategy.embed_texts(["Hello", "World"])
    query_embedding = strategy.embed_query("Search query")
"""

import logging
from typing import List

from openai import OpenAI, OpenAIError

from backend.embeddings.embedding_strategy import EmbeddingStrategy
from config.constants import (
    OPENAI_DEFAULT_EMBEDDING_DIMENSION,
    OPENAI_DEFAULT_EMBEDDING_MODEL,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    """
    OpenAI embedding strategy implementation.

    This strategy uses OpenAI's embedding API to generate embeddings.
    It supports batching for efficient processing of multiple texts.

    Attributes:
        client: OpenAI client instance
        model: Name of the OpenAI embedding model
        dimension: Embedding dimension size
    """

    # Model configurations
    MODEL_CONFIGS = {
        "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8191},
        "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8191},
    }

    def __init__(self, api_key: str, model: str, timeout: int = 60):
        """
        Initialize OpenAI embedding strategy.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small)
            timeout: Request timeout in seconds

        Raises:
            ValueError: If API key is empty or model is not supported
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.model = model or OPENAI_DEFAULT_EMBEDDING_MODEL

        # Validate model
        if self.model not in self.MODEL_CONFIGS:
            logger.warning(
                "Model %s not in known configs. Supported: %s",
                self.model,
                list(self.MODEL_CONFIGS.keys()),
            )

        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=api_key, timeout=timeout)
            logger.info(f"OpenAI client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

        # Set dimension
        if self.model in self.MODEL_CONFIGS:
            self.dimension = self.MODEL_CONFIGS[self.model]["dimension"]
        else:
            self.dimension = OPENAI_DEFAULT_EMBEDDING_DIMENSION

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            OpenAIError: If API call fails
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        try:
            logger.info(f"Embedding {len(texts)} texts with model: {self.model}")

            # Call OpenAI API
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]

            logger.info(
                f"Successfully generated {len(embeddings)} embeddings "
                f"with dimension {len(embeddings[0])}"
            )

            return embeddings

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {str(e)}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector

        Raises:
            OpenAIError: If API call fails
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            logger.debug(f"Embedding query: {query[:100]}...")

            # Call OpenAI API with single text
            response = self.client.embeddings.create(
                input=query.strip(), model=self.model
            )

            embedding = response.data[0].embedding

            logger.debug("Successfully embedded query")

            return embedding

        except OpenAIError as e:
            logger.error(f"OpenAI API error for query: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query embedding: {str(e)}")
            raise

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension size
        """
        return self.dimension

    def is_available(self) -> bool:
        """
        Check if OpenAI embeddings are available.

        Returns:
            True if client is initialized and can make requests
        """
        try:
            # Test with a simple embedding call
            test_response = self.client.embeddings.create(
                input="test", model=self.model
            )
            return test_response is not None
        except Exception as e:
            logger.warning(f"OpenAI embeddings not available: {str(e)}")
            return False

    def get_model_name(self) -> str:
        """
        Get the model name.

        Returns:
            Model name string
        """
        return self.model

    def get_max_tokens(self) -> int:
        """
        Get maximum tokens supported by the model.

        Returns:
            Maximum token count
        """
        if self.model in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[self.model]["max_tokens"]
        return 8191  # Default for most OpenAI embedding models
