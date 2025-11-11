import logging

from backend.embeddings.embedding_strategy import EmbeddingStrategy
from backend.embeddings.local_embeddings import LocalEmbeddingStrategy
from backend.embeddings.openai_embeddings import OpenAIEmbeddingStrategy
from config.constants import OPENAI_DEFAULT_EMBEDDING_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """
    Factory class for creating embedding strategy instances.

    This factory simplifies the creation of different embedding strategies
    and provides a centralized place for strategy instantiation logic.
    """

    @staticmethod
    def create_online_embedding(
        provider: str, api_key: str, model: str = OPENAI_DEFAULT_EMBEDDING_MODEL
    ) -> EmbeddingStrategy:
        """
        Create an online embedding strategy (OpenAI, Google, etc.).

        Args:
            provider: Provider name ("openai", "google", etc.)
            api_key: API key for authentication
            model: Optional specific model name

        Returns:
            EmbeddingStrategy instance for the specified provider

        Raises:
            ValueError: If provider is not supported
            ImportError: If required libraries are not installed
        """

        if provider.lower() == "openai":
            return OpenAIEmbeddingStrategy(api_key=api_key, model=model)
        else:
            raise ValueError(
                f"Unsupported online embedding provider: {provider}. "
                f"Supported providers: openai"
            )

    @staticmethod
    def create_local_embedding(model_name: str) -> EmbeddingStrategy:
        """
        Create a local embedding strategy (Sentence Transformers, etc.).

        Args:
            model_name: Name of the local model to use

        Returns:
            EmbeddingStrategy instance for local embeddings

        Raises:
            NotImplementedError: Local embeddings are not yet implemented
        """

        logger.info("Creating local embedding strategy with model: %s", model_name)
        return LocalEmbeddingStrategy(model_name=model_name)
