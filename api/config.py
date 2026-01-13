"""API configuration from environment."""

from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

from config.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_NUM_RETRIEVAL,
    DEFAULT_SCORE_THRESHOLD,
    IMAGE_COLLECTION_NAME,
    OPENAI_API_KEY,
    OPENAI_DEFAULT_EMBEDDING_DIMENSION,
    OPENAI_DEFAULT_EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    TEXT_COLLECTION_NAME,
)


class Settings(BaseSettings):
    """API settings from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    api_title: str = "RAG Chatbot API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # OpenAI
    openai_api_key: str = OPENAI_API_KEY

    # Qdrant
    qdrant_host: str = QDRANT_HOST.replace("http://", "").replace("https://", "")
    qdrant_port: int = QDRANT_PORT

    # Collections
    text_collection: str = TEXT_COLLECTION_NAME
    image_collection: str = IMAGE_COLLECTION_NAME

    # Embedding
    embedding_model: str = OPENAI_DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = OPENAI_DEFAULT_EMBEDDING_DIMENSION

    # LLM
    llm_model: str = DEFAULT_LLM_MODEL
    llm_temperature: float = 0.2

    # Retrieval defaults
    default_top_k: int = DEFAULT_NUM_RETRIEVAL
    default_score_threshold: float = DEFAULT_SCORE_THRESHOLD


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
