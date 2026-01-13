"""Dependency injection for FastAPI."""

from typing import Optional

from api.config import get_settings, Settings
from api.services.chat_service import ChatService
from api.services.rag_service import RAGService
from api.services.session_service import SessionService
from backend.embeddings.openai_embeddings import OpenAIEmbeddingStrategy
from backend.llms.openai_llm import OpenAILLM
from backend.routing import QueryRouter
from backend.vector_db.qdrant_manager import QdrantManager

# Singleton instances
_session_service: Optional[SessionService] = None
_rag_service: Optional[RAGService] = None
_chat_service: Optional[ChatService] = None


def get_session_service() -> SessionService:
    """Get or create SessionService singleton."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service


def get_llm(settings: Settings = None) -> OpenAILLM:
    """Create LLM instance."""
    settings = settings or get_settings()
    return OpenAILLM(
        api_key=settings.openai_api_key,
        model_version=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def get_embedding(settings: Settings = None) -> OpenAIEmbeddingStrategy:
    """Create embedding strategy."""
    settings = settings or get_settings()
    return OpenAIEmbeddingStrategy(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )


def get_text_manager(settings: Settings = None) -> QdrantManager:
    """Get QdrantManager for text collection."""
    settings = settings or get_settings()
    return QdrantManager(
        collection_name=settings.text_collection,
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def get_image_manager(settings: Settings = None) -> Optional[QdrantManager]:
    """Get QdrantManager for image collection (may not exist)."""
    settings = settings or get_settings()
    try:
        manager = QdrantManager(
            collection_name=settings.image_collection,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        if manager.collection_exists():
            return manager
    except Exception:
        pass
    return None


def get_query_router(settings: Settings = None) -> QueryRouter:
    """Create QueryRouter."""
    settings = settings or get_settings()
    return QueryRouter(openai_api_key=settings.openai_api_key)


def get_rag_service() -> RAGService:
    """Get or create RAGService singleton."""
    global _rag_service
    if _rag_service is None:
        settings = get_settings()
        _rag_service = RAGService(
            router=get_query_router(settings),
            text_manager=get_text_manager(settings),
            image_manager=get_image_manager(settings),
            embedding=get_embedding(settings),
        )
    return _rag_service


def get_chat_service() -> ChatService:
    """Get or create ChatService singleton."""
    global _chat_service
    if _chat_service is None:
        settings = get_settings()
        _chat_service = ChatService(
            llm=get_llm(settings),
            rag_service=get_rag_service(),
            session_service=get_session_service(),
        )
    return _chat_service
