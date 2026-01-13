"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.routers import chat, health, rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    logger.info("Starting RAG Chatbot API...")
    settings = get_settings()
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"LLM: {settings.llm_model}")
    yield
    logger.info("Shutting down RAG Chatbot API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )

    # CORS middleware for Streamlit
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router, prefix=settings.api_prefix)
    app.include_router(rag.router, prefix=settings.api_prefix)
    app.include_router(health.router, prefix=settings.api_prefix)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
