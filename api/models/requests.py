"""
Request Models - Pydantic models for API requests.
"""

from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat request model."""

    query: str
    session_id: str
    mode: Literal["rag", "llm_only"] = "rag"
    top_k: int = 3
    score_threshold: float = 0.5


class RAGSearchRequest(BaseModel):
    """RAG search request model."""

    query: str
    collection_type: Literal["text", "image"] = "text"
    top_k: int = 3
    score_threshold: float = 0.5
