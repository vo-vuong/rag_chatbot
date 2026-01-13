"""Streamlit API client for FastAPI backend."""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Iterator, List, Optional

import httpx
import streamlit as st

from config.constants import API_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Response from chat API."""

    response: str
    route: Optional[str]
    route_reasoning: Optional[str]
    retrieved_chunks: List[tuple]  # [(text, score), ...]
    image_paths: List[str]
    image_captions: List[str]


@dataclass
class StreamEvent:
    """SSE event from streaming API."""

    event: str  # "route", "context", "token", "done", "error"
    data: dict


class StreamlitAPIClient:
    """HTTP client for FastAPI backend."""

    def __init__(self, base_url: str = API_BASE_URL):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)
        self._session_id = self._get_or_create_session_id()

    def _get_or_create_session_id(self) -> str:
        """Get session ID from st.session_state or create new one."""
        if "api_session_id" not in st.session_state:
            st.session_state.api_session_id = str(uuid.uuid4())
        return st.session_state.api_session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self._client.get(f"{self._base_url}/api/v1/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            return False

    def chat(
        self,
        query: str,
        mode: str = "rag",
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> APIResponse:
        """Send chat query to API (synchronous)."""
        response = self._client.post(
            f"{self._base_url}/api/v1/chat/query",
            json={
                "query": query,
                "session_id": self._session_id,
                "mode": mode,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        )
        response.raise_for_status()
        data = response.json()

        return APIResponse(
            response=data["response"],
            route=data.get("route"),
            route_reasoning=data.get("route_reasoning"),
            retrieved_chunks=[
                (c["text"], c["score"]) for c in data.get("retrieved_chunks", [])
            ],
            image_paths=[img["image_path"] for img in data.get("images", [])],
            image_captions=[img["caption"] for img in data.get("images", [])],
        )

    def chat_stream(
        self,
        query: str,
        mode: str = "rag",
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> Iterator[StreamEvent]:
        """Send chat query and stream response (SSE)."""
        with self._client.stream(
            "POST",
            f"{self._base_url}/api/v1/chat/query/stream",
            json={
                "query": query,
                "session_id": self._session_id,
                "mode": mode,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                        yield StreamEvent(
                            event=payload.get("event", "unknown"),
                            data=payload.get("data", {}),
                        )
                    except json.JSONDecodeError:
                        continue

    def search(
        self,
        query: str,
        collection_type: str = "text",
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> dict:
        """Search RAG collections without generation."""
        response = self._client.post(
            f"{self._base_url}/api/v1/rag/search",
            json={
                "query": query,
                "collection_type": collection_type,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close HTTP client."""
        self._client.close()


def get_api_client() -> StreamlitAPIClient:
    """Get or create API client singleton."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = StreamlitAPIClient()
    return st.session_state.api_client
