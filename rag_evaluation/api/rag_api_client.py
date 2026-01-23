"""
RAG API client for retrieval evaluation.

Provides a clean interface for querying the RAG API during evaluation.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.constants import API_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class ChatQueryResult:
    """Result from a chat query containing response and contexts."""

    response: str
    retrieved_contexts: List[str]
    retrieved_ids: List[int]


class RAGAPIClient:
    """
    Client for querying RAG API during evaluation.

    Provides methods for searching the RAG system and retrieving
    document IDs for comparison with ground truth.

    Attributes:
        base_url: Base URL of the RAG API
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout: int = 30,
    ):
        """
        Initialize the RAG API client.

        Args:
            base_url: Base URL for the RAG API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        collection_type: str = "text",
    ) -> List[int]:
        """
        Search using RAG API and return retrieved point IDs.

        Args:
            query: Search query text
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score threshold
            collection_type: Type of collection to search ("text" or "image")

        Returns:
            List of retrieved point IDs in ranked order
        """
        url = f"{self.base_url}/api/v1/rag/search"
        payload = {
            "query": query,
            "collection_type": collection_type,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            point_ids = [
                r.get("point_id")
                for r in results
                if r.get("point_id") is not None
            ]
            return point_ids

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for query '{query[:50]}...': {e}")
            return []

    def health_check(self) -> bool:
        """
        Check if the RAG API is available.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/api/v1/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def chat_query(
        self,
        query: str,
        session_id: str = "eval-session",
        top_k: int = 5,
        score_threshold: float = 0.0,
        mode: str = "rag",
    ) -> Optional[ChatQueryResult]:
        """
        Query the chat endpoint to get LLM response and retrieved contexts.

        Args:
            query: User query text
            session_id: Session identifier for the chat
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            mode: Query mode ("rag" or "llm_only")

        Returns:
            ChatQueryResult with response and contexts, or None if failed
        """
        url = f"{self.base_url}/api/v1/chat/query"
        payload = {
            "query": query,
            "session_id": session_id,
            "mode": mode,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Extract response text
            llm_response = data.get("response", "")

            # Extract contexts and IDs from retrieved chunks
            chunks = data.get("retrieved_chunks", [])
            contexts = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]
            point_ids = [
                chunk.get("point_id")
                for chunk in chunks
                if chunk.get("point_id") is not None
            ]

            return ChatQueryResult(
                response=llm_response,
                retrieved_contexts=contexts,
                retrieved_ids=point_ids,
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Chat query failed for '{query[:50]}...': {e}")
            return None
