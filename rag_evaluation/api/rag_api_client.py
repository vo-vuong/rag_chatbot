"""
RAG API client for retrieval evaluation.

Provides a clean interface for querying the RAG API during evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import List

import requests

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.constants import API_BASE_URL

logger = logging.getLogger(__name__)


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
