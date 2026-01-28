"""
Response collector for RAG evaluation.

Collects API responses and saves them to JSON files for later evaluation.
Supports two modes:
- chat: Collects from /chat/query (response + retrieved_contexts)
- search: Collects from /rag/search (retrieved_contexts only)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

from rag_evaluation.api.rag_api_client import RAGAPIClient
from rag_evaluation.data.data_loader import TestDataLoader

logger = logging.getLogger(__name__)

DEFAULT_TEST_DATA_PATH = (
    Path(__file__).parent.parent / "prepare_testing_data" / "qr_smartphone_dataset.xlsx"
)
DEFAULT_RESPONSES_DIR = Path(__file__).parent.parent / "responses"


class ResponseCollector:
    """
    Collects API responses and saves to JSON for evaluation.

    Supports two collection modes:
    - chat: Uses /chat/query API (gets LLM response + contexts)
    - search: Uses /rag/search API (gets contexts only)
    """

    def __init__(
        self,
        test_data_path: Union[str, Path] = DEFAULT_TEST_DATA_PATH,
        api_base_url: Optional[str] = None,
        output_dir: Union[str, Path] = DEFAULT_RESPONSES_DIR,
        limit: Optional[int] = None,
        delay: float = 7.0,
    ):
        """
        Initialize the response collector.

        Args:
            test_data_path: Path to test data Excel file
            api_base_url: Base URL for RAG API
            output_dir: Directory for output JSON files
            limit: Optional limit on number of queries
            delay: Time to sleep between queries (for rate limiting)
        """
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.delay = delay

        self.data_loader = TestDataLoader(self.test_data_path, limit=limit)

        if api_base_url:
            self.api_client = RAGAPIClient(base_url=api_base_url)
        else:
            self.api_client = RAGAPIClient()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        mode: Literal["chat", "search"] = "chat",
        top_k: int = 5,
        score_threshold: float = 0.0,
        output_path: Optional[Path] = None,
        verbose: bool = False,
    ) -> Path:
        """
        Collect responses from API and save to JSON file.

        Args:
            mode: Collection mode - "chat" or "search"
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            output_path: Custom output file path
            verbose: Print detailed progress

        Returns:
            Path to the saved JSON file
        """
        logger.info(f"Starting response collection in '{mode}' mode")
        logger.info(f"top_k={top_k}, score_threshold={score_threshold}")

        total_queries = self.data_loader.get_valid_query_count()
        responses = []
        processed = 0
        failed = 0

        for idx, query, ground_truth_ids, metadata in self.data_loader.iter_queries():
            if self.delay > 0 and processed > 0:
                time.sleep(self.delay)

            processed += 1
            ground_truth_answer = metadata.get("ground_truth_answer", "")

            if mode == "chat":
                result = self._collect_chat_response(query, top_k, score_threshold)
            else:
                result = self._collect_search_response(query, top_k, score_threshold)

            if result is None:
                logger.warning(f"Query {idx} failed: '{query[:50]}...'")
                failed += 1
                continue

            response_entry = {
                "query_index": idx,
                "query": query,
                "response": result.get("response", ""),
                "retrieved_contexts": result.get("retrieved_contexts", []),
                "ground_truth_answer": ground_truth_answer,
                "ground_truth_ids": ground_truth_ids,
            }
            responses.append(response_entry)

            if verbose:
                ctx_count = len(response_entry["retrieved_contexts"])
                logger.info(
                    f"[{processed}/{total_queries}] Query: '{query[:40]}...' | "
                    f"Contexts: {ctx_count}"
                )
            elif processed % 5 == 0:
                logger.info(f"Processed {processed}/{total_queries} queries...")

        output_data = {
            "metadata": {
                "mode": mode,
                "created_at": datetime.now().isoformat(),
                "test_data_path": str(self.test_data_path),
                "top_k": top_k,
                "score_threshold": score_threshold,
                "total_queries": len(responses),
                "failed_queries": failed,
                "note": "retrieved_contexts may exceed top_k in chat mode due to agent auto-retrieval",
            },
            "responses": responses,
        }

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{mode}_responses_k{top_k}_{timestamp}.json"
            output_path = self.output_dir / filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(responses)} responses to: {output_path}")
        return output_path

    def _collect_chat_response(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> Optional[dict]:
        """Collect response from /chat/query API."""
        try:
            result = self.api_client.chat_query(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            if result is None:
                return None

            return {
                "response": result.response,
                "retrieved_contexts": result.retrieved_contexts,
            }
        except Exception as e:
            logger.error(f"Chat API error: {e}")
            return None

    def _collect_search_response(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> Optional[dict]:
        """Collect response from /rag/search API."""
        try:
            result = self.api_client.search_with_contexts(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            if result is None:
                return None

            return {
                "response": "",
                "retrieved_contexts": result.retrieved_contexts,
            }
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return None


def load_responses(file_path: Union[str, Path]) -> dict:
    """
    Load responses from JSON file.

    Args:
        file_path: Path to JSON response file

    Returns:
        Dictionary with metadata and responses

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Response file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "metadata" not in data or "responses" not in data:
        raise ValueError(
            "Invalid response file format. Expected 'metadata' and 'responses' keys."
        )

    return data
