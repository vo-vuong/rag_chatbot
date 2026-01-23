"""
Base interface for generation metrics.

Defines the abstract base class for generation quality metrics like Faithfulness,
Answer Relevancy, etc. These metrics evaluate the quality of LLM-generated responses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class GenerationQueryResult:
    """Result for a single query evaluation (generation metrics)."""

    query_index: int
    query: str
    response: str
    retrieved_contexts: List[str]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        # Extract ground_truth_answer from metadata for better column ordering
        ground_truth_answer = self.metadata.pop("ground_truth_answer", "")

        result = {
            "query_index": self.query_index,
            "query": self.query,
            "response": self.response,
            "ground_truth_answer": ground_truth_answer,
            "retrieved_contexts": "\n---\n".join(self.retrieved_contexts),
            "num_contexts": len(self.retrieved_contexts),
            "score": self.score,
            **self.metadata,
        }

        # Restore metadata to avoid side effects
        self.metadata["ground_truth_answer"] = ground_truth_answer

        return result


class GenerationMetric(ABC):
    """
    Abstract base class for generation quality metrics.

    Generation metrics (Faithfulness, Answer Relevancy, etc.) evaluate the quality
    of LLM-generated responses. Unlike retrieval metrics that compare IDs,
    these metrics analyze text content.

    Attributes:
        name: Human-readable name of the metric
        short_name: Short identifier used in CLI and registry
        requires_async: Whether the metric requires async execution
    """

    name: str = "Base Generation Metric"
    short_name: str = "gen_base"
    requires_async: bool = True

    @abstractmethod
    async def calculate_query_score(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: List[str],
    ) -> Tuple[float, dict]:
        """
        Calculate metric score for a single query.

        Args:
            user_input: The user's query/question
            response: The LLM-generated response
            retrieved_contexts: List of retrieved context texts

        Returns:
            Tuple of (score, metadata_dict) where:
                - score: Metric value (0.0-1.0)
                - metadata_dict: Additional info (e.g., {"claims_count": 5})
        """
        pass

    @abstractmethod
    def aggregate_scores(
        self,
        query_results: List[GenerationQueryResult],
    ) -> Dict[str, Any]:
        """
        Aggregate individual query scores into final summary.

        Args:
            query_results: List of individual query results

        Returns:
            Dictionary with aggregated statistics
        """
        pass

    def get_result_columns(self) -> List[str]:
        """
        Get column names for per-query result export.

        Returns:
            List of column names for DataFrame export
        """
        return [
            "query_index",
            "query",
            "response",
            "num_contexts",
            "score",
        ]
