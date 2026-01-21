"""
Base interface for retrieval metrics.

Defines the abstract base class that all retrieval metrics must implement,
ensuring consistent API across different metric types.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from rag_evaluation.base.evaluation_result import MetricSummary, QueryResult


class RetrievalMetric(ABC):
    """
    Abstract base class for retrieval metrics.

    All retrieval metrics (Hit@K, Recall@K, MRR, NDCG, etc.) must inherit
    from this class and implement the required methods.

    Attributes:
        name: Human-readable name of the metric
        short_name: Short identifier used in CLI and registry
    """

    name: str = "Base Metric"
    short_name: str = "base"

    @abstractmethod
    def calculate_query_score(
        self,
        ground_truth_ids: List[int],
        retrieved_ids: List[int],
    ) -> Tuple[float, dict]:
        """
        Calculate metric score for a single query.

        Args:
            ground_truth_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)

        Returns:
            Tuple of (score, metadata_dict) where:
                - score: Metric value for this query (e.g., 0/1 for hit, 0.0-1.0 for recall)
                - metadata_dict: Additional info (e.g., {"is_hit": True, "hits_count": 2})
        """
        pass

    @abstractmethod
    def aggregate_scores(
        self,
        query_results: List[QueryResult],
        k: int,
        score_threshold: float,
    ) -> MetricSummary:
        """
        Aggregate individual query scores into final metric summary.

        Args:
            query_results: List of individual query results
            k: K value used in evaluation
            score_threshold: Score threshold used in retrieval

        Returns:
            MetricSummary with aggregated statistics
        """
        pass

    def get_result_columns(self) -> List[str]:
        """
        Get column names for per-query result export.

        Override this method to customize export columns for specific metrics.

        Returns:
            List of column names for DataFrame export
        """
        return [
            "query_index",
            "query",
            "ground_truth_ids",
            "retrieved_ids",
            "score",
        ]
