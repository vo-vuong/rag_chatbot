"""
Precision@K metric implementation.

Measures the proportion of retrieved documents that are relevant
in the top-K results across all queries.
"""

from typing import List, Tuple

from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.base.evaluation_result import QueryResult, MetricSummary
from rag_evaluation.metrics.registry import register_metric


@register_metric("precision")
class PrecisionAtK(RetrievalMetric):
    """
    Precision@K metric.

    Measures the proportion of retrieved documents that are relevant
    in the top-K results. For each query, calculates what fraction
    of the K retrieved documents were actually relevant.

    Formula:
        Query Precision = (# relevant docs in top-K) / K
        Precision@K = Average of all query precisions

    Attributes:
        name: "Precision@K"
        short_name: "precision"
    """

    name: str = "Precision@K"
    short_name: str = "precision"

    def calculate_query_score(
        self,
        ground_truth_ids: List[int],
        retrieved_ids: List[int],
    ) -> Tuple[float, dict]:
        """
        Calculate Precision@K score for a single query.

        Args:
            ground_truth_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (top-K)

        Returns:
            Tuple of (precision_score, metadata) where precision is 0.0-1.0
        """
        if not retrieved_ids:
            return 0.0, {"hits_count": 0, "retrieved_count": 0}

        hits_count = sum(1 for r_id in retrieved_ids if r_id in ground_truth_ids)
        precision = hits_count / len(retrieved_ids)

        metadata = {
            "hits_count": hits_count,
            "retrieved_count": len(retrieved_ids),
        }

        return precision, metadata

    def aggregate_scores(
        self,
        query_results: List[QueryResult],
        k: int,
        score_threshold: float,
    ) -> MetricSummary:
        """
        Aggregate individual query scores into Precision@K summary.

        Args:
            query_results: List of individual query results
            k: K value used in evaluation
            score_threshold: Score threshold used in retrieval

        Returns:
            MetricSummary with average precision and related statistics
        """
        if not query_results:
            return MetricSummary(
                metric_name=self.name,
                k=k,
                score=0.0,
                total_queries=0,
                score_threshold=score_threshold,
            )

        total_precision = sum(qr.score for qr in query_results)
        total = len(query_results)
        avg_precision = total_precision / total

        # Calculate additional statistics
        perfect_precision = sum(1 for qr in query_results if qr.score == 1.0)
        zero_precision = sum(1 for qr in query_results if qr.score == 0.0)

        return MetricSummary(
            metric_name=self.name,
            k=k,
            score=avg_precision,
            total_queries=total,
            score_threshold=score_threshold,
            additional_stats={
                "Perfect Precision Queries": perfect_precision,
                "Zero Precision Queries": zero_precision,
            },
        )

    def get_result_columns(self) -> List[str]:
        """Get column names for per-query result export."""
        return [
            "query_index",
            "query",
            "ground_truth_ids",
            "retrieved_ids",
            "retrieved_count",
            "hits_count",
            "query_precision",
            "difficulty",
            "source_files",
        ]
