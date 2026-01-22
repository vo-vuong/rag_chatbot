"""
F1@K metric implementation.

Measures the harmonic mean of Precision@K and Recall@K,
providing a balanced measure of retrieval quality.
"""

from typing import List, Tuple

from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.base.evaluation_result import QueryResult, MetricSummary
from rag_evaluation.metrics.registry import register_metric


@register_metric("f1")
class F1AtK(RetrievalMetric):
    """
    F1@K metric.

    Measures the harmonic mean of Precision@K and Recall@K for each query,
    then averages across all queries. Provides a balanced measure that
    considers both precision and recall.

    Formula:
        Query Precision = (# relevant docs in top-K) / K
        Query Recall = (# relevant docs in top-K) / (# total relevant docs)
        Query F1 = 2 * Precision * Recall / (Precision + Recall)
        F1@K = Average of all query F1 scores

    Attributes:
        name: "F1@K"
        short_name: "f1"
    """

    name: str = "F1@K"
    short_name: str = "f1"

    def calculate_query_score(
        self,
        ground_truth_ids: List[int],
        retrieved_ids: List[int],
    ) -> Tuple[float, dict]:
        """
        Calculate F1@K score for a single query.

        Args:
            ground_truth_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (top-K)

        Returns:
            Tuple of (f1_score, metadata) where F1 is 0.0-1.0
        """
        if not retrieved_ids or not ground_truth_ids:
            return 0.0, {
                "hits_count": 0,
                "retrieved_count": len(retrieved_ids),
                "ground_truth_count": len(ground_truth_ids),
                "precision": 0.0,
                "recall": 0.0,
            }

        hits_count = sum(1 for r_id in retrieved_ids if r_id in ground_truth_ids)

        precision = hits_count / len(retrieved_ids)
        recall = hits_count / len(ground_truth_ids)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        metadata = {
            "hits_count": hits_count,
            "retrieved_count": len(retrieved_ids),
            "ground_truth_count": len(ground_truth_ids),
            "precision": precision,
            "recall": recall,
        }

        return f1, metadata

    def aggregate_scores(
        self,
        query_results: List[QueryResult],
        k: int,
        score_threshold: float,
    ) -> MetricSummary:
        """
        Aggregate individual query scores into F1@K summary.

        Args:
            query_results: List of individual query results
            k: K value used in evaluation
            score_threshold: Score threshold used in retrieval

        Returns:
            MetricSummary with average F1 and related statistics
        """
        if not query_results:
            return MetricSummary(
                metric_name=self.name,
                k=k,
                score=0.0,
                total_queries=0,
                score_threshold=score_threshold,
            )

        total_f1 = sum(qr.score for qr in query_results)
        total = len(query_results)
        avg_f1 = total_f1 / total

        # Calculate additional statistics
        perfect_f1 = sum(1 for qr in query_results if qr.score == 1.0)
        zero_f1 = sum(1 for qr in query_results if qr.score == 0.0)

        # Calculate average precision and recall from metadata
        avg_precision = sum(
            qr.metadata.get("precision", 0.0) for qr in query_results
        ) / total
        avg_recall = sum(
            qr.metadata.get("recall", 0.0) for qr in query_results
        ) / total

        return MetricSummary(
            metric_name=self.name,
            k=k,
            score=avg_f1,
            total_queries=total,
            score_threshold=score_threshold,
            additional_stats={
                "Avg Precision": round(avg_precision, 4),
                "Avg Recall": round(avg_recall, 4),
                "Perfect F1 Queries": perfect_f1,
                "Zero F1 Queries": zero_f1,
            },
        )

    def get_result_columns(self) -> List[str]:
        """Get column names for per-query result export."""
        return [
            "query_index",
            "query",
            "ground_truth_ids",
            "retrieved_ids",
            "hits_count",
            "precision",
            "recall",
            "query_f1",
            "difficulty",
            "source_files",
        ]
