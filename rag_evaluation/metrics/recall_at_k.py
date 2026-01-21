"""
Recall@K metric implementation.

Measures the average proportion of relevant documents that appear
in the top-K retrieved results across all queries.
"""

from typing import List, Tuple

from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.base.evaluation_result import QueryResult, MetricSummary
from rag_evaluation.metrics.registry import register_metric


@register_metric("recall")
class RecallAtK(RetrievalMetric):
    """
    Recall@K metric.

    Measures the average proportion of relevant documents that appear
    in the top-K retrieved results. For each query, calculates what
    fraction of ground truth documents were successfully retrieved.

    Formula:
        Query Recall = (# ground truth docs in top-K) / (# total ground truth docs)
        Recall@K = Average of all query recalls

    Attributes:
        name: "Recall@K"
        short_name: "recall"
    """

    name: str = "Recall@K"
    short_name: str = "recall"

    def calculate_query_score(
        self,
        ground_truth_ids: List[int],
        retrieved_ids: List[int],
    ) -> Tuple[float, dict]:
        """
        Calculate Recall@K score for a single query.

        Args:
            ground_truth_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs

        Returns:
            Tuple of (recall_score, metadata) where recall is 0.0-1.0
        """
        if not ground_truth_ids:
            return 0.0, {"hits_count": 0, "ground_truth_count": 0}

        hits_count = sum(1 for gt_id in ground_truth_ids if gt_id in retrieved_ids)
        recall = hits_count / len(ground_truth_ids)

        metadata = {
            "hits_count": hits_count,
            "ground_truth_count": len(ground_truth_ids),
        }

        return recall, metadata

    def aggregate_scores(
        self,
        query_results: List[QueryResult],
        k: int,
        score_threshold: float,
    ) -> MetricSummary:
        """
        Aggregate individual query scores into Recall@K summary.

        Args:
            query_results: List of individual query results
            k: K value used in evaluation
            score_threshold: Score threshold used in retrieval

        Returns:
            MetricSummary with average recall and related statistics
        """
        if not query_results:
            return MetricSummary(
                metric_name=self.name,
                k=k,
                score=0.0,
                total_queries=0,
                score_threshold=score_threshold,
            )

        total_recall = sum(qr.score for qr in query_results)
        total = len(query_results)
        avg_recall = total_recall / total

        # Calculate additional statistics
        perfect_recall = sum(1 for qr in query_results if qr.score == 1.0)
        zero_recall = sum(1 for qr in query_results if qr.score == 0.0)

        return MetricSummary(
            metric_name=self.name,
            k=k,
            score=avg_recall,
            total_queries=total,
            score_threshold=score_threshold,
            additional_stats={
                "Perfect Recall Queries": perfect_recall,
                "Zero Recall Queries": zero_recall,
            },
        )

    def get_result_columns(self) -> List[str]:
        """Get column names for per-query result export."""
        return [
            "query_index",
            "query",
            "ground_truth_ids",
            "retrieved_ids",
            "ground_truth_count",
            "hits_count",
            "query_recall",
            "difficulty",
            "source_files",
        ]
