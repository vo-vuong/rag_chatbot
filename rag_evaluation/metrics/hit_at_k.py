"""
Hit@K metric implementation.

Measures the proportion of queries where at least one relevant document
appears in the top-K retrieved results.
"""

from typing import List, Tuple

from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.base.evaluation_result import QueryResult, MetricSummary
from rag_evaluation.metrics.registry import register_metric


@register_metric("hit")
class HitAtK(RetrievalMetric):
    """
    Hit@K (Hit Rate) metric.

    Measures the proportion of queries where at least one relevant document
    appears in the top-K retrieved results. A binary metric where each query
    either has a "hit" (1) or "miss" (0).

    Formula:
        Hit@K = (Number of queries with at least one hit) / (Total queries)

    Attributes:
        name: "Hit@K"
        short_name: "hit"
    """

    name: str = "Hit@K"
    short_name: str = "hit"

    def calculate_query_score(
        self,
        ground_truth_ids: List[int],
        retrieved_ids: List[int],
    ) -> Tuple[float, dict]:
        """
        Calculate Hit@K score for a single query.

        Args:
            ground_truth_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs

        Returns:
            Tuple of (score, metadata) where score is 1.0 for hit, 0.0 for miss
        """
        is_hit = any(gt_id in retrieved_ids for gt_id in ground_truth_ids)
        score = 1.0 if is_hit else 0.0

        metadata = {
            "is_hit": is_hit,
        }

        return score, metadata

    def aggregate_scores(
        self,
        query_results: List[QueryResult],
        k: int,
        score_threshold: float,
    ) -> MetricSummary:
        """
        Aggregate individual query scores into Hit@K summary.

        Args:
            query_results: List of individual query results
            k: K value used in evaluation
            score_threshold: Score threshold used in retrieval

        Returns:
            MetricSummary with hit rate and related statistics
        """
        if not query_results:
            return MetricSummary(
                metric_name=self.name,
                k=k,
                score=0.0,
                total_queries=0,
                score_threshold=score_threshold,
            )

        hits = sum(1 for qr in query_results if qr.score > 0)
        total = len(query_results)
        hit_rate = hits / total

        return MetricSummary(
            metric_name=self.name,
            k=k,
            score=hit_rate,
            total_queries=total,
            score_threshold=score_threshold,
            additional_stats={
                "Hits": hits,
                "Misses": total - hits,
            },
        )

    def get_result_columns(self) -> List[str]:
        """Get column names for per-query result export."""
        return [
            "query_index",
            "query",
            "ground_truth_ids",
            "retrieved_ids",
            "is_hit",
            "difficulty",
            "source_files",
        ]
