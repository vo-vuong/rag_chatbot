"""
MRR@K (Mean Reciprocal Rank) metric implementation.

Measures the average reciprocal rank of the first relevant document
in the top-K retrieved results across all queries.
"""

from typing import List, Tuple

from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.base.evaluation_result import QueryResult, MetricSummary
from rag_evaluation.metrics.registry import register_metric


@register_metric("mrr")
class MRRAtK(RetrievalMetric):
    """
    MRR@K (Mean Reciprocal Rank) metric.

    Measures the average reciprocal rank of the first relevant document
    found in the top-K results. For each query, finds the position of
    the first relevant document and calculates 1/rank.

    Formula:
        Query RR = 1 / rank_of_first_relevant_doc (0 if none found)
        MRR@K = Average of all query reciprocal ranks

    Attributes:
        name: "MRR@K"
        short_name: "mrr"
    """

    name: str = "MRR@K"
    short_name: str = "mrr"

    def calculate_query_score(
        self,
        ground_truth_ids: List[int],
        retrieved_ids: List[int],
    ) -> Tuple[float, dict]:
        """
        Calculate MRR score for a single query.

        Args:
            ground_truth_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (top-K, in ranked order)

        Returns:
            Tuple of (reciprocal_rank, metadata) where RR is 0.0-1.0
        """
        if not retrieved_ids or not ground_truth_ids:
            return 0.0, {"first_relevant_rank": None}

        # Find rank of first relevant document (1-indexed)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_ids:
                reciprocal_rank = 1.0 / rank
                return reciprocal_rank, {"first_relevant_rank": rank}

        # No relevant document found in top-K
        return 0.0, {"first_relevant_rank": None}

    def aggregate_scores(
        self,
        query_results: List[QueryResult],
        k: int,
        score_threshold: float,
    ) -> MetricSummary:
        """
        Aggregate individual query scores into MRR@K summary.

        Args:
            query_results: List of individual query results
            k: K value used in evaluation
            score_threshold: Score threshold used in retrieval

        Returns:
            MetricSummary with MRR and related statistics
        """
        if not query_results:
            return MetricSummary(
                metric_name=self.name,
                k=k,
                score=0.0,
                total_queries=0,
                score_threshold=score_threshold,
            )

        total_rr = sum(qr.score for qr in query_results)
        total = len(query_results)
        mrr = total_rr / total

        # Calculate additional statistics
        found_count = sum(
            1 for qr in query_results if qr.metadata.get("first_relevant_rank")
        )
        not_found_count = total - found_count

        # Rank distribution
        rank_1 = sum(
            1 for qr in query_results if qr.metadata.get("first_relevant_rank") == 1
        )

        return MetricSummary(
            metric_name=self.name,
            k=k,
            score=mrr,
            total_queries=total,
            score_threshold=score_threshold,
            additional_stats={
                "Found in Top-K": found_count,
                "Not Found": not_found_count,
                "Rank 1 (Perfect)": rank_1,
            },
        )

    def get_result_columns(self) -> List[str]:
        """Get column names for per-query result export."""
        return [
            "query_index",
            "query",
            "ground_truth_ids",
            "retrieved_ids",
            "first_relevant_rank",
            "reciprocal_rank",
            "difficulty",
            "source_files",
        ]
