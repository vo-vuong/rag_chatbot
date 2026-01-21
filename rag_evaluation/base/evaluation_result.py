"""
Evaluation result models for RAG metrics.

Provides standardized data structures for storing evaluation results,
enabling consistent export and analysis across different metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class QueryResult:
    """Result for a single query evaluation."""

    query_index: int
    query: str
    ground_truth_ids: List[int]
    retrieved_ids: List[int]
    score: float  # Metric-specific score (hit=0/1, recall=0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "query_index": self.query_index,
            "query": self.query,
            "ground_truth_ids": str(self.ground_truth_ids),
            "retrieved_ids": str(self.retrieved_ids),
            "score": self.score,
            **self.metadata,
        }


@dataclass
class MetricSummary:
    """Summary statistics for a metric evaluation."""

    metric_name: str
    k: int
    score: float
    total_queries: int
    score_threshold: float = 0.0
    additional_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "metric_name": self.metric_name,
            "k": self.k,
            "score": self.score,
            "score_pct": f"{self.score * 100:.2f}%",
            "total_queries": self.total_queries,
            "score_threshold": self.score_threshold,
            **self.additional_stats,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a metric."""

    metric_name: str
    k: int
    score: float
    query_results: List[QueryResult]
    summary: MetricSummary
    evaluation_date: datetime = field(default_factory=datetime.now)
    test_data_path: Optional[str] = None

    @property
    def total_queries(self) -> int:
        """Total number of queries evaluated."""
        return len(self.query_results)

    def get_query_results_as_dicts(self) -> List[Dict[str, Any]]:
        """Get all query results as list of dictionaries."""
        return [qr.to_dict() for qr in self.query_results]
