"""Base classes and interfaces for RAG evaluation."""

from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.base.evaluation_result import (
    EvaluationResult,
    QueryResult,
    MetricSummary,
)

__all__ = [
    "RetrievalMetric",
    "EvaluationResult",
    "QueryResult",
    "MetricSummary",
]
