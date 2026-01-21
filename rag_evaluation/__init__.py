"""
RAG Evaluation Framework.

A modular, extensible framework for evaluating RAG retrieval performance.

Usage:
    # CLI usage
    conda activate rag_chatbot && python -m rag_evaluation --metric hit --k 5
    conda activate rag_chatbot && python -m rag_evaluation --metric hit recall --k 10
    conda activate rag_chatbot && python -m rag_evaluation --metric all --k 5 -v

    # Programmatic usage
    from rag_evaluation import Evaluator
    from rag_evaluation.metrics import HitAtK, RecallAtK

    evaluator = Evaluator(test_data_path="path/to/data.xlsx")
    results = evaluator.run(metrics=["hit", "recall"], k=5)
"""

from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.base.evaluation_result import (
    EvaluationResult,
    QueryResult,
    MetricSummary,
)
from rag_evaluation.evaluator import Evaluator

__all__ = [
    "RetrievalMetric",
    "EvaluationResult",
    "QueryResult",
    "MetricSummary",
    "Evaluator",
]
