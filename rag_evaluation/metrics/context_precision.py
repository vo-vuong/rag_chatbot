"""
Context Precision metric using RAGAS LLMContextPrecisionWithReference.

Measures how well relevant chunks are ranked higher in retrieved contexts.
Uses ground truth reference answer to determine relevance.
"""

import logging
from typing import Any, Dict, List, Tuple

from rag_evaluation.base.generation_metric_interface import (
    GenerationMetric,
    GenerationQueryResult,
)

logger = logging.getLogger(__name__)


class ContextPrecisionMetric(GenerationMetric):
    """
    Context Precision metric using RAGAS LLMContextPrecisionWithReference.

    Evaluates how well the retriever ranks relevant chunks at the top.
    Uses an LLM to compare each retrieved chunk against the reference answer
    to determine if it's relevant.

    Formula:
        Context Precision = Mean of Precision@K for each chunk
        where Precision@K = (relevant chunks at rank K) / K

    Higher scores indicate relevant chunks appear earlier in the ranking.

    Attributes:
        name: Human-readable metric name
        short_name: CLI identifier
        requires_reference: Whether this metric requires ground_truth_answer
    """

    name: str = "Context Precision"
    short_name: str = "context_precision"
    requires_async: bool = True
    requires_reference: bool = True

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the Context Precision metric.

        Args:
            model_name: OpenAI model name for LLM-based relevance judgment
        """
        self.model_name = model_name
        self._scorer = None

    def _get_scorer(self):
        """Lazy initialization of RAGAS LLMContextPrecisionWithReference scorer."""
        if self._scorer is None:
            try:
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper
                from ragas.metrics import LLMContextPrecisionWithReference

                llm = LangchainLLMWrapper(ChatOpenAI(model=self.model_name))
                self._scorer = LLMContextPrecisionWithReference(llm=llm)
                logger.info(
                    f"Initialized ContextPrecision scorer with LLM={self.model_name}"
                )
            except ImportError as e:
                raise ImportError(
                    "RAGAS and langchain-openai are required for ContextPrecision. "
                    "Install with: pip install ragas langchain-openai"
                ) from e
        return self._scorer

    async def calculate_query_score(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: List[str],
        reference: str = "",
    ) -> Tuple[float, dict]:
        """
        Calculate Context Precision score for a single query.

        Args:
            user_input: The user's query/question
            response: Not used (required by interface)
            retrieved_contexts: List of retrieved context texts
            reference: Ground truth answer for relevance comparison

        Returns:
            Tuple of (score, metadata_dict)
        """
        if not user_input or not retrieved_contexts:
            return 0.0, {"error": "Empty user_input or retrieved_contexts"}

        if not reference:
            return 0.0, {"error": "Missing reference (ground_truth_answer)"}

        try:
            from ragas.dataset_schema import SingleTurnSample

            scorer = self._get_scorer()
            sample = SingleTurnSample(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            result = await scorer.single_turn_ascore(sample)

            score = result if isinstance(result, (int, float)) else float(result)
            score = max(0.0, min(1.0, score))

            metadata = {
                "model": self.model_name,
                "num_contexts": len(retrieved_contexts),
            }

            return score, metadata

        except Exception as e:
            logger.error(f"ContextPrecision calculation failed: {e}")
            return 0.0, {"error": str(e)}

    def aggregate_scores(
        self,
        query_results: List[GenerationQueryResult],
    ) -> Dict[str, Any]:
        """
        Aggregate Context Precision scores across all queries.

        Args:
            query_results: List of individual query results

        Returns:
            Dictionary with aggregated statistics
        """
        if not query_results:
            return {
                "metric_name": self.name,
                "score": 0.0,
                "total_queries": 0,
            }

        scores = [qr.score for qr in query_results]
        valid_scores = [s for s in scores if s > 0]

        return {
            "metric_name": self.name,
            "score": sum(scores) / len(scores),
            "total_queries": len(query_results),
            "valid_queries": len(valid_scores),
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "model": self.model_name,
        }
