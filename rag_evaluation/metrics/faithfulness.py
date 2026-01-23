"""
Faithfulness metric using RAGAS.

Measures how factually consistent the LLM response is with the retrieved context.
A response is faithful if all its claims can be supported by the retrieved context.
"""

import logging
from typing import Any, Dict, List, Tuple

from rag_evaluation.base.generation_metric_interface import (
    GenerationMetric,
    GenerationQueryResult,
)

logger = logging.getLogger(__name__)


class FaithfulnessMetric(GenerationMetric):
    """
    Faithfulness metric using RAGAS.

    Measures factual consistency between response and retrieved context.
    Score ranges from 0 to 1, where 1 means all claims are supported.

    Formula:
        Faithfulness = (Claims supported by context) / (Total claims in response)
    """

    name: str = "Faithfulness"
    short_name: str = "faithfulness"
    requires_async: bool = True

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the Faithfulness metric.

        Args:
            model_name: OpenAI model name for claim extraction/verification
        """
        self.model_name = model_name
        self._scorer = None

    def _get_scorer(self):
        """Lazy initialization of RAGAS Faithfulness scorer."""
        if self._scorer is None:
            try:
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper
                from ragas.metrics import Faithfulness

                llm = LangchainLLMWrapper(ChatOpenAI(model=self.model_name))
                self._scorer = Faithfulness(llm=llm)
                logger.info(f"Initialized Faithfulness scorer with {self.model_name}")
            except ImportError as e:
                raise ImportError(
                    "RAGAS and langchain-openai are required for Faithfulness metric. "
                    "Install with: pip install ragas langchain-openai"
                ) from e
        return self._scorer

    async def calculate_query_score(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: List[str],
    ) -> Tuple[float, dict]:
        """
        Calculate Faithfulness score for a single query.

        Args:
            user_input: The user's query/question
            response: The LLM-generated response
            retrieved_contexts: List of retrieved context texts

        Returns:
            Tuple of (score, metadata_dict)
        """
        if not response or not retrieved_contexts:
            return 0.0, {"error": "Empty response or contexts"}

        try:
            from ragas.dataset_schema import SingleTurnSample

            scorer = self._get_scorer()
            sample = SingleTurnSample(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
            result = await scorer.single_turn_ascore(sample)

            score = result if isinstance(result, (int, float)) else float(result)

            metadata = {
                "model": self.model_name,
                "num_contexts": len(retrieved_contexts),
            }

            return score, metadata

        except Exception as e:
            logger.error(f"Faithfulness calculation failed: {e}")
            return 0.0, {"error": str(e)}

    def aggregate_scores(
        self,
        query_results: List[GenerationQueryResult],
    ) -> Dict[str, Any]:
        """
        Aggregate Faithfulness scores across all queries.

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


