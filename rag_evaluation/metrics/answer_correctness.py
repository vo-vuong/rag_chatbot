"""
Answer Correctness metric using RAGAS AnswerCorrectness.

Measures the correctness of LLM response compared to ground truth answer.
Combines factual similarity and semantic similarity.
"""

import logging
from typing import Any, Dict, List, Tuple

from rag_evaluation.base.generation_metric_interface import (
    GenerationMetric,
    GenerationQueryResult,
)

logger = logging.getLogger(__name__)


class AnswerCorrectnessMetric(GenerationMetric):
    """
    Answer Correctness metric using RAGAS AnswerCorrectness.

    Evaluates how correct the LLM response is compared to the ground truth answer
    by combining factual similarity and semantic similarity.

    Formula:
        Answer Correctness = (weight_f × Factual Similarity) + (weight_s × Semantic Similarity)
        Default weights: 0.75 factual + 0.25 semantic

    Factual Similarity uses F1 score of claims:
        F1 = TP / (TP + 0.5 * (FP + FN))
        - TP: Claims in both response and reference
        - FP: Claims in response but not in reference
        - FN: Claims in reference but not in response

    Attributes:
        name: Human-readable metric name
        short_name: CLI identifier
        requires_reference: Whether this metric requires ground_truth_answer
        requires_response: Whether this metric requires LLM response
    """

    name: str = "Answer Correctness"
    short_name: str = "answer_correctness"
    requires_async: bool = True
    requires_reference: bool = True
    requires_response: bool = True

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        weights: Tuple[float, float] = (0.75, 0.25),
    ):
        """
        Initialize the Answer Correctness metric.

        Args:
            model_name: OpenAI model name for LLM-based evaluation
            embedding_model: OpenAI embedding model for semantic similarity
            weights: Tuple of (factual_weight, semantic_weight), must sum to 1.0
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.weights = weights
        self._scorer = None

    def _get_scorer(self):
        """Lazy initialization of RAGAS AnswerCorrectness scorer."""
        if self._scorer is None:
            try:
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from ragas.llms import LangchainLLMWrapper
                from ragas.metrics import AnswerCorrectness
                from ragas.metrics._answer_similarity import AnswerSimilarity

                llm = LangchainLLMWrapper(ChatOpenAI(model=self.model_name))
                embeddings = LangchainEmbeddingsWrapper(
                    OpenAIEmbeddings(model=self.embedding_model)
                )

                # Initialize AnswerSimilarity sub-metric required by AnswerCorrectness
                answer_similarity = AnswerSimilarity(embeddings=embeddings)

                self._scorer = AnswerCorrectness(
                    llm=llm,
                    embeddings=embeddings,
                    weights=list(self.weights),
                    answer_similarity=answer_similarity,
                )
                logger.info(
                    f"Initialized AnswerCorrectness scorer with LLM={self.model_name}, "
                    f"embeddings={self.embedding_model}, weights={self.weights}"
                )
            except ImportError as e:
                raise ImportError(
                    "RAGAS and langchain-openai are required for AnswerCorrectness. "
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
        Calculate Answer Correctness score for a single query.

        Args:
            user_input: The user's query/question
            response: The LLM-generated response
            retrieved_contexts: List of retrieved context texts (not used directly)
            reference: Ground truth answer for comparison

        Returns:
            Tuple of (score, metadata_dict)
        """
        if not response:
            return 0.0, {"error": "Empty response"}

        if not reference:
            return 0.0, {"error": "Missing reference (ground_truth_answer)"}

        try:
            from ragas.dataset_schema import SingleTurnSample

            scorer = self._get_scorer()
            sample = SingleTurnSample(
                user_input=user_input,
                response=response,
                reference=reference,
            )
            result = await scorer.single_turn_ascore(sample)

            score = result if isinstance(result, (int, float)) else float(result)
            score = max(0.0, min(1.0, score))

            metadata = {
                "model": self.model_name,
                "embedding_model": self.embedding_model,
                "weights": f"{self.weights[0]:.2f}f+{self.weights[1]:.2f}s",
            }

            return score, metadata

        except Exception as e:
            logger.error(f"AnswerCorrectness calculation failed: {e}")
            return 0.0, {"error": str(e)}

    def aggregate_scores(
        self,
        query_results: List[GenerationQueryResult],
    ) -> Dict[str, Any]:
        """
        Aggregate Answer Correctness scores across all queries.

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
            "embedding_model": self.embedding_model,
            "weights": f"{self.weights[0]:.2f}f+{self.weights[1]:.2f}s",
        }
