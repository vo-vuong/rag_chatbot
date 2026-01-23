"""
Response Relevancy metric using RAGAS.

Measures how relevant the LLM response is to the user's input/question.
Uses question generation and embedding similarity to evaluate relevance.
"""

import logging
from typing import Any, Dict, List, Tuple

from rag_evaluation.base.generation_metric_interface import (
    GenerationMetric,
    GenerationQueryResult,
)

logger = logging.getLogger(__name__)


class ResponseRelevancyMetric(GenerationMetric):
    """
    Response Relevancy metric using RAGAS.

    Measures how relevant a response is to the user's input by:
    1. Generating artificial questions from the response using LLM
    2. Computing embedding similarity between generated questions and original input
    3. Averaging the similarities to get the final score

    Score ranges from 0 to 1, where 1 means highly relevant.

    Formula:
        ResponseRelevancy = (1/N) * Σ cosine_similarity(E_generated_q, E_original_q)
    """

    name: str = "Response Relevancy"
    short_name: str = "response_relevancy"
    requires_async: bool = True

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the Response Relevancy metric.

        Args:
            model_name: OpenAI model name for question generation
            embedding_model: OpenAI embedding model for similarity calculation
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self._scorer = None

    def _get_scorer(self):
        """Lazy initialization of RAGAS ResponseRelevancy scorer."""
        if self._scorer is None:
            try:
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from ragas.llms import LangchainLLMWrapper
                from ragas.metrics import ResponseRelevancy

                llm = LangchainLLMWrapper(ChatOpenAI(model=self.model_name))
                embeddings = LangchainEmbeddingsWrapper(
                    OpenAIEmbeddings(model=self.embedding_model)
                )

                self._scorer = ResponseRelevancy(llm=llm, embeddings=embeddings)
                logger.info(
                    f"Initialized ResponseRelevancy scorer with "
                    f"LLM={self.model_name}, embeddings={self.embedding_model}"
                )
            except ImportError as e:
                raise ImportError(
                    "RAGAS and langchain-openai are required for ResponseRelevancy. "
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
        Calculate Response Relevancy score for a single query.

        Args:
            user_input: The user's query/question
            response: The LLM-generated response
            retrieved_contexts: List of retrieved context texts (not used but required by interface)

        Returns:
            Tuple of (score, metadata_dict)
        """
        if not user_input or not response:
            return 0.0, {"error": "Empty user_input or response"}

        try:
            from ragas.dataset_schema import SingleTurnSample

            scorer = self._get_scorer()
            sample = SingleTurnSample(
                user_input=user_input,
                response=response,
            )
            result = await scorer.single_turn_ascore(sample)

            score = result if isinstance(result, (int, float)) else float(result)

            # Clamp score to [0, 1] as cosine similarity can be negative
            score = max(0.0, min(1.0, score))

            metadata = {
                "model": self.model_name,
                "embedding_model": self.embedding_model,
            }

            return score, metadata

        except Exception as e:
            logger.error(f"ResponseRelevancy calculation failed: {e}")
            return 0.0, {"error": str(e)}

    def aggregate_scores(
        self,
        query_results: List[GenerationQueryResult],
    ) -> Dict[str, Any]:
        """
        Aggregate Response Relevancy scores across all queries.

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
        }
