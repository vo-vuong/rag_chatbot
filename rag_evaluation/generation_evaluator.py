"""
Evaluator for generation metrics (Faithfulness, etc.).

Orchestrates evaluation of generation quality metrics using the chat API.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rag_evaluation.api.rag_api_client import RAGAPIClient
from rag_evaluation.base.generation_metric_interface import (
    GenerationMetric,
    GenerationQueryResult,
)
from rag_evaluation.data.data_loader import TestDataLoader
from rag_evaluation.export.excel_exporter import ExcelExporter
from rag_evaluation.metrics.faithfulness import FaithfulnessMetric
from rag_evaluation.metrics.response_relevancy import ResponseRelevancyMetric
from rag_evaluation.metrics.context_precision import ContextPrecisionMetric
from rag_evaluation.metrics.context_recall import ContextRecallMetric
from rag_evaluation.metrics.answer_correctness import AnswerCorrectnessMetric

logger = logging.getLogger(__name__)

DEFAULT_TEST_DATA_PATH = (
    Path(__file__).parent / "prepare_testing_data" / "qr_smartphone_dataset.xlsx"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"


class GenerationEvaluator:
    """
    Evaluator for generation quality metrics.

    Uses the chat API to get LLM responses and contexts, then evaluates
    using metrics like Faithfulness.
    """

    def __init__(
        self,
        test_data_path: Union[str, Path] = DEFAULT_TEST_DATA_PATH,
        api_base_url: Optional[str] = None,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        limit: Optional[int] = None,
        delay: float = 0.0,
    ):
        """
        Initialize the generation evaluator.

        Args:
            test_data_path: Path to test data Excel file
            api_base_url: Base URL for RAG API
            output_dir: Directory for output files
            limit: Optional limit on number of queries
            delay: Time to sleep (in seconds) between queries
        """
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.delay = delay

        self.data_loader = TestDataLoader(self.test_data_path, limit=limit)

        if api_base_url:
            self.api_client = RAGAPIClient(base_url=api_base_url)
        else:
            self.api_client = RAGAPIClient()

        self.exporter = ExcelExporter(self.output_dir)

    async def run_async(
        self,
        metric: str = "faithfulness",
        top_k: int = 5,
        score_threshold: float = 0.0,
        verbose: bool = False,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        export: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run generation metric evaluation asynchronously.

        Args:
            metric: Metric to run ("faithfulness", "response_relevancy")
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            verbose: Print detailed per-query results
            model_name: LLM model for evaluation
            embedding_model: Embedding model for response_relevancy metric
            export: Whether to export results to Excel
            output_path: Custom output path for results

        Returns:
            Dictionary with evaluation results
        """
        metric_instance = self._get_metric_instance(metric, model_name, embedding_model)

        logger.info(f"Running {metric_instance.name} evaluation")
        logger.info(f"Model: {model_name}, top_k={top_k}, threshold={score_threshold}")

        query_results = await self._collect_and_evaluate(
            metric_instance, top_k, score_threshold, verbose
        )

        summary = metric_instance.aggregate_scores(query_results)

        result = {
            "metric_name": metric_instance.name,
            "query_results": query_results,
            "summary": summary,
            "evaluation_date": datetime.now().isoformat(),
            "test_data_path": str(self.test_data_path),
            "config": {
                "top_k": top_k,
                "score_threshold": score_threshold,
                "model": model_name,
            },
        }

        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"{metric_instance.name} Results:")
        logger.info(f"  Score: {summary.get('score', 0):.4f} ({summary.get('score', 0)*100:.2f}%)")
        logger.info(f"  Queries: {summary.get('total_queries', 0)}")
        logger.info(f"{'='*50}")

        # Export results
        if export:
            path = self.exporter.export_generation_result(
                result, query_results, output_path
            )
            logger.info(f"Results exported to: {path}")

        return result

    def run(
        self,
        metric: str = "faithfulness",
        top_k: int = 5,
        score_threshold: float = 0.0,
        verbose: bool = False,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        export: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run generation metric evaluation (synchronous wrapper).

        Args:
            metric: Metric to run ("faithfulness", "response_relevancy")
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            verbose: Print detailed per-query results
            model_name: LLM model for evaluation
            embedding_model: Embedding model for response_relevancy metric
            export: Whether to export results to Excel
            output_path: Custom output path for results

        Returns:
            Dictionary with evaluation results
        """
        return asyncio.run(
            self.run_async(
                metric,
                top_k,
                score_threshold,
                verbose,
                model_name,
                embedding_model,
                export,
                output_path,
            )
        )

    def _get_metric_instance(
        self, metric: str, model_name: str, embedding_model: str = "text-embedding-3-small"
    ) -> GenerationMetric:
        """Get metric instance by name."""
        if metric.lower() == "faithfulness":
            return FaithfulnessMetric(model_name=model_name)
        elif metric.lower() == "response_relevancy":
            return ResponseRelevancyMetric(
                model_name=model_name,
                embedding_model=embedding_model,
            )
        elif metric.lower() == "context_precision":
            return ContextPrecisionMetric(model_name=model_name)
        elif metric.lower() == "context_recall":
            return ContextRecallMetric(model_name=model_name)
        elif metric.lower() == "answer_correctness":
            return AnswerCorrectnessMetric(
                model_name=model_name,
                embedding_model=embedding_model,
            )
        else:
            raise ValueError(f"Unknown generation metric: {metric}")

    async def run_from_responses_async(
        self,
        responses_data: dict,
        metric: str = "faithfulness",
        verbose: bool = False,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        export: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation using pre-collected responses from JSON.

        Args:
            responses_data: Dictionary loaded from response JSON file
            metric: Metric to run
            verbose: Print detailed per-query results
            model_name: LLM model for evaluation
            embedding_model: Embedding model for response_relevancy metric
            export: Whether to export results to Excel
            output_path: Custom output path for results

        Returns:
            Dictionary with evaluation results
        """
        metric_instance = self._get_metric_instance(metric, model_name, embedding_model)

        # Validate metric requirements against response data
        requires_response = getattr(metric_instance, "requires_response", True)
        response_mode = responses_data["metadata"].get("mode", "chat")

        if requires_response and response_mode == "search":
            raise ValueError(
                f"Metric '{metric}' requires LLM response but responses were collected "
                f"in 'search' mode. Use 'chat' mode responses or collect new data."
            )

        logger.info(f"Running {metric_instance.name} from pre-collected responses")
        logger.info(f"Response mode: {response_mode}, Total queries: {len(responses_data['responses'])}")

        query_results = await self._evaluate_from_responses(
            metric_instance, responses_data["responses"], verbose
        )

        summary = metric_instance.aggregate_scores(query_results)
        top_k = responses_data["metadata"].get("top_k", 5)

        result = {
            "metric_name": metric_instance.name,
            "query_results": query_results,
            "summary": summary,
            "evaluation_date": datetime.now().isoformat(),
            "responses_file": responses_data["metadata"].get("created_at", ""),
            "config": {
                "top_k": top_k,
                "score_threshold": responses_data["metadata"].get("score_threshold", 0.0),
                "model": model_name,
                "from_responses": True,
            },
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"{metric_instance.name} Results (from responses):")
        logger.info(f"  Score: {summary.get('score', 0):.4f} ({summary.get('score', 0)*100:.2f}%)")
        logger.info(f"  Queries: {summary.get('total_queries', 0)}")
        logger.info(f"{'='*50}")

        if export:
            path = self.exporter.export_generation_result(
                result, query_results, output_path
            )
            logger.info(f"Results exported to: {path}")

        return result

    def run_from_responses(
        self,
        responses_data: dict,
        metric: str = "faithfulness",
        verbose: bool = False,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        export: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation from pre-collected responses (synchronous wrapper).

        Args:
            responses_data: Dictionary loaded from response JSON file
            metric: Metric to run
            verbose: Print detailed per-query results
            model_name: LLM model for evaluation
            embedding_model: Embedding model for response_relevancy metric
            export: Whether to export results to Excel
            output_path: Custom output path for results

        Returns:
            Dictionary with evaluation results
        """
        return asyncio.run(
            self.run_from_responses_async(
                responses_data,
                metric,
                verbose,
                model_name,
                embedding_model,
                export,
                output_path,
            )
        )

    async def _evaluate_from_responses(
        self,
        metric: GenerationMetric,
        responses: List[dict],
        verbose: bool,
    ) -> List[GenerationQueryResult]:
        """
        Evaluate metric using pre-collected responses.

        Args:
            metric: The metric instance to use
            responses: List of response dictionaries from JSON
            verbose: Print detailed output

        Returns:
            List of GenerationQueryResult objects
        """
        query_results = []
        total = len(responses)
        requires_reference = getattr(metric, "requires_reference", False)

        for i, resp in enumerate(responses, 1):
            query = resp["query"]
            response = resp.get("response", "")
            retrieved_contexts = resp.get("retrieved_contexts", [])
            ground_truth_answer = resp.get("ground_truth_answer", "")
            metadata = {
                "ground_truth_answer": ground_truth_answer,
                "ground_truth_ids": resp.get("ground_truth_ids", []),
            }

            if requires_reference:
                score, extra_metadata = await metric.calculate_query_score(
                    user_input=query,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                    reference=ground_truth_answer,
                )
            else:
                score, extra_metadata = await metric.calculate_query_score(
                    user_input=query,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )

            query_result = GenerationQueryResult(
                query_index=resp.get("query_index", i),
                query=query,
                response=response,
                retrieved_contexts=retrieved_contexts,
                score=score,
                metadata={**metadata, **extra_metadata},
            )
            query_results.append(query_result)

            if verbose:
                logger.info(
                    f"[{i}/{total}] Query: '{query[:40]}...' | Score: {score:.4f}"
                )
            elif i % 5 == 0:
                logger.info(f"Evaluated {i}/{total} queries...")

        logger.info(f"Completed evaluation for {len(query_results)} queries")
        return query_results

    async def _collect_and_evaluate(
        self,
        metric: GenerationMetric,
        top_k: int,
        score_threshold: float,
        verbose: bool,
    ) -> List[GenerationQueryResult]:
        """
        Collect chat responses and evaluate with the metric.

        Args:
            metric: The metric instance to use
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            verbose: Print detailed output

        Returns:
            List of GenerationQueryResult objects
        """
        total_queries = self.data_loader.get_valid_query_count()
        query_results = []
        processed = 0

        # Check metric requirements
        requires_reference = getattr(metric, "requires_reference", False)
        requires_response = getattr(metric, "requires_response", True)

        for idx, query, ground_truth_ids, metadata in self.data_loader.iter_queries():
            # Add delay if needed
            if self.delay > 0 and processed > 0:
                time.sleep(self.delay)

            processed += 1
            ground_truth_answer = metadata.get("ground_truth_answer", "")

            # Use different API based on metric requirements
            if requires_response:
                # Use chat API for metrics needing LLM response
                chat_result = self.api_client.chat_query(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )

                if chat_result is None:
                    logger.warning(f"Skipping query {idx}: Chat API call failed")
                    continue

                retrieved_contexts = chat_result.retrieved_contexts
                response = chat_result.response
            else:
                # Use search API for metrics not needing LLM response
                search_result = self.api_client.search_with_contexts(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )

                if search_result is None:
                    logger.warning(f"Skipping query {idx}: Search API call failed")
                    continue

                retrieved_contexts = search_result.retrieved_contexts
                response = ""  # Not needed

            # Calculate metric score
            if requires_reference:
                score, extra_metadata = await metric.calculate_query_score(
                    user_input=query,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                    reference=ground_truth_answer,
                )
            else:
                score, extra_metadata = await metric.calculate_query_score(
                    user_input=query,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )

            query_result = GenerationQueryResult(
                query_index=idx,
                query=query,
                response=response,
                retrieved_contexts=retrieved_contexts,
                score=score,
                metadata={**metadata, **extra_metadata},
            )
            query_results.append(query_result)

            if verbose:
                logger.info(
                    f"[{processed}/{total_queries}] Query: '{query[:40]}...' | "
                    f"Score: {score:.4f}"
                )
            elif processed % 5 == 0:
                logger.info(f"Processed {processed}/{total_queries} queries...")

        logger.info(f"Completed evaluation for {len(query_results)} queries")
        return query_results


def run_faithfulness_evaluation(
    test_data_path: Optional[Path] = None,
    api_base_url: Optional[str] = None,
    top_k: int = 5,
    score_threshold: float = 0.0,
    limit: Optional[int] = None,
    verbose: bool = False,
    model_name: str = "gpt-4o-mini",
    export: bool = True,
    output_path: Optional[Path] = None,
    delay: float = 0.0,
) -> Dict[str, Any]:
    """
    Convenience function to run Faithfulness evaluation.

    Args:
        test_data_path: Path to test data file
        api_base_url: RAG API base URL
        top_k: Number of documents to retrieve
        score_threshold: Minimum similarity score
        limit: Limit number of queries
        verbose: Print detailed results
        model_name: LLM model for evaluation
        export: Whether to export results to Excel
        output_path: Custom output path for results

    Returns:
        Dictionary with evaluation results
    """
    evaluator = GenerationEvaluator(
        test_data_path=test_data_path or DEFAULT_TEST_DATA_PATH,
        api_base_url=api_base_url,
        limit=limit,
        delay=delay,
    )

    return evaluator.run(
        metric="faithfulness",
        top_k=top_k,
        score_threshold=score_threshold,
        verbose=verbose,
        model_name=model_name,
        export=export,
        output_path=output_path,
    )
