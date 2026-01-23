"""
Evaluator for generation metrics (Faithfulness, etc.).

Orchestrates evaluation of generation quality metrics using the chat API.
"""

import asyncio
import logging
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
    ):
        """
        Initialize the generation evaluator.

        Args:
            test_data_path: Path to test data Excel file
            api_base_url: Base URL for RAG API
            output_dir: Directory for output files
            limit: Optional limit on number of queries
        """
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.limit = limit

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
        export: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run generation metric evaluation asynchronously.

        Args:
            metric: Metric to run ("faithfulness")
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            verbose: Print detailed per-query results
            model_name: LLM model for evaluation
            export: Whether to export results to Excel
            output_path: Custom output path for results

        Returns:
            Dictionary with evaluation results
        """
        metric_instance = self._get_metric_instance(metric, model_name)

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
        export: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run generation metric evaluation (synchronous wrapper).

        Args:
            metric: Metric to run ("faithfulness")
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            verbose: Print detailed per-query results
            model_name: LLM model for evaluation
            export: Whether to export results to Excel
            output_path: Custom output path for results

        Returns:
            Dictionary with evaluation results
        """
        return asyncio.run(
            self.run_async(
                metric, top_k, score_threshold, verbose, model_name, export, output_path
            )
        )

    def _get_metric_instance(
        self, metric: str, model_name: str
    ) -> GenerationMetric:
        """Get metric instance by name."""
        if metric.lower() == "faithfulness":
            return FaithfulnessMetric(model_name=model_name)
        else:
            raise ValueError(f"Unknown generation metric: {metric}")

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

        for idx, query, ground_truth_ids, metadata in self.data_loader.iter_queries():
            # Call chat API to get response and contexts
            chat_result = self.api_client.chat_query(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
            )

            processed += 1

            if chat_result is None:
                logger.warning(f"Skipping query {idx}: API call failed")
                continue

            # Calculate metric score
            score, extra_metadata = await metric.calculate_query_score(
                user_input=query,
                response=chat_result.response,
                retrieved_contexts=chat_result.retrieved_contexts,
            )

            query_result = GenerationQueryResult(
                query_index=idx,
                query=query,
                response=chat_result.response,
                retrieved_contexts=chat_result.retrieved_contexts,
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
