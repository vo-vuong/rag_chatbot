"""
Main evaluator orchestrator for RAG evaluation.

Coordinates data loading, API calls, metric calculation, and result export.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from rag_evaluation.api.rag_api_client import RAGAPIClient
from rag_evaluation.base.evaluation_result import EvaluationResult, QueryResult
from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.data.data_loader import TestDataLoader
from rag_evaluation.export.excel_exporter import ExcelExporter
from rag_evaluation.metrics.registry import MetricRegistry

# Import metrics to trigger registration
import rag_evaluation.metrics.hit_at_k  # noqa: F401
import rag_evaluation.metrics.recall_at_k  # noqa: F401

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TEST_DATA_PATH = (
    Path(__file__).parent / "prepare_testing_data" / "qr_smartphone_dataset.xlsx"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"


class Evaluator:
    """
    Main evaluator for RAG retrieval metrics.

    Orchestrates the evaluation process by coordinating:
    - Data loading from test files
    - API calls to the RAG system
    - Metric calculations
    - Result export

    Attributes:
        data_loader: Test data loader instance
        api_client: RAG API client instance
        exporter: Excel exporter instance
    """

    def __init__(
        self,
        test_data_path: Union[str, Path] = DEFAULT_TEST_DATA_PATH,
        api_base_url: Optional[str] = None,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        limit: Optional[int] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            test_data_path: Path to test data Excel file
            api_base_url: Base URL for RAG API (uses default if None)
            output_dir: Directory for output files
            limit: Optional limit on number of queries to evaluate
        """
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.limit = limit

        # Initialize components
        self.data_loader = TestDataLoader(self.test_data_path, limit=limit)

        if api_base_url:
            self.api_client = RAGAPIClient(base_url=api_base_url)
        else:
            self.api_client = RAGAPIClient()

        self.exporter = ExcelExporter(self.output_dir)

    def run(
        self,
        metrics: Union[str, List[str]] = "all",
        k: int = 5,
        score_threshold: float = 0.0,
        verbose: bool = False,
        export: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, EvaluationResult]:
        """
        Run evaluation with specified metrics.

        Args:
            metrics: Metric name(s) to run ("hit", "recall", "all", or list)
            k: Number of top results to consider
            score_threshold: Minimum similarity score threshold
            verbose: Print detailed per-query results
            export: Whether to export results to Excel
            output_path: Custom output path for results

        Returns:
            Dictionary mapping metric names to EvaluationResult objects
        """
        # Resolve metrics to run
        metric_instances = self._resolve_metrics(metrics)

        if not metric_instances:
            raise ValueError(f"No valid metrics found for: {metrics}")

        logger.info(f"Running evaluation with metrics: {[m.name for m in metric_instances]}")
        logger.info(f"K={k}, score_threshold={score_threshold}")

        # Collect query results (single pass through data, single API call per query)
        all_query_data = self._collect_query_results(k, score_threshold, verbose)

        # Calculate results for each metric
        results: Dict[str, EvaluationResult] = {}

        for metric in metric_instances:
            result = self._calculate_metric(metric, all_query_data, k, score_threshold)
            results[metric.short_name] = result

            # Log summary
            logger.info(f"\n{'='*50}")
            logger.info(f"{metric.name} Results:")
            logger.info(f"  Score: {result.score:.4f} ({result.score*100:.2f}%)")
            logger.info(f"  Queries: {result.total_queries}")
            logger.info(f"{'='*50}")

        # Export results
        if export:
            self._export_results(results, output_path)

        return results

    def _resolve_metrics(
        self, metrics: Union[str, List[str]]
    ) -> List[RetrievalMetric]:
        """Resolve metric specification to list of metric instances."""
        if isinstance(metrics, str):
            if metrics.lower() == "all":
                return MetricRegistry.get_all_instances()
            else:
                instance = MetricRegistry.get_instance(metrics)
                return [instance] if instance else []
        else:
            instances = []
            for name in metrics:
                instance = MetricRegistry.get_instance(name)
                if instance:
                    instances.append(instance)
                else:
                    logger.warning(f"Unknown metric: {name}")
            return instances

    def _collect_query_results(
        self,
        k: int,
        score_threshold: float,
        verbose: bool,
    ) -> List[dict]:
        """
        Collect query results with single API call per query.

        Returns list of dicts with query info and retrieved IDs.
        """
        total_queries = len(self.data_loader)
        all_query_data = []
        processed = 0

        for idx, query, ground_truth_ids, metadata in self.data_loader.iter_queries():
            # Single API call per query
            retrieved_ids = self.api_client.search(
                query=query,
                top_k=k,
                score_threshold=score_threshold,
            )

            processed += 1

            query_data = {
                "idx": idx,
                "query": query,
                "ground_truth_ids": ground_truth_ids,
                "retrieved_ids": retrieved_ids,
                "metadata": metadata,
            }
            all_query_data.append(query_data)

            if verbose:
                logger.info(
                    f"[{processed}/{total_queries}] Query: '{query[:50]}...' | "
                    f"GT: {ground_truth_ids} | Retrieved: {retrieved_ids}"
                )
            elif processed % 10 == 0:
                logger.info(f"Processed {processed}/{total_queries} queries...")

        logger.info(f"Collected results for {len(all_query_data)} queries")
        return all_query_data

    def _calculate_metric(
        self,
        metric: RetrievalMetric,
        query_data: List[dict],
        k: int,
        score_threshold: float,
    ) -> EvaluationResult:
        """Calculate a single metric from collected query data."""
        query_results: List[QueryResult] = []

        for qd in query_data:
            score, extra_metadata = metric.calculate_query_score(
                qd["ground_truth_ids"],
                qd["retrieved_ids"],
            )

            # Merge metadata
            combined_metadata = {**qd["metadata"], **extra_metadata}

            query_result = QueryResult(
                query_index=qd["idx"],
                query=qd["query"],
                ground_truth_ids=qd["ground_truth_ids"],
                retrieved_ids=qd["retrieved_ids"],
                score=score,
                metadata=combined_metadata,
            )
            query_results.append(query_result)

        # Aggregate scores
        summary = metric.aggregate_scores(query_results, k, score_threshold)

        return EvaluationResult(
            metric_name=metric.name,
            k=k,
            score=summary.score,
            query_results=query_results,
            summary=summary,
            test_data_path=str(self.test_data_path),
        )

    def _export_results(
        self,
        results: Dict[str, EvaluationResult],
        output_path: Optional[Path],
    ) -> None:
        """Export evaluation results to Excel."""
        result_list = list(results.values())

        if len(result_list) == 1:
            path = self.exporter.export_single_metric(result_list[0], output_path)
        else:
            path = self.exporter.export_multiple_metrics(result_list, output_path)

        logger.info(f"Results exported to: {path}")


def run_evaluation(
    metrics: Union[str, List[str]] = "all",
    k: int = 5,
    score_threshold: float = 0.0,
    test_data_path: Optional[Path] = None,
    api_base_url: Optional[str] = None,
    limit: Optional[int] = None,
    verbose: bool = False,
    output_path: Optional[Path] = None,
) -> Dict[str, EvaluationResult]:
    """
    Convenience function to run evaluation.

    Args:
        metrics: Metric name(s) to run ("hit", "recall", "all", or list)
        k: Number of top results to consider
        score_threshold: Minimum similarity score threshold
        test_data_path: Path to test data file
        api_base_url: Base URL for RAG API
        limit: Limit number of queries to evaluate
        verbose: Print detailed per-query results
        output_path: Custom output path for results

    Returns:
        Dictionary mapping metric names to EvaluationResult objects
    """
    evaluator = Evaluator(
        test_data_path=test_data_path or DEFAULT_TEST_DATA_PATH,
        api_base_url=api_base_url,
        limit=limit,
    )

    return evaluator.run(
        metrics=metrics,
        k=k,
        score_threshold=score_threshold,
        verbose=verbose,
        output_path=output_path,
    )
