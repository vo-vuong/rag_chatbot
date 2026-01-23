"""
Excel export utilities for evaluation results.

Provides standardized export of evaluation results to Excel format
with summary and per-query detail sheets.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from rag_evaluation.base.evaluation_result import EvaluationResult
from rag_evaluation.base.generation_metric_interface import GenerationQueryResult

logger = logging.getLogger(__name__)


class ExcelExporter:
    """
    Exporter for evaluation results to Excel format.

    Creates Excel files with multiple sheets:
    - Summary: Aggregated metrics and statistics
    - Per-Query Results: Detailed results for each query
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the Excel exporter.

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_single_metric(
        self,
        result: EvaluationResult,
        output_path: Path = None,
    ) -> Path:
        """
        Export single metric evaluation result to Excel.

        Args:
            result: Evaluation result to export
            output_path: Optional custom output path

        Returns:
            Path to the created Excel file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.metric_name}_k{result.k}_results_{timestamp}.xlsx"
            output_path = self.output_dir / filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create summary DataFrame
        summary_df = self._create_summary_df(result)

        # Create per-query results DataFrame
        results_df = pd.DataFrame(result.get_query_results_as_dicts())

        # Write to Excel
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            results_df.to_excel(writer, sheet_name="Per-Query Results", index=False)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def export_multiple_metrics(
        self,
        results: List[EvaluationResult],
        output_path: Path = None,
    ) -> Path:
        """
        Export multiple metric results to a single Excel file.

        Args:
            results: List of evaluation results to export
            output_path: Optional custom output path

        Returns:
            Path to the created Excel file
        """
        if not results:
            raise ValueError("No results to export")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            k = results[0].k
            filename = f"retrieval_combined_k{k}_results_{timestamp}.xlsx"
            output_path = self.output_dir / filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create combined summary DataFrame
        combined_summary_df = self._create_combined_summary_df(results)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            combined_summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Add per-query sheet for each metric
            for result in results:
                sheet_name = f"Per-Query ({result.metric_name})"
                # Truncate sheet name if too long (Excel limit: 31 chars)
                sheet_name = sheet_name[:31]
                results_df = pd.DataFrame(result.get_query_results_as_dicts())
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Combined results saved to: {output_path}")
        return output_path

    def _create_summary_df(self, result: EvaluationResult) -> pd.DataFrame:
        """Create summary DataFrame for a single metric result."""
        summary = result.summary
        data = {
            "Metric": [
                "Evaluation Date",
                "Metric Name",
                "K Value",
                "Score Threshold",
                "Total Queries",
                "Score",
                "Score (%)",
            ],
            "Value": [
                result.evaluation_date.strftime("%Y-%m-%d %H:%M:%S"),
                result.metric_name,
                result.k,
                summary.score_threshold,
                result.total_queries,
                f"{result.score:.4f}",
                f"{result.score * 100:.2f}%",
            ],
        }

        # Add additional stats
        for key, value in summary.additional_stats.items():
            data["Metric"].append(key)
            data["Value"].append(value)

        return pd.DataFrame(data)

    def _create_combined_summary_df(
        self, results: List[EvaluationResult]
    ) -> pd.DataFrame:
        """Create combined summary DataFrame for multiple metrics."""
        first_result = results[0]

        data: Dict[str, List[Any]] = {
            "Metric": [
                "Evaluation Date",
                "K Value",
                "Score Threshold",
                "Total Queries",
                "",
            ],
            "Value": [
                first_result.evaluation_date.strftime("%Y-%m-%d %H:%M:%S"),
                first_result.k,
                first_result.summary.score_threshold,
                first_result.total_queries,
                "",
            ],
        }

        for result in results:
            # Add section header
            data["Metric"].append(f"--- {result.metric_name} ---")
            data["Value"].append("")

            # Add metric score
            data["Metric"].append(f"{result.metric_name} Score")
            data["Value"].append(f"{result.score:.4f}")

            data["Metric"].append(f"{result.metric_name} Score (%)")
            data["Value"].append(f"{result.score * 100:.2f}%")

            # Add additional stats
            for key, value in result.summary.additional_stats.items():
                data["Metric"].append(key)
                data["Value"].append(value)

            data["Metric"].append("")
            data["Value"].append("")

        return pd.DataFrame(data)

    def export_generation_result(
        self,
        result: Dict[str, Any],
        query_results: List[GenerationQueryResult],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export generation metric evaluation result to Excel.

        Args:
            result: Dictionary with evaluation results (summary, config, etc.)
            query_results: List of GenerationQueryResult objects
            output_path: Optional custom output path

        Returns:
            Path to the created Excel file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metric_name = result.get("metric_name", "generation")
            top_k = result.get("config", {}).get("top_k", 5)
            filename = f"{metric_name}_k{top_k}_results_{timestamp}.xlsx"
            output_path = self.output_dir / filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create summary DataFrame
        summary_df = self._create_generation_summary_df(result)

        # Create per-query results DataFrame
        results_df = pd.DataFrame([qr.to_dict() for qr in query_results])

        # Write to Excel
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            results_df.to_excel(writer, sheet_name="Per-Query Results", index=False)

        logger.info(f"Generation results saved to: {output_path}")
        return output_path

    def _create_generation_summary_df(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Create summary DataFrame for generation metric result."""
        summary = result.get("summary", {})
        config = result.get("config", {})

        data = {
            "Metric": [
                "Evaluation Date",
                "Metric Name",
                "Model",
                "Top K",
                "Score Threshold",
                "Total Queries",
                "Valid Queries",
                "Score",
                "Score (%)",
                "Min Score",
                "Max Score",
            ],
            "Value": [
                result.get("evaluation_date", ""),
                result.get("metric_name", ""),
                config.get("model", ""),
                config.get("top_k", ""),
                config.get("score_threshold", ""),
                summary.get("total_queries", 0),
                summary.get("valid_queries", 0),
                f"{summary.get('score', 0):.4f}",
                f"{summary.get('score', 0) * 100:.2f}%",
                f"{summary.get('min_score', 0):.4f}",
                f"{summary.get('max_score', 0):.4f}",
            ],
        }

        return pd.DataFrame(data)

    def export_multiple_generation_results(
        self,
        results: List[Dict[str, Any]],
        query_results_list: List[List[GenerationQueryResult]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export multiple generation metric results to a single Excel file.

        Args:
            results: List of result dictionaries (one per metric)
            query_results_list: List of query result lists (one per metric)
            output_path: Optional custom output path

        Returns:
            Path to the created Excel file
        """
        if not results:
            raise ValueError("No results to export")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            top_k = results[0].get("config", {}).get("top_k", 5)
            filename = f"generation_combined_k{top_k}_results_{timestamp}.xlsx"
            output_path = self.output_dir / filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create combined summary DataFrame
        combined_summary_df = self._create_combined_generation_summary_df(results)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            combined_summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Add per-query sheet for each metric
            for result, query_results in zip(results, query_results_list):
                metric_name = result.get("metric_name", "Unknown")
                sheet_name = f"Per-Query ({metric_name})"
                # Truncate sheet name if too long (Excel limit: 31 chars)
                sheet_name = sheet_name[:31]
                results_df = pd.DataFrame([qr.to_dict() for qr in query_results])
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Combined generation results saved to: {output_path}")
        return output_path

    def _create_combined_generation_summary_df(
        self, results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create combined summary DataFrame for multiple generation metrics."""
        first_result = results[0]
        first_config = first_result.get("config", {})

        data: Dict[str, List[Any]] = {
            "Metric": [
                "Evaluation Date",
                "Model",
                "Top K",
                "Score Threshold",
                "Total Queries",
                "",
            ],
            "Value": [
                first_result.get("evaluation_date", ""),
                first_config.get("model", ""),
                first_config.get("top_k", ""),
                first_config.get("score_threshold", ""),
                first_result.get("summary", {}).get("total_queries", 0),
                "",
            ],
        }

        for result in results:
            metric_name = result.get("metric_name", "Unknown")
            summary = result.get("summary", {})

            # Add section header
            data["Metric"].append(f"--- {metric_name} ---")
            data["Value"].append("")

            # Add metric score
            data["Metric"].append(f"{metric_name} Score")
            data["Value"].append(f"{summary.get('score', 0):.4f}")

            data["Metric"].append(f"{metric_name} Score (%)")
            data["Value"].append(f"{summary.get('score', 0) * 100:.2f}%")

            data["Metric"].append("Min Score")
            data["Value"].append(f"{summary.get('min_score', 0):.4f}")

            data["Metric"].append("Max Score")
            data["Value"].append(f"{summary.get('max_score', 0):.4f}")

            data["Metric"].append("")
            data["Value"].append("")

        return pd.DataFrame(data)
