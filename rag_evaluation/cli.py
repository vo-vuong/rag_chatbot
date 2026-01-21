"""
Command-line interface for RAG evaluation.

Provides a unified CLI for running retrieval metrics evaluation.

Usage:
    # Run Hit@K
    conda activate rag_chatbot && python -m rag_evaluation --metric hit --k 5

    # Run Recall@K
    conda activate rag_chatbot && python -m rag_evaluation --metric recall --k 10

    # Run all metrics
    conda activate rag_chatbot && python -m rag_evaluation --metric all --k 5 -v

    # Run multiple specific metrics
    conda activate rag_chatbot && python -m rag_evaluation --metric hit recall --k 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.constants import API_BASE_URL
from rag_evaluation.evaluator import Evaluator, DEFAULT_TEST_DATA_PATH, DEFAULT_OUTPUT_DIR
from rag_evaluation.metrics.registry import MetricRegistry

# Import metrics to register them
import rag_evaluation.metrics.hit_at_k  # noqa: F401
import rag_evaluation.metrics.recall_at_k  # noqa: F401


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    available_metrics = MetricRegistry.list_metrics()

    parser = argparse.ArgumentParser(
        description="RAG Retrieval Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run Hit@K with k=5
  python -m rag_evaluation --metric hit --k 5

  # Run Recall@K with k=10 and verbose output
  python -m rag_evaluation --metric recall --k 10 -v

  # Run all metrics
  python -m rag_evaluation --metric all --k 5

  # Run multiple metrics
  python -m rag_evaluation --metric hit recall --k 5

Available metrics: {', '.join(available_metrics)}
        """,
    )

    parser.add_argument(
        "-m", "--metric",
        nargs="+",
        default=["all"],
        help=f"Metric(s) to run: {', '.join(available_metrics)}, or 'all' (default: all)",
    )

    parser.add_argument(
        "-k", "--k",
        type=int,
        default=5,
        help="Number of top results to consider (default: 5)",
    )

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold (default: 0.0)",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help=f"Path to test data Excel file (default: {DEFAULT_TEST_DATA_PATH.name})",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=API_BASE_URL,
        help=f"RAG API base URL (default: {API_BASE_URL})",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (default: all)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed per-query results",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output Excel file path (default: auto-generated in results/)",
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Don't export results to Excel",
    )

    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()

    # Handle --list-metrics
    if args.list_metrics:
        print("Available metrics:")
        for name in MetricRegistry.list_metrics():
            metric = MetricRegistry.get_instance(name)
            if metric:
                print(f"  {name}: {metric.name}")
        return 0

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Resolve test data path
    test_data_path = Path(args.test_data) if args.test_data else DEFAULT_TEST_DATA_PATH

    # Resolve output path
    output_path = Path(args.output) if args.output else None

    # Resolve metrics
    metrics = args.metric[0] if len(args.metric) == 1 else args.metric

    try:
        # Create evaluator
        evaluator = Evaluator(
            test_data_path=test_data_path,
            api_base_url=args.api_url,
            output_dir=DEFAULT_OUTPUT_DIR,
            limit=args.limit,
        )

        # Run evaluation
        results = evaluator.run(
            metrics=metrics,
            k=args.k,
            score_threshold=args.threshold,
            verbose=args.verbose,
            export=not args.no_export,
            output_path=output_path,
        )

        # Print final summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        for name, result in results.items():
            print(f"\n{result.metric_name}:")
            print(f"  Score: {result.score:.4f} ({result.score*100:.2f}%)")
            print(f"  Queries: {result.total_queries}")

            for stat_name, stat_value in result.summary.additional_stats.items():
                print(f"  {stat_name}: {stat_value}")

        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
