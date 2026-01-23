"""
Command-line interface for RAG evaluation.

Provides a unified CLI for running retrieval and generation metrics evaluation.

Usage:
    # Run Hit@K
    conda activate rag_chatbot && python -m rag_evaluation --metric hit --k 5

    # Run Recall@K
    conda activate rag_chatbot && python -m rag_evaluation --metric recall --k 10

    # Run all retrieval metrics
    conda activate rag_chatbot && python -m rag_evaluation --metric all --k 5 -v

    # Run Faithfulness (generation metric)
    conda activate rag_chatbot && python -m rag_evaluation --metric faithfulness --k 5

    # Run with limit
    conda activate rag_chatbot && python -m rag_evaluation --metric all --k 5 --limit 5
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
import rag_evaluation.metrics.precision_at_k  # noqa: F401
import rag_evaluation.metrics.f1_at_k  # noqa: F401
import rag_evaluation.metrics.mrr_at_k  # noqa: F401

# Generation metrics (handled separately)
GENERATION_METRICS = ["faithfulness", "response_relevancy"]


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
    all_metrics = available_metrics + GENERATION_METRICS

    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework (Retrieval + Generation Metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run Hit@K with k=5
  python -m rag_evaluation --metric hit --k 5

  # Run Recall@K with k=10 and verbose output
  python -m rag_evaluation --metric recall --k 10 -v

  # Run all retrieval metrics
  python -m rag_evaluation --metric all --k 5

  # Run all generation metrics
  python -m rag_evaluation --metric all_generation --k 5

  # Run Faithfulness (generation metric)
  python -m rag_evaluation --metric faithfulness --k 5

  # Run Response Relevancy (generation metric)
  python -m rag_evaluation --metric response_relevancy --k 5

  # Run Faithfulness with custom model
  python -m rag_evaluation --metric faithfulness --k 5 --model gpt-4o

  # Run Response Relevancy with custom embedding model
  python -m rag_evaluation --metric response_relevancy --k 5 --embedding-model text-embedding-3-large

Available retrieval metrics: {', '.join(available_metrics)}
Available generation metrics: {', '.join(GENERATION_METRICS)}
        """,
    )

    parser.add_argument(
        "-m", "--metric",
        nargs="+",
        default=["all"],
        help=f"Metric(s) to run: {', '.join(all_metrics)}, 'all' (retrieval), or 'all_generation' (default: all)",
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

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for generation metrics (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model for response_relevancy metric (default: text-embedding-3-small)",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()

    # Handle --list-metrics
    if args.list_metrics:
        print("Available retrieval metrics:")
        for name in MetricRegistry.list_metrics():
            metric = MetricRegistry.get_instance(name)
            if metric:
                print(f"  {name}: {metric.name}")
        print("\nAvailable generation metrics:")
        for name in GENERATION_METRICS:
            if name == "faithfulness":
                print(f"  {name}: Faithfulness (RAGAS)")
            elif name == "response_relevancy":
                print(f"  {name}: Response Relevancy (RAGAS)")
        return 0

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Resolve test data path
    test_data_path = Path(args.test_data) if args.test_data else DEFAULT_TEST_DATA_PATH

    # Resolve output path
    output_path = Path(args.output) if args.output else None

    # Resolve metrics
    metrics = args.metric[0] if len(args.metric) == 1 else args.metric

    # Expand "all_generation" to all generation metrics
    requested_metrics = [metrics] if isinstance(metrics, str) else metrics
    if "all_generation" in requested_metrics:
        # Replace "all_generation" with all generation metric names
        requested_metrics = [
            m for m in requested_metrics if m != "all_generation"
        ] + GENERATION_METRICS

    # Check if any generation metrics are requested
    generation_requested = any(m in GENERATION_METRICS for m in requested_metrics)
    retrieval_requested = any(
        m in MetricRegistry.list_metrics() or m == "all"
        for m in requested_metrics
    )

    try:
        # Run generation metrics (Faithfulness)
        if generation_requested:
            from rag_evaluation.generation_evaluator import GenerationEvaluator

            gen_evaluator = GenerationEvaluator(
                test_data_path=test_data_path,
                api_base_url=args.api_url,
                limit=args.limit,
            )

            for metric_name in requested_metrics:
                if metric_name in GENERATION_METRICS:
                    result = gen_evaluator.run(
                        metric=metric_name,
                        top_k=args.k,
                        score_threshold=args.threshold,
                        verbose=args.verbose,
                        model_name=args.model,
                        embedding_model=args.embedding_model,
                    )

                    print("\n" + "=" * 60)
                    print(f"{result['metric_name'].upper()} EVALUATION COMPLETE")
                    print("=" * 60)
                    print(f"\n{result['metric_name']}:")
                    print(f"  Score: {result['summary']['score']:.4f} "
                          f"({result['summary']['score']*100:.2f}%)")
                    print(f"  Queries: {result['summary']['total_queries']}")
                    print(f"  Model: {result['summary']['model']}")
                    if 'embedding_model' in result['summary']:
                        print(f"  Embedding Model: {result['summary']['embedding_model']}")
                    print("=" * 60)

        # Run retrieval metrics
        if retrieval_requested and not (
            metrics == "faithfulness" or
            metrics == "response_relevancy" or
            metrics == "all_generation" or
            (isinstance(metrics, list) and set(metrics).issubset(set(GENERATION_METRICS + ["all_generation"])))
        ):
            # Filter out generation metrics for retrieval evaluator
            retrieval_metrics = metrics
            if isinstance(metrics, list):
                retrieval_metrics = [
                    m for m in metrics
                    if m not in GENERATION_METRICS and m != "all_generation"
                ]
                if not retrieval_metrics:
                    return 0
                retrieval_metrics = retrieval_metrics[0] if len(retrieval_metrics) == 1 else retrieval_metrics

            evaluator = Evaluator(
                test_data_path=test_data_path,
                api_base_url=args.api_url,
                output_dir=DEFAULT_OUTPUT_DIR,
                limit=args.limit,
            )

            results = evaluator.run(
                metrics=retrieval_metrics,
                k=args.k,
                score_threshold=args.threshold,
                verbose=args.verbose,
                export=not args.no_export,
                output_path=output_path,
            )

            print("\n" + "=" * 60)
            print("RETRIEVAL EVALUATION COMPLETE")
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
