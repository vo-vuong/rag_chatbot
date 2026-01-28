"""
Command-line interface for RAG evaluation.

Provides a unified CLI for running retrieval and generation metrics evaluation.

Usage:
    # Collect responses (saves to JSON for later evaluation)
    python -m rag_evaluation collect --mode chat --k 5
    python -m rag_evaluation collect --mode search --k 5

    # Run evaluation (from API - original behavior)
    python -m rag_evaluation eval --metric hit --k 5
    python -m rag_evaluation eval --metric all_generation --k 5

    # Run evaluation from pre-collected responses
    python -m rag_evaluation eval --metric faithfulness --from-responses responses.json

    # Shortcut: run eval without subcommand (backward compatible)
    python -m rag_evaluation --metric hit --k 5
"""

import argparse
import logging
import os
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
GENERATION_METRICS = [
    "faithfulness", "response_relevancy", "context_precision",
    "context_recall", "answer_correctness"
]

DEFAULT_RESPONSES_DIR = Path(__file__).parent / "responses"


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework (Retrieval + Generation Metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Collect subcommand
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect API responses and save to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect chat responses (for all generation metrics)
  python -m rag_evaluation collect --k 5

  # With custom output path
  python -m rag_evaluation collect --k 5 -o my_responses.json

  # With limit for testing
  python -m rag_evaluation collect --k 5 --limit 5 -v
        """,
    )
    _add_collect_args(collect_parser)

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from API (original behavior)
  python -m rag_evaluation eval --metric hit --k 5
  python -m rag_evaluation eval --metric all_generation --k 5

  # Run from pre-collected responses (saves API calls)
  python -m rag_evaluation eval --metric all_generation --from-chat-responses chat_responses.json
  python -m rag_evaluation eval --metric faithfulness --from-chat-responses chat_responses.json
        """,
    )
    _add_eval_args(eval_parser)

    # For backward compatibility, also add eval args to main parser
    _add_eval_args(parser)

    return parser


def _add_collect_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for collect subcommand."""
    parser.add_argument(
        "-k", "--k",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)",
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
        help="Limit number of queries to collect (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=7.0,
        help="Delay in seconds between queries (default: 7.0)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress",
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for eval subcommand."""
    available_metrics = MetricRegistry.list_metrics()
    all_metrics = available_metrics + GENERATION_METRICS

    parser.add_argument(
        "-m", "--metric",
        nargs="+",
        default=["all"],
        help=f"Metric(s) to run: {', '.join(all_metrics)}, 'all', 'all_retrieval', 'all_generation'",
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
        "--delay",
        type=float,
        default=7.0,
        help="Delay in seconds between queries (default: 7.0)",
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
        help="Output Excel file path (default: auto-generated)",
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
        help="Embedding model for response_relevancy (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--from-chat-responses",
        type=str,
        default=None,
        help="Path to chat responses JSON file (for all generation metrics)",
    )


def run_collect(args: argparse.Namespace) -> int:
    """Run the collect subcommand."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Check memory setting
    memory_enabled = os.getenv("AGENT_MEMORY_ENABLED", "true").lower()
    if memory_enabled in ("true", "1", "yes"):
        print("\n" + "!" * 70)
        print("ERROR: AGENT_MEMORY_ENABLED=true")
        print("Set AGENT_MEMORY_ENABLED=false in .env and restart API server.")
        print("!" * 70 + "\n")
        return 1

    from rag_evaluation.data.response_collector import ResponseCollector

    test_data_path = Path(args.test_data) if args.test_data else DEFAULT_TEST_DATA_PATH
    output_path = Path(args.output) if args.output else None

    try:
        collector = ResponseCollector(
            test_data_path=test_data_path,
            api_base_url=args.api_url,
            output_dir=DEFAULT_RESPONSES_DIR,
            limit=args.limit,
            delay=args.delay,
        )

        result_path = collector.collect(
            mode="chat",
            top_k=args.k,
            score_threshold=args.threshold,
            output_path=output_path,
            verbose=args.verbose,
        )

        print("\n" + "=" * 60)
        print("RESPONSE COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Output: {result_path}")
        print("=" * 60)

        print("\nTo evaluate with this file:")
        print(f"  python -m rag_evaluation eval --metric all_generation --from-chat-responses {result_path.name}")

        return 0

    except Exception as e:
        logger.exception(f"Collection failed: {e}")
        return 1


def run_eval(args: argparse.Namespace) -> int:
    """Run the eval subcommand."""
    # Handle --list-metrics
    if getattr(args, "list_metrics", False):
        print("Available retrieval metrics:")
        for name in MetricRegistry.list_metrics():
            metric = MetricRegistry.get_instance(name)
            if metric:
                print(f"  {name}: {metric.name}")
        print("\nAvailable generation metrics:")
        for name in GENERATION_METRICS:
            print(f"  {name}: {name.replace('_', ' ').title()} (RAGAS)")
        return 0

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Handle response file option
    from_chat_responses = getattr(args, "from_chat_responses", None)

    # Check memory setting (skip if using response files)
    if not from_chat_responses:
        memory_enabled = os.getenv("AGENT_MEMORY_ENABLED", "true").lower()
        if memory_enabled in ("true", "1", "yes"):
            print("\n" + "!" * 70)
            print("ERROR: AGENT_MEMORY_ENABLED=true")
            print("Set AGENT_MEMORY_ENABLED=false in .env and restart API server.")
            print("!" * 70 + "\n")
            return 1

    test_data_path = Path(args.test_data) if args.test_data else DEFAULT_TEST_DATA_PATH
    output_path = Path(args.output) if args.output else None

    # Resolve metrics
    metrics = args.metric[0] if len(args.metric) == 1 else args.metric
    requested_metrics = [metrics] if isinstance(metrics, str) else metrics

    # Expand "all" to both retrieval and generation
    if "all" in requested_metrics:
        requested_metrics = [
            m for m in requested_metrics if m != "all"
        ] + ["all_retrieval"] + GENERATION_METRICS

    # Expand "all_generation"
    if "all_generation" in requested_metrics:
        requested_metrics = [
            m for m in requested_metrics if m != "all_generation"
        ] + GENERATION_METRICS

    # Handle "all_retrieval"
    has_all_retrieval = "all_retrieval" in requested_metrics
    if has_all_retrieval:
        requested_metrics = [m for m in requested_metrics if m != "all_retrieval"]

    # Determine what's requested
    generation_requested = any(m in GENERATION_METRICS for m in requested_metrics)
    retrieval_requested = has_all_retrieval or any(
        m in MetricRegistry.list_metrics() for m in requested_metrics
    )

    try:
        # Run generation metrics
        if generation_requested:
            _run_generation_eval(args, requested_metrics, from_chat_responses, output_path)

        # Run retrieval metrics
        if retrieval_requested:
            _run_retrieval_eval(
                args, requested_metrics, has_all_retrieval,
                test_data_path, output_path
            )

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


def _run_generation_eval(
    args: argparse.Namespace,
    requested_metrics: list,
    from_chat_responses: str,
    output_path: Path,
) -> None:
    """Run generation metrics evaluation."""
    from rag_evaluation.generation_evaluator import GenerationEvaluator
    from rag_evaluation.export.excel_exporter import ExcelExporter
    from rag_evaluation.data.response_collector import load_responses

    test_data_path = Path(args.test_data) if args.test_data else DEFAULT_TEST_DATA_PATH

    gen_evaluator = GenerationEvaluator(
        test_data_path=test_data_path,
        api_base_url=args.api_url,
        limit=args.limit,
        delay=0.0,
    )

    generation_results = []
    generation_query_results = []

    # Load response file if provided
    responses_data = None
    if from_chat_responses:
        chat_path = Path(from_chat_responses)
        if not chat_path.is_absolute():
            chat_path = DEFAULT_RESPONSES_DIR / chat_path
        responses_data = load_responses(chat_path)
        print(f"\nLoaded chat responses from: {chat_path}")
        print(f"  Queries: {len(responses_data['responses'])}")

    for metric_name in requested_metrics:
        if metric_name not in GENERATION_METRICS:
            continue

        if responses_data:
            result = gen_evaluator.run_from_responses(
                responses_data=responses_data,
                metric=metric_name,
                verbose=args.verbose,
                model_name=args.model,
                embedding_model=args.embedding_model,
                export=False,
            )
        else:
            result = gen_evaluator.run(
                metric=metric_name,
                top_k=args.k,
                score_threshold=args.threshold,
                verbose=args.verbose,
                model_name=args.model,
                embedding_model=args.embedding_model,
                export=False,
            )

        generation_results.append(result)
        generation_query_results.append(result["query_results"])

        print("\n" + "=" * 60)
        print(f"{result['metric_name'].upper()} EVALUATION COMPLETE")
        print("=" * 60)
        print(f"  Score: {result['summary']['score']:.4f} "
              f"({result['summary']['score']*100:.2f}%)")
        print(f"  Queries: {result['summary']['total_queries']}")
        print(f"  Model: {result['summary']['model']}")
        if 'embedding_model' in result['summary']:
            print(f"  Embedding: {result['summary']['embedding_model']}")
        print("=" * 60)

    # Export combined results
    if generation_results and not args.no_export:
        exporter = ExcelExporter(DEFAULT_OUTPUT_DIR)
        if len(generation_results) == 1:
            export_path = exporter.export_generation_result(
                generation_results[0],
                generation_query_results[0],
                output_path,
            )
        else:
            export_path = exporter.export_multiple_generation_results(
                generation_results,
                generation_query_results,
                output_path,
            )
        print(f"\nGeneration results exported to: {export_path}")


def _run_retrieval_eval(
    args: argparse.Namespace,
    requested_metrics: list,
    has_all_retrieval: bool,
    test_data_path: Path,
    output_path: Path,
) -> None:
    """Run retrieval metrics evaluation."""
    if has_all_retrieval:
        retrieval_metrics = "all"
    else:
        retrieval_metrics = [
            m for m in requested_metrics
            if m in MetricRegistry.list_metrics()
        ]
        if not retrieval_metrics:
            return
        if len(retrieval_metrics) == 1:
            retrieval_metrics = retrieval_metrics[0]

    evaluator = Evaluator(
        test_data_path=test_data_path,
        api_base_url=args.api_url,
        output_dir=DEFAULT_OUTPUT_DIR,
        limit=args.limit,
        delay=args.delay,
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


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Route to appropriate handler
    if args.command == "collect":
        return run_collect(args)
    elif args.command == "eval":
        return run_eval(args)
    else:
        # Backward compatible: no subcommand means eval
        return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
