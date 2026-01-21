"""
Retrieval metrics for RAG evaluation.

This module provides Hit@K and Recall@K metric calculations for evaluating retrieval performance.

Usage:
    # Hit@K (default)
    conda activate rag_chatbot && python rag_evaluation/metrics/retrieval_metrics.py --k 5
    conda activate rag_chatbot && python rag_evaluation/metrics/retrieval_metrics.py --k 10 --limit 50

    # Recall@K
    conda activate rag_chatbot && python rag_evaluation/metrics/retrieval_metrics.py --metric recall --k 5

    # Both metrics (single execution, optimized API calls)
    conda activate rag_chatbot && python rag_evaluation/metrics/retrieval_metrics.py --metric all --k 5
    conda activate rag_chatbot && python rag_evaluation/metrics/retrieval_metrics.py --metric all --k 10 -v
"""

import argparse
import ast
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.constants import API_BASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TEST_DATA_PATH = (
    Path(__file__).parent.parent / "prepare_testing_data" / "qr_smartphone_dataset.xlsx"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "results"


def parse_point_ids(point_ids_value) -> List[int]:
    """
    Parse Point_ids from various formats to list of integers.

    Handles:
    - Single integer: 5 -> [5]
    - String of single int: "5" -> [5]
    - String list: "[1, 2, 3]" -> [1, 2, 3]
    - Actual list: [1, 2, 3] -> [1, 2, 3]
    - Comma-separated: "1, 2, 3" -> [1, 2, 3]

    Args:
        point_ids_value: Raw value from DataFrame

    Returns:
        List of integer point IDs
    """
    if point_ids_value is None or (isinstance(point_ids_value, float) and pd.isna(point_ids_value)):
        return []

    if isinstance(point_ids_value, int):
        return [point_ids_value]

    if isinstance(point_ids_value, list):
        return [int(x) for x in point_ids_value]

    if isinstance(point_ids_value, str):
        point_ids_value = point_ids_value.strip()
        # Try parsing as Python literal (list)
        if point_ids_value.startswith("["):
            try:
                parsed = ast.literal_eval(point_ids_value)
                return [int(x) for x in parsed]
            except (ValueError, SyntaxError):
                pass
        # Try comma-separated format
        if "," in point_ids_value:
            return [int(x.strip()) for x in point_ids_value.split(",") if x.strip()]
        # Single value
        return [int(point_ids_value)]

    return []


def search_rag_api(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    api_base_url: str = API_BASE_URL,
) -> List[int]:
    """
    Search using RAG API endpoint and return retrieved point IDs.

    Args:
        query: Search query text
        top_k: Number of results to retrieve
        score_threshold: Minimum similarity score threshold
        api_base_url: Base URL for the API

    Returns:
        List of retrieved point IDs
    """
    url = f"{api_base_url}/api/v1/rag/search"
    payload = {
        "query": query,
        "collection_type": "text",
        "top_k": top_k,
        "score_threshold": score_threshold,
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract point_ids from results
        results = data.get("results", [])
        point_ids = [r.get("point_id") for r in results if r.get("point_id") is not None]
        return point_ids

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for query '{query[:50]}...': {e}")
        return []


def hit_k(
    test_data_path: Path = DEFAULT_TEST_DATA_PATH,
    k: int = 5,
    score_threshold: float = 0.0,
    api_base_url: str = API_BASE_URL,
    limit: Optional[int] = None,
    verbose: bool = False,
    output_path: Optional[Path] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Hit@K metric for RAG retrieval evaluation.

    Hit@K measures the proportion of queries where at least one relevant document
    appears in the top-K retrieved results.

    Args:
        test_data_path: Path to Excel file with test data
        k: Number of top results to consider
        score_threshold: Minimum similarity score threshold
        api_base_url: Base URL for the RAG API
        limit: Limit number of queries to evaluate (None = all)
        verbose: Print detailed per-query results
        output_path: Path to save results Excel file (None = no save)

    Returns:
        Tuple of (Hit@K score, list of per-query result dicts)
    """
    # Load test data
    logger.info(f"Loading test data from: {test_data_path}")
    df = pd.read_excel(test_data_path)

    if limit:
        df = df.head(limit)
        logger.info(f"Limited to first {limit} queries")

    total_queries = len(df)
    logger.info(f"Total queries to evaluate: {total_queries}")

    hits = 0
    processed = 0
    results_list: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        query = row.get("Query", "")
        if not query or (isinstance(query, float) and pd.isna(query)):
            logger.warning(f"Skipping row {idx}: empty query")
            continue

        # Parse ground truth point IDs
        ground_truth_ids = parse_point_ids(row.get("Point_ids"))
        if not ground_truth_ids:
            logger.warning(f"Skipping row {idx}: no ground truth Point_ids")
            continue

        # Query the RAG API
        retrieved_ids = search_rag_api(
            query=query,
            top_k=k,
            score_threshold=score_threshold,
            api_base_url=api_base_url,
        )

        # Check for hit (any ground truth ID in retrieved results)
        is_hit = any(gt_id in retrieved_ids for gt_id in ground_truth_ids)
        if is_hit:
            hits += 1

        processed += 1

        # Store result for export
        result_record = {
            "query_index": idx,
            "query": query,
            "ground_truth_ids": str(ground_truth_ids),
            "retrieved_ids": str(retrieved_ids),
            "is_hit": is_hit,
            "difficulty": row.get("difficulty", ""),
            "source_files": row.get("Source_files", ""),
        }
        results_list.append(result_record)

        if verbose:
            status = "HIT" if is_hit else "MISS"
            logger.info(
                f"[{processed}/{total_queries}] {status} | "
                f"Query: '{query[:50]}...' | "
                f"GT: {ground_truth_ids} | Retrieved: {retrieved_ids}"
            )
        elif processed % 10 == 0:
            logger.info(f"Processed {processed}/{total_queries} queries...")

    if processed == 0:
        logger.warning("No valid queries processed")
        return 0.0, []

    hit_rate = hits / processed
    logger.info(f"\n{'='*50}")
    logger.info(f"Hit@{k} Results:")
    logger.info(f"  Hits: {hits}/{processed}")
    logger.info(f"  Hit Rate: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    logger.info(f"{'='*50}")

    # Save results to Excel if output path specified
    if output_path:
        save_results_to_excel(
            results_list=results_list,
            output_path=output_path,
            k=k,
            score_threshold=score_threshold,
            hits=hits,
            total=processed,
            hit_rate=hit_rate,
        )

    return hit_rate, results_list


def recall_k(
    test_data_path: Path = DEFAULT_TEST_DATA_PATH,
    k: int = 5,
    score_threshold: float = 0.0,
    api_base_url: str = API_BASE_URL,
    limit: Optional[int] = None,
    verbose: bool = False,
    output_path: Optional[Path] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Recall@K metric for RAG retrieval evaluation.

    Recall@K measures the average proportion of relevant documents that appear
    in the top-K retrieved results for each query.

    Formula: For each query, recall = (# of ground truth docs found in top-K) / (# of ground truth docs)
    Final score = average of all query recalls

    Args:
        test_data_path: Path to Excel file with test data
        k: Number of top results to consider
        score_threshold: Minimum similarity score threshold
        api_base_url: Base URL for the RAG API
        limit: Limit number of queries to evaluate (None = all)
        verbose: Print detailed per-query results
        output_path: Path to save results Excel file (None = no save)

    Returns:
        Tuple of (Recall@K score, list of per-query result dicts)
    """
    # Load test data
    logger.info(f"Loading test data from: {test_data_path}")
    df = pd.read_excel(test_data_path)

    if limit:
        df = df.head(limit)
        logger.info(f"Limited to first {limit} queries")

    total_queries = len(df)
    logger.info(f"Total queries to evaluate: {total_queries}")

    total_recall = 0.0
    processed = 0
    results_list: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        query = row.get("Query", "")
        if not query or (isinstance(query, float) and pd.isna(query)):
            logger.warning(f"Skipping row {idx}: empty query")
            continue

        # Parse ground truth point IDs
        ground_truth_ids = parse_point_ids(row.get("Point_ids"))
        if not ground_truth_ids:
            logger.warning(f"Skipping row {idx}: no ground truth Point_ids")
            continue

        # Query the RAG API
        retrieved_ids = search_rag_api(
            query=query,
            top_k=k,
            score_threshold=score_threshold,
            api_base_url=api_base_url,
        )

        # Count how many ground truth IDs were retrieved
        hits_count = sum(1 for gt_id in ground_truth_ids if gt_id in retrieved_ids)
        query_recall = hits_count / len(ground_truth_ids)
        total_recall += query_recall

        processed += 1

        # Store result for export
        result_record = {
            "query_index": idx,
            "query": query,
            "ground_truth_ids": str(ground_truth_ids),
            "retrieved_ids": str(retrieved_ids),
            "ground_truth_count": len(ground_truth_ids),
            "hits_count": hits_count,
            "query_recall": query_recall,
            "difficulty": row.get("difficulty", ""),
            "source_files": row.get("Source_files", ""),
        }
        results_list.append(result_record)

        if verbose:
            logger.info(
                f"[{processed}/{total_queries}] Recall: {query_recall:.4f} | "
                f"Query: '{query[:50]}...' | "
                f"GT: {ground_truth_ids} | Retrieved: {retrieved_ids} | "
                f"Hits: {hits_count}/{len(ground_truth_ids)}"
            )
        elif processed % 10 == 0:
            logger.info(f"Processed {processed}/{total_queries} queries...")

    if processed == 0:
        logger.warning("No valid queries processed")
        return 0.0, []

    avg_recall = total_recall / processed
    logger.info(f"\n{'='*50}")
    logger.info(f"Recall@{k} Results:")
    logger.info(f"  Queries Processed: {processed}")
    logger.info(f"  Average Recall: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    logger.info(f"{'='*50}")

    # Save results to Excel if output path specified
    if output_path:
        save_recall_results_to_excel(
            results_list=results_list,
            output_path=output_path,
            k=k,
            score_threshold=score_threshold,
            total=processed,
            avg_recall=avg_recall,
        )

    return avg_recall, results_list


def save_recall_results_to_excel(
    results_list: List[Dict[str, Any]],
    output_path: Path,
    k: int,
    score_threshold: float,
    total: int,
    avg_recall: float,
) -> None:
    """
    Save Recall@K evaluation results to Excel file with two sheets.

    Args:
        results_list: List of per-query result dictionaries
        output_path: Path to output Excel file
        k: K value used in evaluation
        score_threshold: Score threshold used
        total: Total queries processed
        avg_recall: Calculated average recall
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrames
    results_df = pd.DataFrame(results_list)

    # Calculate additional statistics
    perfect_recall_count = sum(1 for r in results_list if r["query_recall"] == 1.0)
    zero_recall_count = sum(1 for r in results_list if r["query_recall"] == 0.0)

    summary_data = {
        "Metric": [
            "Evaluation Date",
            "K Value",
            "Score Threshold",
            "Total Queries",
            "Average Recall",
            "Average Recall (%)",
            "Perfect Recall Queries",
            "Zero Recall Queries",
        ],
        "Value": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            k,
            score_threshold,
            total,
            f"{avg_recall:.4f}",
            f"{avg_recall * 100:.2f}%",
            perfect_recall_count,
            zero_recall_count,
        ],
    }
    summary_df = pd.DataFrame(summary_data)

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        results_df.to_excel(writer, sheet_name="Per-Query Results", index=False)

    logger.info(f"Results saved to: {output_path}")


def save_results_to_excel(
    results_list: List[Dict[str, Any]],
    output_path: Path,
    k: int,
    score_threshold: float,
    hits: int,
    total: int,
    hit_rate: float,
) -> None:
    """
    Save evaluation results to Excel file with two sheets.

    Args:
        results_list: List of per-query result dictionaries
        output_path: Path to output Excel file
        k: K value used in evaluation
        score_threshold: Score threshold used
        hits: Number of hits
        total: Total queries processed
        hit_rate: Calculated hit rate
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrames
    results_df = pd.DataFrame(results_list)

    summary_data = {
        "Metric": [
            "Evaluation Date",
            "K Value",
            "Score Threshold",
            "Total Queries",
            "Hits",
            "Misses",
            "Hit Rate",
            "Hit Rate (%)",
        ],
        "Value": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            k,
            score_threshold,
            total,
            hits,
            total - hits,
            f"{hit_rate:.4f}",
            f"{hit_rate * 100:.2f}%",
        ],
    }
    summary_df = pd.DataFrame(summary_data)

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        results_df.to_excel(writer, sheet_name="Per-Query Results", index=False)

    logger.info(f"Results saved to: {output_path}")


def evaluate_all(
    test_data_path: Path = DEFAULT_TEST_DATA_PATH,
    k: int = 5,
    score_threshold: float = 0.0,
    api_base_url: str = API_BASE_URL,
    limit: Optional[int] = None,
    verbose: bool = False,
    output_path: Optional[Path] = None,
) -> Tuple[float, float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Calculate both Hit@K and Recall@K metrics in a single execution.

    Optimizes API calls by making a single request per query and computing
    both metrics from the same retrieved results.

    Args:
        test_data_path: Path to Excel file with test data
        k: Number of top results to consider
        score_threshold: Minimum similarity score threshold
        api_base_url: Base URL for the RAG API
        limit: Limit number of queries to evaluate (None = all)
        verbose: Print detailed per-query results
        output_path: Path to save results Excel file (None = no save)

    Returns:
        Tuple of (hit_rate, avg_recall, hit_results_list, recall_results_list)
    """
    # Load test data
    logger.info(f"Loading test data from: {test_data_path}")
    df = pd.read_excel(test_data_path)

    if limit:
        df = df.head(limit)
        logger.info(f"Limited to first {limit} queries")

    total_queries = len(df)
    logger.info(f"Total queries to evaluate: {total_queries}")

    # Counters for metrics
    hits = 0
    total_recall = 0.0
    processed = 0

    # Result lists for each metric
    hit_results_list: List[Dict[str, Any]] = []
    recall_results_list: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        query = row.get("Query", "")
        if not query or (isinstance(query, float) and pd.isna(query)):
            logger.warning(f"Skipping row {idx}: empty query")
            continue

        # Parse ground truth point IDs
        ground_truth_ids = parse_point_ids(row.get("Point_ids"))
        if not ground_truth_ids:
            logger.warning(f"Skipping row {idx}: no ground truth Point_ids")
            continue

        # Single API call per query
        retrieved_ids = search_rag_api(
            query=query,
            top_k=k,
            score_threshold=score_threshold,
            api_base_url=api_base_url,
        )

        # Calculate Hit@K (binary)
        is_hit = any(gt_id in retrieved_ids for gt_id in ground_truth_ids)
        if is_hit:
            hits += 1

        # Calculate Recall@K (fraction)
        hits_count = sum(1 for gt_id in ground_truth_ids if gt_id in retrieved_ids)
        query_recall = hits_count / len(ground_truth_ids)
        total_recall += query_recall

        processed += 1

        # Common metadata
        difficulty = row.get("difficulty", "")
        source_files = row.get("Source_files", "")

        # Store Hit@K result
        hit_results_list.append({
            "query_index": idx,
            "query": query,
            "ground_truth_ids": str(ground_truth_ids),
            "retrieved_ids": str(retrieved_ids),
            "is_hit": is_hit,
            "difficulty": difficulty,
            "source_files": source_files,
        })

        # Store Recall@K result
        recall_results_list.append({
            "query_index": idx,
            "query": query,
            "ground_truth_ids": str(ground_truth_ids),
            "retrieved_ids": str(retrieved_ids),
            "ground_truth_count": len(ground_truth_ids),
            "hits_count": hits_count,
            "query_recall": query_recall,
            "difficulty": difficulty,
            "source_files": source_files,
        })

        if verbose:
            hit_status = "HIT" if is_hit else "MISS"
            logger.info(
                f"[{processed}/{total_queries}] {hit_status} | Recall: {query_recall:.4f} | "
                f"Query: '{query[:40]}...' | "
                f"Hits: {hits_count}/{len(ground_truth_ids)}"
            )
        elif processed % 10 == 0:
            logger.info(f"Processed {processed}/{total_queries} queries...")

    if processed == 0:
        logger.warning("No valid queries processed")
        return 0.0, 0.0, [], []

    hit_rate = hits / processed
    avg_recall = total_recall / processed

    logger.info(f"\n{'='*50}")
    logger.info(f"Combined Evaluation Results (k={k}):")
    logger.info(f"  Hit@{k}:    {hit_rate:.4f} ({hit_rate*100:.2f}%) - {hits}/{processed} queries")
    logger.info(f"  Recall@{k}: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    logger.info(f"{'='*50}")

    # Save results to Excel if output path specified
    if output_path:
        save_combined_results_to_excel(
            hit_results_list=hit_results_list,
            recall_results_list=recall_results_list,
            output_path=output_path,
            k=k,
            score_threshold=score_threshold,
            hits=hits,
            total=processed,
            hit_rate=hit_rate,
            avg_recall=avg_recall,
        )

    return hit_rate, avg_recall, hit_results_list, recall_results_list


def save_combined_results_to_excel(
    hit_results_list: List[Dict[str, Any]],
    recall_results_list: List[Dict[str, Any]],
    output_path: Path,
    k: int,
    score_threshold: float,
    hits: int,
    total: int,
    hit_rate: float,
    avg_recall: float,
) -> None:
    """
    Save combined Hit@K and Recall@K evaluation results to Excel file.

    Creates three sheets:
    - Summary: Combined summary with both metrics
    - Per-Query Results (Hit): Hit@K per-query results
    - Per-Query Results (Recall): Recall@K per-query results

    Args:
        hit_results_list: List of Hit@K per-query result dictionaries
        recall_results_list: List of Recall@K per-query result dictionaries
        output_path: Path to output Excel file
        k: K value used in evaluation
        score_threshold: Score threshold used
        hits: Number of hits
        total: Total queries processed
        hit_rate: Calculated hit rate
        avg_recall: Calculated average recall
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrames
    hit_results_df = pd.DataFrame(hit_results_list)
    recall_results_df = pd.DataFrame(recall_results_list)

    # Calculate additional statistics
    perfect_recall_count = sum(1 for r in recall_results_list if r["query_recall"] == 1.0)
    zero_recall_count = sum(1 for r in recall_results_list if r["query_recall"] == 0.0)

    summary_data = {
        "Metric": [
            "Evaluation Date",
            "K Value",
            "Score Threshold",
            "Total Queries",
            "",
            "--- Hit@K Metrics ---",
            "Hits",
            "Misses",
            "Hit Rate",
            "Hit Rate (%)",
            "",
            "--- Recall@K Metrics ---",
            "Average Recall",
            "Average Recall (%)",
            "Perfect Recall Queries",
            "Zero Recall Queries",
        ],
        "Value": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            k,
            score_threshold,
            total,
            "",
            "",
            hits,
            total - hits,
            f"{hit_rate:.4f}",
            f"{hit_rate * 100:.2f}%",
            "",
            "",
            f"{avg_recall:.4f}",
            f"{avg_recall * 100:.2f}%",
            perfect_recall_count,
            zero_recall_count,
        ],
    }
    summary_df = pd.DataFrame(summary_data)

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        hit_results_df.to_excel(writer, sheet_name="Per-Query Results (Hit)", index=False)
        recall_results_df.to_excel(writer, sheet_name="Per-Query Results (Recall)", index=False)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate retrieval metrics (Hit@K or Recall@K) for RAG evaluation"
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["hit", "recall", "all"],
        default="hit",
        help="Metric to calculate: 'hit' for Hit@K, 'recall' for Recall@K, 'all' for both (default: hit)",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=5,
        help="Number of top results to consider (default: 5)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold (default: 0.0)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data Excel file (default: qr_smartphone_dataset.xlsx)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (default: all)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed per-query results",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output Excel file path for results (default: auto-generated in results/)",
    )

    args = parser.parse_args()

    test_data_path = Path(args.test_data) if args.test_data else DEFAULT_TEST_DATA_PATH

    # Generate output path if not specified
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
    elif args.metric == "recall":
        output_path = DEFAULT_OUTPUT_DIR / f"recall_k{args.k}_results_{timestamp}.xlsx"
    elif args.metric == "all":
        output_path = DEFAULT_OUTPUT_DIR / f"combined_k{args.k}_results_{timestamp}.xlsx"
    else:
        output_path = DEFAULT_OUTPUT_DIR / f"hit_k{args.k}_results_{timestamp}.xlsx"

    if args.metric == "all":
        hit_rate, avg_recall, _, _ = evaluate_all(
            test_data_path=test_data_path,
            k=args.k,
            score_threshold=args.threshold,
            api_base_url=args.api_url,
            limit=args.limit,
            verbose=args.verbose,
            output_path=output_path,
        )
        print(f"\nFinal Hit@{args.k}: {hit_rate:.4f}")
        print(f"Final Recall@{args.k}: {avg_recall:.4f}")
    elif args.metric == "recall":
        score, _ = recall_k(
            test_data_path=test_data_path,
            k=args.k,
            score_threshold=args.threshold,
            api_base_url=args.api_url,
            limit=args.limit,
            verbose=args.verbose,
            output_path=output_path,
        )
        print(f"\nFinal Recall@{args.k}: {score:.4f}")
    else:
        score, _ = hit_k(
            test_data_path=test_data_path,
            k=args.k,
            score_threshold=args.threshold,
            api_base_url=args.api_url,
            limit=args.limit,
            verbose=args.verbose,
            output_path=output_path,
        )
        print(f"\nFinal Hit@{args.k}: {score:.4f}")

    print(f"Results saved to: {output_path}")
