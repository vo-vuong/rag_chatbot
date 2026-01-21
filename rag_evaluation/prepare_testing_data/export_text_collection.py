"""
Export all data from TEXT_COLLECTION_NAME to JSON format.

Retrieves all points from the Qdrant text collection and exports them
to a JSON file with the following structure:
[
    {
        "Point_id": 1,
        "document_chunk_index": 0,
        "chunk": "chunk text",
        "Source_sections": ["Heading 1", "Heading 2"],
        "Source_files": "filename.pdf"
    }
]

Usage:
    conda activate rag_chatbot && python rag_evaluation/prepare_testing_data/export_text_collection.py --remove-ellipsis
    conda activate rag_chatbot && python rag_evaluation/prepare_testing_data/export_text_collection.py --limit-docs 10
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.vector_db.qdrant_manager import QdrantManager
from config.constants import QDRANT_HOST, QDRANT_PORT, TEXT_COLLECTION_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clean_ellipsis_patterns(text: str) -> str:
    """
    Remove strings between [...] and [...] from text.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text with ellipsis patterns removed
    """
    # Pattern to match [...] <any content> [...]
    # This uses non-greedy matching to find the shortest match
    pattern = r'\[\.\.\.\][^[].*?\[\.\.\.\]'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned


def get_all_points(
    manager: QdrantManager,
    collection_name: str,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retrieve all points from a collection using scroll pagination.

    Args:
        manager: QdrantManager instance
        collection_name: Name of the collection to retrieve from
        batch_size: Number of points to retrieve per batch

    Returns:
        List of all points with id and payload
    """
    all_points = []
    next_offset = None

    logger.info(f"Retrieving all points from collection: {collection_name}")

    while True:
        result = manager.get_collection_points(
            collection_name=collection_name,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            start_from_offset=next_offset,
        )

        if "error" in result:
            logger.error(f"Error retrieving points: {result['error']}")
            break

        points = result.get("points", [])
        all_points.extend(points)

        logger.info(f"Retrieved {len(all_points)} points so far...")

        if not result.get("has_more", False):
            break

        next_offset = result.get("next_page_offset")

    logger.info(f"Total points retrieved: {len(all_points)}")
    return all_points


def transform_point(
    point: Dict[str, Any], remove_ellipsis: bool = False
) -> Dict[str, Any]:
    """
    Transform a Qdrant point to the target JSON format.

    Args:
        point: Raw point data with id and payload
        remove_ellipsis: If True, remove strings between [...] and [...] from chunk

    Returns:
        Transformed dict with Point_id, document_chunk_index, chunk, Source_sections, Source_files
    """
    payload = point.get("payload", {})

    # Extract chunk content (handles both 'chunk' and 'content' fields)
    chunk = payload.get("chunk", payload.get("content", ""))

    # Remove ellipsis patterns if requested
    if remove_ellipsis:
        chunk = clean_ellipsis_patterns(chunk)

    # Extract headings/sections from metadata
    # Headings may be stored directly in payload or nested in metadata
    source_sections = payload.get("headings", [])

    # Ensure source_sections is a list
    if source_sections is None:
        source_sections = []
    elif isinstance(source_sections, str):
        source_sections = [source_sections] if source_sections else []

    # Extract source file
    source_file = payload.get("source_file", "Unknown")

    # Extract chunk document index
    document_chunk_index = payload.get("document_chunk_index")

    return {
        "Point_id": point.get("id", ""),
        "document_chunk_index": document_chunk_index,
        "chunk": chunk,
        "Source_sections": source_sections,
        "Source_files": source_file,
    }


def export_to_json(
    output_path: Path,
    data: List[Dict[str, Any]],
    indent: int = 2,
) -> None:
    """
    Export data to JSON file.

    Args:
        output_path: Path to output JSON file
        data: List of transformed point data
        indent: JSON indentation level
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    logger.info(f"Exported {len(data)} records to {output_path}")


def filter_first_n_documents(
    points: List[Dict[str, Any]], limit_docs: int
) -> List[Dict[str, Any]]:
    """
    Filter points to include only those from the first N unique documents.

    Points are sorted by id (ascending), and unique Source_files are collected
    in order of first encounter. Only points belonging to the first N documents
    are returned.

    Args:
        points: List of all points with id and payload
        limit_docs: Number of unique documents to include

    Returns:
        Filtered list of points belonging to the first N documents
    """
    # Sort points by id ascending
    sorted_points = sorted(points, key=lambda p: p.get("id", 0))

    # Collect unique Source_files in order of encounter
    seen_files: set = set()
    ordered_files: List[str] = []

    for point in sorted_points:
        source_file = point.get("payload", {}).get("source_file", "Unknown")
        if source_file not in seen_files:
            seen_files.add(source_file)
            ordered_files.append(source_file)

    # Get the first N documents
    first_n_docs = set(ordered_files[:limit_docs])
    logger.info(f"Limiting to first {limit_docs} documents: {sorted(first_n_docs)}")

    # Filter points to only include those from the first N documents
    filtered_points = [
        p
        for p in sorted_points
        if p.get("payload", {}).get("source_file", "Unknown") in first_n_docs
    ]

    logger.info(
        f"Filtered from {len(points)} to {len(filtered_points)} points "
        f"({len(first_n_docs)} documents)"
    )

    return filtered_points


def main(
    output_file: Optional[str] = None,
    collection_name: Optional[str] = None,
    batch_size: int = 100,
    remove_ellipsis: bool = False,
    limit_docs: Optional[int] = None,
) -> None:
    """
    Main function to export TEXT_COLLECTION_NAME data to JSON.

    Args:
        output_file: Output JSON file path (default: text_collection_data.json)
        collection_name: Collection name override (default: TEXT_COLLECTION_NAME)
        batch_size: Batch size for retrieval
        remove_ellipsis: If True, remove strings between [...] and [...] from chunk field
        limit_docs: If set, limit export to first N unique documents (by Point_id order)
    """
    # Set defaults
    collection = collection_name or TEXT_COLLECTION_NAME
    output_path = (
        Path(output_file)
        if output_file
        else Path(__file__).parent / "text_collection_data.json"
    )

    logger.info(f"Starting export from collection: {collection}")
    logger.info(f"Qdrant server: {QDRANT_HOST}:{QDRANT_PORT}")
    if remove_ellipsis:
        logger.info("Ellipsis patterns ([...] ... [...]) will be removed from chunks")
    if limit_docs:
        logger.info(f"Will limit export to first {limit_docs} documents (by Point_id order)")

    # Initialize Qdrant manager
    try:
        manager = QdrantManager(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=collection,
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        sys.exit(1)

    # Check if collection exists
    if not manager.collection_exists():
        logger.error(f"Collection '{collection}' does not exist")
        sys.exit(1)

    # Get collection info
    info = manager.get_collection_info()
    if info:
        logger.info(f"Collection has {info.get('points_count', 0)} points")

    # Retrieve all points
    points = get_all_points(manager, collection, batch_size)

    if not points:
        logger.warning("No points found in collection")
        return

    # Filter to first N documents if limit_docs is set
    if limit_docs:
        points = filter_first_n_documents(points, limit_docs)
        if not points:
            logger.warning("No points remaining after filtering")
            return

    # Transform to target format
    transformed_data = [
        transform_point(p, remove_ellipsis=remove_ellipsis) for p in points
    ]

    # Export to JSON
    export_to_json(output_path, transformed_data)

    logger.info("Export completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TEXT_COLLECTION_NAME data to JSON"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: text_collection_data.json)",
    )
    parser.add_argument(
        "-c",
        "--collection",
        type=str,
        default=None,
        help=f"Collection name (default: {TEXT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for retrieval (default: 100)",
    )
    parser.add_argument(
        "--remove-ellipsis",
        action="store_true",
        help="Remove strings between [...] and [...] from chunk field",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Limit export to first N unique documents (by Point_id order)",
    )

    args = parser.parse_args()

    main(
        output_file=args.output,
        collection_name=args.collection,
        batch_size=args.batch_size,
        remove_ellipsis=args.remove_ellipsis,
        limit_docs=args.limit_docs,
    )
