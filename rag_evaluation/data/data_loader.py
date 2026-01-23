"""
Test data loading utilities.

Provides standardized loading and validation of test data for RAG evaluation.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd

from rag_evaluation.data.point_id_parser import parse_point_ids

logger = logging.getLogger(__name__)


class TestDataLoader:
    """
    Loader for RAG evaluation test data.

    Handles loading test data from Excel files and provides iteration
    over valid query-ground truth pairs.

    Attributes:
        data_path: Path to the test data file
        df: Loaded DataFrame
    """

    REQUIRED_COLUMNS = ["Query", "Point_ids"]

    def __init__(self, data_path: Path, limit: Optional[int] = None):
        """
        Initialize the test data loader.

        Args:
            data_path: Path to Excel file containing test data
            limit: Optional limit on number of queries to load

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing
        """
        self.data_path = Path(data_path)
        self._validate_file_exists()

        self.df = pd.read_excel(self.data_path)
        self._validate_columns()

        if limit:
            self.df = self.df.head(limit)
            logger.info(f"Limited to first {limit} queries")

        logger.info(f"Loaded {len(self.df)} queries from {self.data_path.name}")

    def _validate_file_exists(self) -> None:
        """Validate that the data file exists."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Test data file not found: {self.data_path}")

    def _validate_columns(self) -> None:
        """Validate that required columns exist in the data."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __len__(self) -> int:
        """Return total number of rows in the dataset."""
        return len(self.df)

    def iter_queries(self) -> Iterator[Tuple[int, str, List[int], dict]]:
        """
        Iterate over valid query-ground truth pairs.

        Yields:
            Tuple of (index, query_text, ground_truth_ids, row_metadata)
            Only yields rows with valid query and ground truth IDs.
        """
        for idx, row in self.df.iterrows():
            query = row.get("Query", "")

            # Skip empty queries
            if not query or (isinstance(query, float) and pd.isna(query)):
                logger.warning(f"Skipping row {idx}: empty query")
                continue

            # Parse ground truth IDs
            ground_truth_ids = parse_point_ids(row.get("Point_ids"))
            if not ground_truth_ids:
                logger.warning(f"Skipping row {idx}: no ground truth Point_ids")
                continue

            # Extract metadata
            metadata = {
                "ground_truth_answer": row.get("Ground_truth_answer", ""),
                "difficulty": row.get("difficulty", ""),
                "source_files": row.get("Source_files", ""),
            }

            yield idx, query, ground_truth_ids, metadata

    def get_valid_query_count(self) -> int:
        """Count queries with valid ground truth."""
        count = 0
        for _ in self.iter_queries():
            count += 1
        return count
