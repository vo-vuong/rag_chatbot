"""
CSV chunking with column-based grouping capabilities.

This module provides flexible chunking strategies for CSV data, supporting
column-based grouping and configurable chunk parameters.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class CSVGroupingChunker:
    """
    Advanced CSV chunking with column-based grouping capabilities.

    Enhanced version with large file support, memory optimization, and
    intelligent chunk splitting strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CSV chunker with configuration.

        Args:
            config: Configuration dictionary for chunking parameters
        """
        self.config = config or {}

        # Enhanced default configuration
        self.default_config = {
            "max_chunk_size": 2000,  # characters
            "max_rows_per_chunk": 10,
            "include_headers": True,
            "delimiter": " | ",
            "null_placeholder": "N/A",
            "chunk_overlap": 0.1,  # 10% overlap
            "min_chunk_size": 100,  # minimum characters
            "large_group_threshold": 50,  # rows
            "memory_efficient": True,
            "preserve_order": True,
            "group_separator": "\n---\n",
        }

        # Merge with provided config
        self.chunking_config = {**self.default_config, **self.config}

    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        group_columns: Optional[List[str]] = None,
        max_rows_per_chunk: Optional[int] = None,
    ) -> List[Dict]:
        """
        Chunk DataFrame using specified strategy with enhanced logic.

        Args:
            df: DataFrame to chunk
            group_columns: Columns to group by (None for row-by-row)
            max_rows_per_chunk: Maximum rows per chunk

        Returns:
            List of chunk dictionaries with enhanced metadata
        """
        if max_rows_per_chunk is None:
            max_rows_per_chunk = self.chunking_config["max_rows_per_chunk"]

        # At this point, max_rows_per_chunk is guaranteed to be an int
        assert max_rows_per_chunk is not None

        # Memory optimization for large DataFrames
        if self.chunking_config["memory_efficient"] and len(df) > 10000:
            df = self._optimize_dataframe_memory(df)

        if not group_columns:
            # Row-by-row chunking with optional batching
            return self._chunk_by_rows_enhanced(df)

        # Column-based grouping with advanced splitting
        return self._chunk_by_columns_enhanced(df, group_columns, max_rows_per_chunk)

    def _chunk_by_rows(self, df: pd.DataFrame) -> List[Dict]:
        """
        Chunk DataFrame row by row (current behavior).

        Args:
            df: DataFrame to chunk

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        for idx, row in df.iterrows():
            chunk_text = self._format_row_as_text(row)

            chunk = {
                "chunk": chunk_text,
                "row_index": idx,
                "row_count": 1,
                "doc_id": str(uuid.uuid4()),
                "metadata": {
                    "chunking_strategy": "row_by_row",
                    "source_rows": [idx],
                    "columns": list(df.columns),
                    "group_key": f"row_{idx}",
                },
            }
            chunks.append(chunk)

        return chunks

    def _chunk_by_columns(
        self, df: pd.DataFrame, group_columns: List[str], max_rows_per_chunk: int
    ) -> List[Dict]:
        """
        Chunk DataFrame by grouping columns.

        Args:
            df: DataFrame to chunk
            group_columns: Columns to group by
            max_rows_per_chunk: Maximum rows per chunk

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        try:
            # Validate group columns exist
            missing_cols = [col for col in group_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Group columns not found in DataFrame: {missing_cols}")
                return self._chunk_by_rows(df)

            # Group by selected columns
            grouped = df.groupby(group_columns, observed=True, dropna=False)

            for group_key, group_df in grouped:
                # Handle large groups by splitting them
                if len(group_df) <= max_rows_per_chunk:
                    # Create single chunk for group
                    chunk = self._create_group_chunk(group_df, group_columns, group_key)
                    chunks.append(chunk)
                else:
                    # Split large group into smaller chunks
                    sub_chunks = self._split_large_group(
                        group_df, group_columns, group_key, max_rows_per_chunk
                    )
                    chunks.extend(sub_chunks)

        except Exception as e:
            logger.error(f"Column grouping failed: {e}")
            # Fallback to row-by-row if grouping fails
            return self._chunk_by_rows(df)

        return chunks

    def _create_group_chunk(
        self, group_df: pd.DataFrame, group_columns: List[str], group_key: Any
    ) -> Dict:
        """
        Create a chunk from grouped DataFrame.

        Args:
            group_df: Grouped DataFrame
            group_columns: Columns used for grouping
            group_key: The group key value

        Returns:
            Chunk dictionary
        """
        chunk_text = self._format_group_as_text(group_df, group_columns, group_key)

        return {
            "chunk": chunk_text,
            "group_key": str(group_key) if group_key is not None else "default",
            "row_count": len(group_df),
            "doc_id": str(uuid.uuid4()),
            "metadata": {
                "chunking_strategy": "column_grouping",
                "group_columns": group_columns,
                "group_value": str(group_key) if group_key is not None else None,
                "source_rows": group_df.index.tolist(),
                "columns": list(group_df.columns),
                "column_count": len(group_df.columns),
            },
        }

    def _format_row_as_text(self, row: pd.Series) -> str:
        """
        Format a single row as text.

        Args:
            row: Pandas Series representing a row

        Returns:
            Formatted text string
        """
        include_headers = self.chunking_config["include_headers"]
        delimiter = self.chunking_config["delimiter"]
        null_placeholder = self.chunking_config["null_placeholder"]

        parts = []
        for col, val in row.items():
            if pd.notna(val):
                if include_headers:
                    parts.append(f"{col}: {val}")
                else:
                    parts.append(str(val))
            elif not include_headers:
                parts.append(null_placeholder)

        return delimiter.join(parts)

    def _format_group_as_text(
        self, group_df: pd.DataFrame, group_columns: List[str], group_key: Any
    ) -> str:
        """
        Format a group of rows as text.

        Args:
            group_df: Grouped DataFrame
            group_columns: Columns used for grouping
            group_key: The group key value

        Returns:
            Formatted text string
        """
        include_headers = self.chunking_config["include_headers"]
        delimiter = self.chunking_config["delimiter"]
        null_placeholder = self.chunking_config["null_placeholder"]

        lines = []

        # Add group header if grouping columns specified
        if group_columns:
            if len(group_columns) == 1:
                lines.append(f"{group_columns[0]}: {group_key}")
            else:
                # Format multi-column group key
                if isinstance(group_key, tuple):
                    key_parts = [
                        f"{col}: {val}" for col, val in zip(group_columns, group_key)
                    ]
                    lines.append(" | ".join(key_parts))
                else:
                    lines.append(f"Group: {group_key}")
            lines.append("")  # Empty line for readability

        # Add column headers if requested
        if include_headers:
            headers = delimiter.join(group_df.columns.tolist())
            lines.append(f"Columns: {headers}")
            lines.append("")  # Empty line before data

        # Add each row
        for _, row in group_df.iterrows():
            row_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    if include_headers:
                        row_parts.append(f"{col}: {val}")
                    else:
                        row_parts.append(str(val))
                else:
                    row_parts.append(null_placeholder)

            lines.append(delimiter.join(row_parts))

        return "\n".join(lines)

    def estimate_chunk_size(self, text: str) -> int:
        """
        Estimate the size of a chunk in characters.

        Args:
            text: Text to measure

        Returns:
            Estimated character count
        """
        return len(text)

    def validate_chunk_size(self, chunk_text: str) -> bool:
        """
        Validate if a chunk is within size limits.

        Args:
            chunk_text: Text of the chunk

        Returns:
            True if chunk size is acceptable
        """
        chunk_size = self.estimate_chunk_size(chunk_text)
        max_size = self.chunking_config.get("max_chunk_size", 4000)
        return chunk_size <= max_size

    def get_chunking_stats(self, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the chunking process.
        """
        if not chunks:
            return {"total_chunks": 0, "total_characters": 0}

        chunk_sizes = [len(chunk.get("chunk", "")) for chunk in chunks]
        row_counts = [chunk.get("row_count", 0) for chunk in chunks]

        strategies = {}
        for chunk in chunks:
            strategy = chunk.get("metadata", {}).get("chunking_strategy", "unknown")
            strategies[strategy] = strategies.get(strategy, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_rows": sum(row_counts),
            "avg_rows_per_chunk": sum(row_counts) / len(row_counts),
            "chunking_strategies": strategies,
            "memory_efficient": self.chunking_config["memory_efficient"],
            "overlap_enabled": self.chunking_config["chunk_overlap"] > 0,
        }

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage for large files.

        Uses appropriate data types and reduces memory footprint.
        """
        original_memory = df.memory_usage(deep=True).sum()

        # Convert object columns to categorical where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        # Downcast numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        # Convert datetime columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Cast to Any to avoid type checking issues with pd.to_datetime
                df[col] = pd.to_datetime(df[col], errors='ignore')  # type: ignore
            except Exception:
                pass

        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100

        logger.info(f"DataFrame memory optimized: {reduction:.1f}% reduction")

        return df

    def _chunk_by_rows_enhanced(self, df: pd.DataFrame) -> List[Dict]:
        """
        Enhanced row-by-row chunking with optional batching.

        For large DataFrames, can batch rows to reduce chunk count while
        maintaining reasonable chunk sizes.
        """
        chunks = []
        max_rows_per_chunk = self.chunking_config["max_rows_per_chunk"]
        chunk_overlap = self.chunking_config["chunk_overlap"]

        # Determine optimal batch size based on content
        avg_row_length = self._estimate_average_row_length(df)
        optimal_rows_per_chunk = min(
            max_rows_per_chunk,
            max(1, self.chunking_config["max_chunk_size"] // avg_row_length),
        )

        # Create chunks with overlap
        for i in range(0, len(df), optimal_rows_per_chunk):
            # Calculate overlap
            overlap_rows = int(optimal_rows_per_chunk * chunk_overlap)
            start_idx = max(0, i - overlap_rows)
            end_idx = min(len(df), i + optimal_rows_per_chunk + overlap_rows)

            chunk_df = df.iloc[start_idx:end_idx]

            if not chunk_df.empty:
                chunk_text = self._format_batch_as_text(chunk_df, start_idx)

                chunk = {
                    "chunk": chunk_text,
                    "row_index": start_idx,
                    "row_count": len(chunk_df),
                    "doc_id": str(uuid.uuid4()),
                    "metadata": {
                        "chunking_strategy": "row_by_row_batched",
                        "source_rows": chunk_df.index.tolist(),
                        "columns": list(df.columns),
                        "batch_size": optimal_rows_per_chunk,
                        "overlap_rows": overlap_rows if start_idx > 0 else 0,
                        "dataframe_memory_mb": df.memory_usage(deep=True).sum()
                        / 1024
                        / 1024,
                    },
                }
                chunks.append(chunk)

        return chunks

    def _chunk_by_columns_enhanced(
        self, df: pd.DataFrame, group_columns: List[str], max_rows_per_chunk: int
    ) -> List[Dict]:
        """
        Enhanced column-based grouping with intelligent splitting.

        Handles large groups, creates overlapping chunks, and preserves
        context for better retrieval performance.
        """
        chunks = []

        try:
            # Group by selected columns
            grouped = df.groupby(group_columns, observed=True, dropna=False)

            for group_key, group_df in grouped:
                # Process each group
                if len(group_df) <= max_rows_per_chunk:
                    # Small group - single chunk
                    chunk = self._create_group_chunk(group_df, group_columns, group_key)
                    chunks.append(chunk)
                else:
                    # Large group - split into multiple chunks with overlap
                    group_chunks = self._split_large_group(
                        group_df, group_columns, group_key, max_rows_per_chunk
                    )
                    chunks.extend(group_chunks)

        except Exception as e:
            logger.error(f"Enhanced column grouping failed: {e}")
            # Fallback to enhanced row-by-row chunking
            return self._chunk_by_rows_enhanced(df)

        return chunks

    def _split_large_group(
        self,
        group_df: pd.DataFrame,
        group_columns: List[str],
        group_key: Any,
        max_rows_per_chunk: int,
    ) -> List[Dict]:
        """
        Split a large group into multiple chunks with intelligent overlap.

        Creates overlapping chunks to maintain context across boundaries
        and improve retrieval performance.
        """
        chunks = []
        chunk_overlap = self.chunking_config["chunk_overlap"]
        overlap_rows = max(1, int(max_rows_per_chunk * chunk_overlap))

        # Sort by index to maintain order if required
        if self.chunking_config["preserve_order"]:
            group_df = group_df.sort_index()

        # Create overlapping chunks
        for i in range(0, len(group_df), max_rows_per_chunk):
            start_idx = i
            end_idx = min(len(group_df), i + max_rows_per_chunk + overlap_rows)

            chunk_df = group_df.iloc[start_idx:end_idx]

            if not chunk_df.empty:
                chunk_text = self._format_group_chunk_text(
                    chunk_df, group_columns, group_key, start_idx, len(group_df)
                )

                chunk = {
                    "chunk": chunk_text,
                    "group_key": str(group_key) if group_key else "default",
                    "row_index": start_idx,
                    "row_count": len(chunk_df),
                    "doc_id": str(uuid.uuid4()),
                    "metadata": {
                        "chunking_strategy": "column_grouping_split",
                        "group_columns": group_columns,
                        "group_value": str(group_key) if group_key else None,
                        "source_rows": chunk_df.index.tolist(),
                        "columns": list(group_df.columns),
                        "chunk_part": i // max_rows_per_chunk + 1,
                        "total_chunks": (len(group_df) + max_rows_per_chunk - 1)
                        // max_rows_per_chunk,
                        "overlap_rows": overlap_rows if end_idx < len(group_df) else 0,
                        "group_total_rows": len(group_df),
                    },
                }
                chunks.append(chunk)

        return chunks

    def _format_group_chunk_text(
        self,
        chunk_df: pd.DataFrame,
        group_columns: List[str],
        group_key: Any,
        chunk_start: Optional[int] = None,
        group_total_rows: Optional[int] = None,
    ) -> str:
        """
        Format group chunk as structured text with enhanced metadata.

        Includes context information about grouping and chunk position.
        """
        include_headers = self.chunking_config["include_headers"]
        delimiter = self.chunking_config["delimiter"]
        null_placeholder = self.chunking_config["null_placeholder"]

        lines = []

        # Enhanced group header with context
        if group_columns:
            if len(group_columns) == 1:
                lines.append(f"{group_columns[0]}: {group_key}")
            else:
                lines.append(f"Group: {group_key}")

            # Add chunk context if this is a split group
            if chunk_start is not None and group_total_rows is not None:
                chunk_end = chunk_start + len(chunk_df) - 1
                lines.append(f"Rows {chunk_start}-{chunk_end} of {group_total_rows}")

            lines.append("")  # Empty line for readability

        # Enhanced column information
        if include_headers:
            lines.append(f"Columns: {delimiter.join(chunk_df.columns.tolist())}")

            # Add data type information
            type_info = delimiter.join(
                [f"{col} ({dtype})" for col, dtype in chunk_df.dtypes.items()]
            )
            lines.append(f"Types: {type_info}")
            lines.append("")  # Empty line before data

        # Add each row with row numbers for reference
        for idx, (_, row) in enumerate(chunk_df.iterrows()):
            if chunk_start is not None:
                row_prefix = f"Row {chunk_start + idx}: "
            else:
                row_prefix = f"Row {idx}: "

            row_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    if include_headers:
                        row_parts.append(f"{col}: {val}")
                    else:
                        row_parts.append(str(val))
                else:
                    row_parts.append(
                        f"{col}: {null_placeholder}"
                        if include_headers
                        else null_placeholder
                    )

            lines.append(row_prefix + delimiter.join(row_parts))

        return "\n".join(lines)

    def _format_batch_as_text(self, batch_df: pd.DataFrame, start_idx: int) -> str:
        """Format a batch of rows as text with row numbering."""
        include_headers = self.chunking_config["include_headers"]
        delimiter = self.chunking_config["delimiter"]
        null_placeholder = self.chunking_config["null_placeholder"]

        lines = []

        # Add batch information
        lines.append(f"Data Batch: Rows {start_idx} to {start_idx + len(batch_df) - 1}")
        lines.append("")

        if include_headers:
            lines.append(f"Columns: {delimiter.join(batch_df.columns.tolist())}")
            lines.append("")

        # Add rows with numbering
        for i, (_, row) in enumerate(batch_df.iterrows()):
            row_prefix = f"Row {start_idx + i}: "
            row_parts = []

            for col, val in row.items():
                if pd.notna(val):
                    if include_headers:
                        row_parts.append(f"{col}: {val}")
                    else:
                        row_parts.append(str(val))
                else:
                    row_parts.append(null_placeholder)

            lines.append(row_prefix + delimiter.join(row_parts))

        return "\n".join(lines)

    def _estimate_average_row_length(self, df: pd.DataFrame) -> int:
        """Estimate average character length of a row in the DataFrame."""
        if df.empty:
            return 100

        # Sample a few rows to estimate length
        sample_size = min(100, len(df))
        sample_df = df.head(sample_size)

        total_length = 0
        include_headers = self.chunking_config["include_headers"]
        delimiter = self.chunking_config["delimiter"]

        for _, row in sample_df.iterrows():
            row_text = delimiter.join(
                [
                    f"{col}: {val}" if include_headers else str(val)
                    for col, val in row.items()
                    if pd.notna(val)
                ]
            )
            total_length += len(row_text)

        return max(50, total_length // sample_size)  # Minimum 50 chars
