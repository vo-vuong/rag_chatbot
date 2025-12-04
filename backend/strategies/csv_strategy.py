"""
CSV processing strategy with column-based chunking.

This module provides comprehensive CSV processing capabilities with column-based grouping,
configurable chunking parameters, and integration with the existing document processor architecture.
"""

import gc
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.chunking import CSVGroupingChunker
from backend.errors import CSVEncodingError, CSVParsingError, CSVProcessingError
from backend.strategies import DocumentProcessingStrategy, ProcessingResult
from backend.utils import CSVOptimizer, CSVPerformanceMonitor

# Configure logging
logger = logging.getLogger(__name__)


class CSVProcessingStrategy(DocumentProcessingStrategy):
    """
    Enhanced CSV processing strategy with column-based chunking capabilities.

    This strategy provides comprehensive CSV processing capabilities including:
    - Column-based grouping for intelligent chunking
    - Streaming processing for large files
    - Memory optimization and monitoring
    - Enhanced error handling and recovery
    - Performance benchmarking and optimization
    - Configurable chunking parameters
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        chunker: Optional[CSVGroupingChunker] = None,
    ):
        """
        Initialize enhanced CSV processing strategy.

        Args:
            config: Configuration dictionary for processing parameters
            chunker: Optional CSV chunker instance
        """
        super().__init__(config)

        # Enhanced default configuration
        self.default_config = {
            "max_rows_per_chunk": 10,
            "include_headers": True,
            "null_value_handling": "skip",
            "encoding": "utf-8",
            "delimiter": ",",
            "max_file_size_mb": 50,
            "streaming_threshold_mb": 10,
            "memory_check_interval": 1000,
            "enable_progress_tracking": True,
            "memory_optimization": True,
        }

        # Merge with provided config
        self.processing_config = {**self.default_config, **self.config}

        # Initialize enhanced chunker
        self.chunker = chunker or CSVGroupingChunker(self.processing_config)

        # Set processing parameters from config
        self.max_rows_per_chunk = self.processing_config.get("max_rows_per_chunk", 10)
        self.include_headers = self.processing_config.get("include_headers", True)
        self.encoding = self.processing_config.get("encoding", "utf-8")
        self.delimiter = self.processing_config.get("delimiter", ",")
        self.null_value_handling = self.processing_config.get(
            "null_value_handling", "skip"
        )

        # Process tracking
        self.processing_stats = {
            "rows_processed": 0,
            "chunks_created": 0,
            "memory_peak_mb": 0.0,
            "processing_time_seconds": 0.0,
        }

        # Performance monitoring
        self.performance_monitor = None

    @property
    def supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return [".csv"]

    @property
    def strategy_name(self) -> str:
        """Get the name of this processing strategy."""
        return "CSV Processing Strategy"

    def can_process(self, file_path: str) -> bool:
        """
        Check if this strategy can process the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this strategy can process the file, False otherwise
        """
        return self.supports_file(file_path)

    def extract_elements(self, file_path: str, **kwargs) -> ProcessingResult:
        """
        Extract elements from CSV file with enhanced column-based chunking.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional processing parameters

        Returns:
            ProcessingResult containing extracted elements and metadata
        """
        start_time = time.time()

        # Initialize performance monitoring
        self.performance_monitor = CSVPerformanceMonitor()
        self.performance_monitor.start_monitoring()

        try:
            # Validate file and configuration
            file_path, validation_result = self._validate_file_and_config(file_path)
            if not validation_result["valid"]:
                return self._create_error_result(
                    validation_result["error"], error_type="ValidationError"
                )

            # Extract processing parameters
            selected_columns = kwargs.get("selected_columns", [])
            max_rows_per_chunk = kwargs.get(
                "max_rows_per_chunk", self.max_rows_per_chunk
            )
            include_headers = kwargs.get("include_headers", self.include_headers)

            # Determine processing strategy based on file size
            file_size_mb = self._get_file_size_mb(file_path)
            self.performance_monitor.add_checkpoint(
                "validation_complete",
                additional_info={
                    "file_size_mb": file_size_mb,
                    "selected_columns": selected_columns,
                },
            )

            if file_size_mb > self.processing_config["streaming_threshold_mb"]:
                logger.info(
                    f"Using streaming processing for large file ({file_size_mb:.1f}MB)"
                )
                chunks = self._process_csv_streaming(
                    file_path, selected_columns, max_rows_per_chunk, include_headers
                )
                processing_mode = "streaming"
            else:
                logger.info(
                    f"Using standard processing for file ({file_size_mb:.1f}MB)"
                )
                chunks = self._process_csv_standard(
                    file_path, selected_columns, max_rows_per_chunk, include_headers
                )
                processing_mode = "standard"

            # Calculate processing statistics
            self.processing_stats["processing_time_seconds"] = time.time() - start_time

            # Create processing elements in expected format
            elements = []
            for chunk in chunks:
                element = {
                    "text": chunk["chunk"],
                    "element_type": "csv_chunk",
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "source_file": Path(file_path).name,
                        "file_type": "CSV",
                        "doc_id": chunk.get("doc_id", str(uuid.uuid4())),
                        "processing_mode": processing_mode,
                    },
                }
                elements.append(element)

            # Create enhanced metadata
            metadata = self._extract_enhanced_metadata(
                file_path, selected_columns, file_size_mb
            )
            metadata.update(
                {
                    "selected_columns": selected_columns,
                    "max_rows_per_chunk": max_rows_per_chunk,
                    "include_headers": include_headers,
                    "chunking_strategy": (
                        "column_grouping" if selected_columns else "row_by_row"
                    ),
                    "processing_mode": processing_mode,
                    "performance_stats": self._calculate_enhanced_stats(
                        chunks, file_size_mb
                    ),
                }
            )

            # Final performance checkpoint
            self.performance_monitor.add_checkpoint(
                "processing_complete",
                len(chunks),
                {"elements_created": len(elements), "processing_mode": processing_mode},
            )

            return self._create_success_result(
                elements=elements,
                metadata=metadata,
                processing_time=time.time() - start_time,
            )

        except MemoryError:
            logger.error(f"Memory error processing {file_path}: file too large")
            return self._create_error_result(
                "CSV file too large for available memory. Try splitting into smaller files.",
                error_type="MemoryError",
            )

        except CSVProcessingError as e:
            logger.error(f"CSV processing error for {file_path}: {e}")
            return self._create_error_result(
                str(e),
                error_type=e.error_type,
                error_details={"suggestions": e.suggestions} if e.suggestions else None,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"CSV processing failed for {file_path}: {e}")
            return self._create_error_result(
                f"CSV processing failed: {str(e)}",
                error_type="ProcessingError",
                processing_time=processing_time,
            )

        finally:
            # Clean up memory
            gc.collect()
            if self.performance_monitor:
                logger.info("Performance monitoring completed")

    def _validate_file_and_config(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Validate file and processing configuration.

        Returns tuple of (file_path, validation_result)
        """
        validation_result = {"valid": True, "error": None}

        # Check if file exists
        if not Path(file_path).exists():
            validation_result["valid"] = False
            validation_result["error"] = f"File not found: {file_path}"
            return file_path, validation_result

        # Check file size
        file_size_mb = self._get_file_size_mb(file_path)
        max_size_mb = self.processing_config["max_file_size_mb"]

        if file_size_mb > max_size_mb:
            validation_result["valid"] = False
            validation_result["error"] = (
                f"CSV file too large: {file_size_mb:.1f}MB. "
                f"Maximum allowed: {max_size_mb}MB"
            )
            return file_path, validation_result

        # Validate configuration parameters
        max_rows = self.processing_config["max_rows_per_chunk"]
        if max_rows < 1 or max_rows > 1000:
            validation_result["valid"] = False
            validation_result["error"] = (
                f"Invalid max_rows_per_chunk: {max_rows}. "
                f"Must be between 1 and 1000"
            )

        return file_path, validation_result

    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in megabytes."""
        try:
            size_bytes = Path(file_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0

    def _process_csv_standard(
        self,
        file_path: str,
        selected_columns: List[str],
        max_rows_per_chunk: int,
        include_headers: bool,
    ) -> List[Dict]:
        """Standard CSV processing for smaller files."""
        try:
            # Read CSV file with enhanced options
            df = self._read_csv_file_enhanced(file_path)

            if df.empty:
                return []

            # Check memory usage during processing
            self._check_memory_usage("after_reading_csv")
            self.performance_monitor.add_checkpoint("csv_loaded", len(df))

            # Apply memory optimization if enabled
            if self.processing_config.get("memory_optimization", True):
                df = CSVOptimizer.optimize_dataframe_memory(df)
                self.performance_monitor.add_checkpoint("memory_optimized", len(df))

            # Process with chunking
            chunks = self.chunker.chunk_dataframe(
                df, selected_columns, max_rows_per_chunk
            )

            self.processing_stats["rows_processed"] = len(df)
            self.processing_stats["chunks_created"] = len(chunks)

            return chunks

        except Exception as e:
            logger.error(f"Standard CSV processing failed: {e}")
            raise

    def _process_csv_streaming(
        self,
        file_path: str,
        selected_columns: List[str],
        max_rows_per_chunk: int,
        include_headers: bool,
    ) -> List[Dict]:
        """
        Streaming CSV processing for large files.

        Processes the file in chunks to minimize memory usage.
        """
        chunks = []
        total_rows = 0

        try:
            # Get CSV structure first (read just header and first few rows)
            header_info = self._analyze_csv_structure(file_path)
            self.performance_monitor.add_checkpoint("structure_analyzed", 0)

            if not selected_columns:
                # For streaming row-by-row processing
                chunks = self._stream_row_by_row(file_path, header_info)
            else:
                # For streaming with column grouping, we need to be more creative
                chunks = self._stream_with_grouping(
                    file_path, selected_columns, max_rows_per_chunk, header_info
                )

            total_rows = sum(chunk.get("row_count", 0) for chunk in chunks)
            self.processing_stats["rows_processed"] = total_rows
            self.processing_stats["chunks_created"] = len(chunks)

            self.performance_monitor.add_checkpoint("streaming_complete", total_rows)

            return chunks

        except Exception as e:
            logger.error(f"Streaming CSV processing failed: {e}")
            raise

    def _analyze_csv_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV structure without reading entire file."""
        try:
            # Read just header and first few rows to understand structure
            sample_df = pd.read_csv(file_path, nrows=10)

            return {
                "columns": list(sample_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                "sample_rows": sample_df.head().to_dict('records'),
            }

        except Exception as e:
            logger.error(f"Failed to analyze CSV structure: {e}")
            raise CSVParsingError(f"Unable to read CSV structure: {str(e)}")

    def _stream_row_by_row(
        self, file_path: str, header_info: Dict[str, Any]
    ) -> List[Dict]:
        """Stream process CSV row by row for very large files."""
        chunks = []
        chunk_size = self.processing_config["max_rows_per_chunk"]
        rows_buffer = []

        try:
            # Use pandas read_csv with chunksize for streaming
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
                for _, row in chunk_df.iterrows():
                    # Add to buffer
                    rows_buffer.append(row)

                    # Create chunk when buffer is full
                    if len(rows_buffer) >= chunk_size:
                        chunk_text = self._format_rows_as_text(
                            pd.DataFrame(rows_buffer), header_info["columns"]
                        )

                        chunk = {
                            "chunk": chunk_text,
                            "row_index": len(chunks) * chunk_size,
                            "row_count": len(rows_buffer),
                            "doc_id": str(uuid.uuid4()),
                            "metadata": {
                                "chunking_strategy": "streaming_row_by_row",
                                "source_rows": [r.name for r in rows_buffer],
                                "columns": header_info["columns"],
                                "streaming": True,
                            },
                        }
                        chunks.append(chunk)
                        rows_buffer = []  # Clear buffer

                    # Periodic memory check
                    if (
                        len(chunks) % self.processing_config["memory_check_interval"]
                        == 0
                    ):
                        self._check_memory_usage(
                            f"during_streaming_chunk_{len(chunks)}"
                        )

            # Process remaining rows in buffer
            if rows_buffer:
                chunk_text = self._format_rows_as_text(
                    pd.DataFrame(rows_buffer), header_info["columns"]
                )

                chunk = {
                    "chunk": chunk_text,
                    "row_index": len(chunks) * chunk_size,
                    "row_count": len(rows_buffer),
                    "doc_id": str(uuid.uuid4()),
                    "metadata": {
                        "chunking_strategy": "streaming_row_by_row",
                        "source_rows": [r.name for r in rows_buffer],
                        "columns": header_info["columns"],
                        "streaming": True,
                        "final_chunk": True,
                    },
                }
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"Streaming row-by-row processing failed: {e}")
            raise

        return chunks

    def _stream_with_grouping(
        self,
        file_path: str,
        selected_columns: List[str],
        max_rows_per_chunk: int,
        header_info: Dict[str, Any],
    ) -> List[Dict]:
        """
        Stream process CSV with column grouping.

        This is more complex and may require reading the file multiple times
        or using external tools for very large datasets.
        """
        # For Phase 2, implement a simplified version
        # In production, this would need more sophisticated handling

        logger.warning(
            "Column grouping with streaming is simplified for Phase 2. "
            "Large files with column grouping may still use significant memory."
        )

        # Fall back to standard processing with memory warnings
        try:
            df = self._read_csv_file_enhanced(file_path)
            return self.chunker.chunk_dataframe(
                df, selected_columns, max_rows_per_chunk
            )

        except MemoryError:
            # If even standard processing fails, try a very basic approach
            logger.warning("Memory limit reached, using minimal grouping approach")
            return self._stream_row_by_row(file_path, header_info)

    def _format_rows_as_text(self, df: pd.DataFrame, columns: List[str]) -> str:
        """Format DataFrame rows as text."""
        include_headers = self.processing_config["include_headers"]
        delimiter = self.processing_config["delimiter"]

        lines = []

        if include_headers:
            lines.append(f"Columns: {delimiter.join(columns)}")
            lines.append("")

        for _, row in df.iterrows():
            row_parts = [
                f"{col}: {val}" if include_headers else str(val)
                for col, val in row.items()
                if pd.notna(val)
            ]
            lines.append(delimiter.join(row_parts))

        return "\n".join(lines)

    def _read_csv_file_enhanced(self, file_path: str) -> pd.DataFrame:
        """Read CSV file with enhanced error handling and optimization."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                # Read with optimization for large files
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=self.processing_config["delimiter"],
                    low_memory=False,  # Disable low_memory for better type inference
                )

                logger.info(
                    f"Successfully read CSV with {encoding} encoding: {len(df)} rows"
                )
                return df

            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Failed to read CSV with {encoding}: {e}")
                continue

        raise CSVEncodingError(file_path, encodings)

    def _check_memory_usage(self, context: str) -> None:
        """Check current memory usage and log if concerning."""
        try:
            system_resources = CSVOptimizer.get_system_resources()
            memory_mb = system_resources["process_memory"]["rss_mb"]

            # Track peak memory usage
            if memory_mb > self.processing_stats["memory_peak_mb"]:
                self.processing_stats["memory_peak_mb"] = memory_mb

            # Log warnings for high memory usage
            if memory_mb > 1000:  # 1GB
                logger.warning(f"High memory usage at {context}: {memory_mb:.1f}MB")

            if memory_mb > 2000:  # 2GB
                logger.error(f"Very high memory usage at {context}: {memory_mb:.1f}MB")

        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")

    def _extract_enhanced_metadata(
        self, file_path: str, selected_columns: List[str], file_size_mb: float
    ) -> Dict[str, Any]:
        """Extract enhanced metadata from processing."""
        file_path_obj = Path(file_path)

        return {
            "file_type": "CSV",
            "file_name": file_path_obj.name,
            "file_size_mb": file_size_mb,
            "file_path": str(file_path_obj.absolute()),
            "selected_columns": selected_columns,
            "processing_mode": (
                "streaming"
                if file_size_mb > self.processing_config["streaming_threshold_mb"]
                else "standard"
            ),
            "encoding": self.processing_config["encoding"],
            "delimiter": self.processing_config["delimiter"],
            "strategy_version": "enhanced_phase_2",
        }

    def _calculate_enhanced_stats(
        self, chunks: List[Dict], file_size_mb: float
    ) -> Dict[str, Any]:
        """Calculate enhanced processing statistics."""
        base_stats = self.chunker.get_chunking_stats(chunks)

        # Add CSV-specific stats
        base_stats.update(
            {
                "file_size_mb": file_size_mb,
                "processing_strategy": "CSVProcessingStrategy_Enhanced",
                "memory_peak_mb": self.processing_stats["memory_peak_mb"],
                "rows_processed": self.processing_stats["rows_processed"],
                "chunks_created": self.processing_stats["chunks_created"],
                "processing_time_seconds": self.processing_stats[
                    "processing_time_seconds"
                ],
                "throughput_rows_per_second": (
                    self.processing_stats["rows_processed"]
                    / max(1, self.processing_stats["processing_time_seconds"])
                ),
            }
        )

        # Add performance monitoring data if available
        if self.performance_monitor:
            perf_report = self.performance_monitor.get_performance_report()
            base_stats["performance_monitoring"] = perf_report

        return base_stats

    def _read_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Read CSV file with error handling and encoding detection.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame containing the CSV data
        """
        try:
            # Primary attempt with specified encoding
            df = pd.read_csv(
                file_path, encoding=self.encoding, delimiter=self.delimiter
            )
            return df

        except UnicodeDecodeError:
            # Try common alternative encodings
            encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self._logger.info(f"Trying encoding: {encoding}")
                    df = pd.read_csv(file_path, encoding=encoding)
                    self._logger.info(f"Successfully read CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue

            raise ValueError("Unable to read CSV file with any supported encoding")

        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")

        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing error: {str(e)}")

        except Exception as e:
            raise ValueError(f"Unexpected error reading CSV: {str(e)}")

    def _extract_metadata(
        self, df: pd.DataFrame, selected_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Extract metadata from DataFrame.

        Args:
            df: DataFrame to analyze
            selected_columns: List of selected grouping columns

        Returns:
            Dictionary containing metadata
        """
        return {
            "file_type": "CSV",
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "selected_columns": selected_columns,
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_bytes": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "columns": list(df.columns),
            "processing_strategy": self.strategy_name,
            "chunker_config": (
                self.chunker.config if hasattr(self.chunker, 'config') else {}
            ),
        }

    def get_csv_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a CSV file without full processing.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary containing CSV file information
        """
        if not self.validate_file(file_path):
            return {"error": "Invalid file"}

        try:
            # Read sample of CSV for analysis
            df_sample = pd.read_csv(file_path, nrows=100)

            # Get file size
            file_size = Path(file_path).stat().st_size

            return {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "total_columns": len(df_sample.columns),
                "sample_rows": len(df_sample),
                "columns": list(df_sample.columns),
                "column_types": {
                    col: str(dtype) for col, dtype in df_sample.dtypes.items()
                },
                "sample_data": df_sample.head().to_dict('records'),
                "null_counts": df_sample.isnull().sum().to_dict(),
                "strategy_info": self.get_strategy_info(),
            }

        except Exception as e:
            return {"error": f"Error analyzing CSV: {str(e)}"}

    def validate_csv_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Validate CSV file structure and identify potential issues.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary containing validation results
        """
        try:
            # Read CSV for validation
            df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows

            issues = []
            warnings = []

            # Check for empty DataFrame
            if df.empty:
                issues.append("CSV file appears to be empty")
                return {"valid": False, "issues": issues, "warnings": warnings}

            # Check for duplicate columns
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                issues.append(f"Duplicate columns found: {duplicate_cols}")

            # Check for excessive null values
            null_percentages = (df.isnull().sum() / len(df) * 100).round(2)
            high_null_cols = null_percentages[null_percentages > 50].index.tolist()
            if high_null_cols:
                warnings.append(f"Columns with >50% null values: {high_null_cols}")

            # Check column count
            if len(df.columns) > 200:
                warnings.append(
                    f"High column count ({len(df.columns)}) may affect performance"
                )

            # Check for unnamed columns
            unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
            if unnamed_cols:
                warnings.append(f"Unnamed columns detected: {unnamed_cols}")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "summary": {
                    "rows_sampled": len(df),
                    "total_columns": len(df.columns),
                    "duplicate_columns": duplicate_cols,
                    "high_null_columns": high_null_cols,
                    "unnamed_columns": unnamed_cols,
                },
            }

        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
            }
