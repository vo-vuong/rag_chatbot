"""
CSV performance optimization utilities with memory monitoring and benchmarks.

This module provides comprehensive performance monitoring and optimization
utilities for CSV processing operations.
"""

import gc
import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import psutil

logger = logging.getLogger(__name__)


class CSVPerformanceMonitor:
    """Monitor and optimize CSV processing performance."""

    def __init__(self):
        self.start_time = None
        self.memory_baseline = None
        self.checkpoints = []

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.memory_baseline = self._get_memory_usage()
        self.checkpoints = []
        logger.info(
            f"Performance monitoring started - baseline memory: {self.memory_baseline:.1f}MB"
        )

    def add_checkpoint(
        self,
        name: str,
        rows_processed: int = 0,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Add a performance checkpoint."""
        current_time = time.time()
        memory_usage = self._get_memory_usage()

        checkpoint = {
            "name": name,
            "elapsed_time": current_time - self.start_time if self.start_time else 0,
            "memory_usage_mb": memory_usage,
            "memory_increase_mb": (
                memory_usage - self.memory_baseline if self.memory_baseline else 0
            ),
            "rows_processed": rows_processed,
            "throughput_rows_per_second": (
                rows_processed / max(1, current_time - self.start_time)
                if self.start_time
                else 0
            ),
            "additional_info": additional_info or {},
        }

        self.checkpoints.append(checkpoint)
        logger.info(
            f"Checkpoint '{name}': {checkpoint['elapsed_time']:.2f}s, "
            f"{checkpoint['memory_increase_mb']:.1f}MB increase"
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.checkpoints:
            return {"error": "No checkpoints recorded"}

        total_time = self.checkpoints[-1]["elapsed_time"]
        total_memory_increase = self.checkpoints[-1]["memory_increase_mb"]
        final_checkpoint = self.checkpoints[-1]

        # Calculate memory efficiency
        memory_efficiency = self._calculate_memory_efficiency()

        report = {
            "summary": {
                "total_processing_time_seconds": total_time,
                "total_memory_increase_mb": total_memory_increase,
                "final_throughput_rows_per_second": final_checkpoint.get(
                    "throughput_rows_per_second", 0
                ),
                "checkpoints_count": len(self.checkpoints),
                "memory_efficiency_score": memory_efficiency,
            },
            "checkpoints": self.checkpoints,
            "recommendations": self._generate_recommendations(),
            "memory_timeline": self._create_memory_timeline(),
            "performance_metrics": self._calculate_performance_metrics(),
        }

        return report

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0

    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score (0-100)."""
        if not self.checkpoints:
            return 0

        max_memory_increase = max(
            checkpoint["memory_increase_mb"] for checkpoint in self.checkpoints
        )
        total_rows = max(
            checkpoint["rows_processed"] for checkpoint in self.checkpoints
        )

        # Efficiency based on rows processed per MB of memory increase
        if max_memory_increase > 0:
            rows_per_mb = total_rows / max_memory_increase
            # Score 0-100, where 1000+ rows/MB = 100 points
            return min(100, (rows_per_mb / 1000) * 100)
        return 100  # Perfect if no memory increase

    def _create_memory_timeline(self) -> List[Dict[str, Any]]:
        """Create memory usage timeline."""
        return [
            {
                "time": checkpoint["elapsed_time"],
                "memory_mb": checkpoint["memory_usage_mb"],
                "memory_increase_mb": checkpoint["memory_increase_mb"],
                "checkpoint_name": checkpoint["name"],
            }
            for checkpoint in self.checkpoints
        ]

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        if len(self.checkpoints) < 2:
            return {}

        # Calculate throughput at different stages
        throughputs = [
            cp.get("throughput_rows_per_second", 0)
            for cp in self.checkpoints
            if cp.get("throughput_rows_per_second", 0) > 0
        ]

        # Calculate memory growth rate
        memory_growth_rates = []
        for i in range(1, len(self.checkpoints)):
            prev_cp = self.checkpoints[i - 1]
            curr_cp = self.checkpoints[i]

            time_diff = curr_cp["elapsed_time"] - prev_cp["elapsed_time"]
            memory_diff = curr_cp["memory_increase_mb"] - prev_cp["memory_increase_mb"]

            if time_diff > 0:
                growth_rate = memory_diff / time_diff
                memory_growth_rates.append(growth_rate)

        return {
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "max_throughput": max(throughputs) if throughputs else 0,
            "min_throughput": min(throughputs) if throughputs else 0,
            "avg_memory_growth_rate_mb_per_sec": (
                sum(memory_growth_rates) / len(memory_growth_rates)
                if memory_growth_rates
                else 0
            ),
            "peak_memory_mb": max(cp["memory_usage_mb"] for cp in self.checkpoints),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if not self.checkpoints:
            return recommendations

        total_memory_increase = self.checkpoints[-1]["memory_increase_mb"]
        final_throughput = self.checkpoints[-1].get("throughput_rows_per_second", 0)
        memory_efficiency = self._calculate_memory_efficiency()

        # Memory-related recommendations
        if total_memory_increase > 500:  # 500MB increase
            recommendations.append(
                f"High memory usage detected ({total_memory_increase:.1f}MB). "
                "Consider enabling streaming processing or reducing chunk sizes."
            )

        if memory_efficiency < 50:
            recommendations.append(
                f"Low memory efficiency ({memory_efficiency:.1f}/100). "
                "Consider using memory optimization techniques."
            )

        # Performance-related recommendations
        if final_throughput < 1000:  # Less than 1000 rows/second
            recommendations.append(
                f"Low processing throughput ({final_throughput:.0f} rows/sec). "
                "Check for I/O bottlenecks or consider data type optimization."
            )

        # Check for memory leaks (continuous increase)
        if len(self.checkpoints) > 3:
            recent_increases = []
            for i in range(2, min(5, len(self.checkpoints))):
                curr_mem = self.checkpoints[i]["memory_increase_mb"]
                prev_mem = self.checkpoints[i - 1]["memory_increase_mb"]
                recent_increases.append(curr_mem - prev_mem)

            if all(inc > 50 for inc in recent_increases):  # Consistent 50MB+ increases
                recommendations.append(
                    "Potential memory leak detected. "
                    "Review data handling and cleanup procedures."
                )

        return recommendations


class CSVOptimizer:
    """Utilities for optimizing CSV processing performance."""

    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage aggressively.

        Returns optimized DataFrame and memory reduction percentage.
        """
        original_memory = df.memory_usage(deep=True).sum()

        # Convert to categorical for low-cardinality columns
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')

        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        # Convert to datetime where appropriate
        date_columns = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].dtype.name == 'category':
                # Skip categorical columns for datetime conversion
                continue

            # Heuristic: check if column name suggests date/time
            col_lower = col.lower()
            if any(
                keyword in col_lower
                for keyword in ['date', 'time', 'created', 'updated', 'timestamp']
            ):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    date_columns.append(col)
                except Exception:
                    pass

        # Convert boolean columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col in date_columns:
                continue

            unique_values = df[col].dropna().unique()
            if len(unique_values) == 2 and all(
                str(val).lower() in ['true', 'false', 'yes', 'no', '1', '0']
                for val in unique_values
            ):
                df[col] = df[col].astype('boolean')

        optimized_memory = df.memory_usage(deep=True).sum()
        reduction_percentage = (
            (original_memory - optimized_memory) / original_memory
        ) * 100

        logger.info(
            f"DataFrame memory optimized: {reduction_percentage:.1f}% reduction "
            f"({original_memory / 1024 / 1024:.1f}MB -> {optimized_memory / 1024 / 1024:.1f}MB)"
        )

        return df

    @staticmethod
    def estimate_processing_time(
        file_size_mb: float,
        rows_estimate: Optional[int] = None,
        complexity_factor: float = 1.0,
    ) -> Dict[str, float]:
        """
        Estimate CSV processing time based on file characteristics.

        Returns estimated times for different operations.
        """
        # Base processing rates (rows per second)
        base_read_rate = 50000  # Conservative estimate
        base_chunking_rate = 10000
        base_embedding_rate = 100

        # Adjust for file size (larger files are typically slower)
        size_factor = min(2.0, 1.0 + (file_size_mb / 100))  # Max 2x slower for 100MB+

        # Estimate row count if not provided
        if rows_estimate is None:
            # Rough estimate: 1 row ≈ 100 bytes average
            rows_estimate = int((file_size_mb * 1024 * 1024) / 100)

        # Calculate estimated times
        read_time = (rows_estimate / base_read_rate) * size_factor * complexity_factor
        chunking_time = (
            (rows_estimate / base_chunking_rate) * size_factor * complexity_factor
        )
        embedding_time = (
            (rows_estimate / base_embedding_rate) * size_factor * complexity_factor
        )

        total_time = read_time + chunking_time + embedding_time

        return {
            "read_time_seconds": read_time,
            "chunking_time_seconds": chunking_time,
            "embedding_time_seconds": embedding_time,
            "total_time_seconds": total_time,
            "estimated_rows": rows_estimate,
        }

    @staticmethod
    def get_optimal_chunk_size(
        file_size_mb: float, available_memory_mb: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Calculate optimal chunk size based on file characteristics and available memory.

        Returns recommended chunk sizes for different strategies.
        """
        if available_memory_mb is None:
            # Estimate available memory (use 70% of total as conservative estimate)
            total_memory = psutil.virtual_memory().total / 1024 / 1024
            available_memory_mb = total_memory * 0.7

        # Base chunk sizes
        base_row_chunk = 10
        base_char_chunk = 2000

        # Adjust based on file size
        if file_size_mb < 1:
            # Small file - can use larger chunks
            row_multiplier = 2.0
            char_multiplier = 1.5
        elif file_size_mb < 10:
            # Medium file - standard chunks
            row_multiplier = 1.0
            char_multiplier = 1.0
        else:
            # Large file - smaller chunks for memory efficiency
            row_multiplier = 0.5
            char_multiplier = 0.8

        # Adjust based on available memory
        memory_factor = min(
            2.0, available_memory_mb / 1000
        )  # Scale with available memory

        optimal_rows = max(1, int(base_row_chunk * row_multiplier * memory_factor))
        optimal_chars = max(500, int(base_char_chunk * char_multiplier * memory_factor))

        return {
            "optimal_rows_per_chunk": optimal_rows,
            "optimal_chars_per_chunk": optimal_chars,
            "max_safe_rows": min(100, optimal_rows * 2),
            "max_safe_chars": min(5000, optimal_chars * 2),
        }

    @staticmethod
    def benchmark_csv_processing(
        file_path: str, processing_function, **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark CSV processing performance.

        Args:
            file_path: Path to CSV file
            processing_function: Function to benchmark
            **kwargs: Arguments for processing function

        Returns:
            Benchmark results
        """
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # Get system info
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "memory_available_gb": psutil.virtual_memory().available
            / 1024
            / 1024
            / 1024,
        }

        # Run benchmark
        monitor = CSVPerformanceMonitor()
        monitor.start_monitoring()

        start_time = time.time()
        gc.collect()  # Clean up before benchmark

        try:
            result = processing_function(file_path, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            result = None

        end_time = time.time()

        # Final checkpoint
        monitor.add_checkpoint(
            "processing_complete",
            additional_info={
                "success": success,
                "error": error,
                "file_size_mb": file_size_mb,
            },
        )

        # Calculate performance metrics
        total_time = end_time - start_time
        performance_report = monitor.get_performance_report()

        benchmark_results = {
            "file_info": {
                "path": file_path,
                "size_mb": file_size_mb,
                "size_category": (
                    "small"
                    if file_size_mb < 1
                    else "medium" if file_size_mb < 10 else "large"
                ),
            },
            "system_info": system_info,
            "performance": {
                "total_time_seconds": total_time,
                "processing_rate_mb_per_sec": (
                    file_size_mb / total_time if total_time > 0 else 0
                ),
                "success": success,
                "error": error,
            },
            "detailed_monitoring": performance_report,
            "result_summary": {
                "chunks_created": (
                    len(result) if result and isinstance(result, list) else 0
                ),
                "memory_efficiency": performance_report.get("summary", {}).get(
                    "memory_efficiency_score", 0
                ),
            },
        }

        return benchmark_results

    @staticmethod
    def cleanup_memory():
        """Force garbage collection to free memory."""
        gc.collect()
        logger.info("Memory cleanup completed")

    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        process = psutil.Process()

        return {
            "system_memory": {
                "total_gb": memory.total / 1024 / 1024 / 1024,
                "available_gb": memory.available / 1024 / 1024 / 1024,
                "used_gb": memory.used / 1024 / 1024 / 1024,
                "percent_used": memory.percent,
            },
            "process_memory": {
                "rss_mb": process.memory_info().rss / 1024 / 1024,
                "vms_mb": process.memory_info().vms / 1024 / 1024,
                "percent_of_system": (process.memory_info().rss / memory.total) * 100,
            },
            "cpu": {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
            },
        }
