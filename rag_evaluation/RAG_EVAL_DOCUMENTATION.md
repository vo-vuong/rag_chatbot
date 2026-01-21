# RAG Evaluation Framework

A modular, extensible framework for evaluating RAG retrieval performance using standardized metrics.

## Overview

This module provides tools to evaluate how well your RAG system retrieves relevant documents. It supports multiple metrics, configurable parameters, and exports results to Excel for analysis.

## Quick Start

```bash
# Run Hit@K evaluation with k=5
conda activate rag_chatbot && python -m rag_evaluation --metric hit --k 5

# Run Recall@K with verbose output
conda activate rag_chatbot && python -m rag_evaluation --metric recall --k 10 -v

# Run all metrics
conda activate rag_chatbot && python -m rag_evaluation --metric all --k 5

# List available metrics
conda activate rag_chatbot && python -m rag_evaluation --list-metrics
```

## Available Metrics

| Metric | Short Name | Description |
|--------|------------|-------------|
| Hit@K | `hit` | Proportion of queries where at least one relevant document appears in top-K |
| Recall@K | `recall` | Average proportion of relevant documents retrieved in top-K |

## CLI Usage

```bash
python -m rag_evaluation [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --metric` | Metric(s) to run: `hit`, `recall`, `all` | `all` |
| `-k, --k` | Number of top results to consider | `5` |
| `-t, --threshold` | Minimum similarity score threshold | `0.0` |
| `--test-data` | Path to test data Excel file | `qr_smartphone_dataset.xlsx` |
| `--api-url` | RAG API base URL | `http://localhost:8000` |
| `--limit` | Limit number of queries to evaluate | all |
| `-v, --verbose` | Print detailed per-query results | off |
| `-o, --output` | Custom output Excel file path | auto-generated |
| `--no-export` | Don't export results to Excel | off |
| `--list-metrics` | List available metrics and exit | - |

### Examples

```bash
# Multiple metrics
python -m rag_evaluation --metric hit recall --k 5

# Custom test data with score threshold
python -m rag_evaluation --metric recall --k 10 -t 0.7 --test-data custom_data.xlsx

# Limit queries for quick testing
python -m rag_evaluation --metric hit --k 5 --limit 10 -v

# Custom output path
python -m rag_evaluation --metric all --k 5 -o my_results.xlsx
```

## Programmatic Usage

```python
from rag_evaluation import Evaluator

# Create evaluator
evaluator = Evaluator(
    test_data_path="path/to/test_data.xlsx",
    api_base_url="http://localhost:8000",
    limit=None,  # Optional: limit queries
)

# Run evaluation
results = evaluator.run(
    metrics=["hit", "recall"],  # or "all" or single metric
    k=5,
    score_threshold=0.0,
    verbose=True,
    export=True,
)

# Access results
for name, result in results.items():
    print(f"{result.metric_name}: {result.score:.4f}")
    print(f"  Total queries: {result.total_queries}")
```

## Adding New Metrics

The framework uses a Strategy + Registry pattern for easy extensibility.

### Step 1: Create Metric Class

Create a new file in `metrics/` (e.g., `metrics/mrr_at_k.py`):

```python
from typing import List, Tuple

from rag_evaluation.base.metric_interface import RetrievalMetric
from rag_evaluation.base.evaluation_result import QueryResult, MetricSummary
from rag_evaluation.metrics.registry import register_metric


@register_metric("mrr")
class MRRAtK(RetrievalMetric):
    """Mean Reciprocal Rank metric."""

    name: str = "MRR@K"
    short_name: str = "mrr"

    def calculate_query_score(
        self,
        ground_truth_ids: List[int],
        retrieved_ids: List[int],
    ) -> Tuple[float, dict]:
        """Calculate MRR for a single query."""
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in ground_truth_ids:
                return 1.0 / rank, {"first_hit_rank": rank}
        return 0.0, {"first_hit_rank": None}

    def aggregate_scores(
        self,
        query_results: List[QueryResult],
        k: int,
        score_threshold: float,
    ) -> MetricSummary:
        """Aggregate MRR scores."""
        if not query_results:
            return MetricSummary(
                metric_name=self.name,
                k=k,
                score=0.0,
                total_queries=0,
                score_threshold=score_threshold,
            )

        total = len(query_results)
        avg_mrr = sum(qr.score for qr in query_results) / total

        return MetricSummary(
            metric_name=self.name,
            k=k,
            score=avg_mrr,
            total_queries=total,
            score_threshold=score_threshold,
        )
```

### Step 2: Register the Metric

Import the new metric in `cli.py` to trigger registration:

```python
import rag_evaluation.metrics.mrr_at_k  # noqa: F401
```

### Step 3: Use the Metric

```bash
python -m rag_evaluation --metric mrr --k 5
```

## Test Data Format

The test data Excel file should have the following columns:

| Column | Description |
|--------|-------------|
| `query` | The search query text |
| `point_ids` | Ground truth document IDs (comma-separated or JSON list) |
| `difficulty` | (Optional) Query difficulty level |
| `source_files` | (Optional) Source file references |

## Module Structure

```
rag_evaluation/
├── __init__.py          # Package exports
├── __main__.py          # Entry point for python -m
├── cli.py               # Command-line interface
├── evaluator.py         # Main evaluation orchestrator
├── base/
│   ├── metric_interface.py    # Abstract base for metrics
│   └── evaluation_result.py   # Result dataclasses
├── metrics/
│   ├── registry.py      # Metric registration
│   ├── hit_at_k.py      # Hit@K implementation
│   └── recall_at_k.py   # Recall@K implementation
├── data/
│   ├── data_loader.py   # Test data loading
│   └── point_id_parser.py  # ID parsing utilities
├── api/
│   └── rag_api_client.py  # RAG API client
├── export/
│   └── excel_exporter.py  # Excel export
├── results/             # Default output directory
└── prepare_testing_data/
    └── qr_smartphone_dataset.xlsx  # Sample test data
```

## Output

Results are exported to Excel with:
- **Summary sheet**: Overall metric scores and configuration
- **Details sheet**: Per-query results with ground truth vs retrieved IDs

Default output location: `rag_evaluation/results/`
