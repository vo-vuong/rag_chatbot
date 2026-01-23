# RAG Evaluation Framework

A modular, extensible framework for evaluating RAG performance using retrieval and generation metrics.

## Overview

This module provides tools to evaluate:
1. **Retrieval Quality**: How well your RAG system retrieves relevant documents (Hit@K, Recall@K, etc.)
2. **Generation Quality**: How faithful and accurate the LLM responses are (Faithfulness via RAGAS)

## Quick Start

```bash
# Run Hit@K evaluation with k=5
conda activate rag_chatbot && python -m rag_evaluation --metric hit --k 5

# Run Recall@K with verbose output
conda activate rag_chatbot && python -m rag_evaluation --metric recall --k 10 -v

# Run all metrics (retrieval + generation)
conda activate rag_chatbot && python -m rag_evaluation --metric all --k 5

# Run all retrieval metrics only
conda activate rag_chatbot && python -m rag_evaluation --metric all_retrieval --k 5

# Run all generation metrics only
conda activate rag_chatbot && python -m rag_evaluation --metric all_generation --k 5

# Run Faithfulness (generation metric)
conda activate rag_chatbot && python -m rag_evaluation --metric faithfulness --k 5

# List available metrics
conda activate rag_chatbot && python -m rag_evaluation --list-metrics
```

## Available Metrics

### Retrieval Metrics

| Metric | Short Name | Description |
|--------|------------|-------------|
| Hit@K | `hit` | Proportion of queries where at least one relevant document appears in top-K |
| Recall@K | `recall` | Average proportion of relevant documents retrieved in top-K |
| Precision@K | `precision` | Average proportion of retrieved documents that are relevant |
| F1@K | `f1` | Harmonic mean of Precision@K and Recall@K |
| MRR@K | `mrr` | Mean Reciprocal Rank of first relevant document |

### Generation Metrics

| Metric | Short Name | Description |
|--------|------------|-------------|
| Faithfulness | `faithfulness` | Measures factual consistency between LLM response and retrieved context (via RAGAS) |
| Response Relevancy | `response_relevancy` | Measures how relevant the response is to the user's question (via RAGAS) |
| Context Precision | `context_precision` | Measures how well relevant chunks are ranked higher in retrieved results (via RAGAS) |

## CLI Usage

```bash
python -m rag_evaluation [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --metric` | Metric(s) to run: `hit`, `recall`, `precision`, `f1`, `mrr`, `faithfulness`, `response_relevancy`, `context_precision`, `all`, `all_retrieval`, `all_generation` | `all` |
| `-k, --k` | Number of top results to consider | `5` |
| `-t, --threshold` | Minimum similarity score threshold | `0.0` |
| `--test-data` | Path to test data Excel file | `qr_smartphone_dataset.xlsx` |
| `--api-url` | RAG API base URL | `http://localhost:8000` |
| `--limit` | Limit number of queries to evaluate | all |
| `-v, --verbose` | Print detailed per-query results | off |
| `-o, --output` | Custom output Excel file path | auto-generated |
| `--no-export` | Don't export results to Excel | off |
| `--list-metrics` | List available metrics and exit | - |
| `--model` | LLM model for generation metrics | `gpt-4o-mini` |
| `--embedding-model` | Embedding model for response_relevancy | `text-embedding-3-small` |

### Examples

```bash
# Multiple retrieval metrics
python -m rag_evaluation --metric hit recall --k 5

# All metrics (retrieval + generation)
python -m rag_evaluation --metric all --k 5

# All retrieval metrics only
python -m rag_evaluation --metric all_retrieval --k 5

# All generation metrics
python -m rag_evaluation --metric all_generation --k 5

# Faithfulness with custom model
python -m rag_evaluation --metric faithfulness --k 5 --model gpt-4o

# Response Relevancy with custom embedding model
python -m rag_evaluation --metric response_relevancy --k 5 --embedding-model text-embedding-3-large

# Context Precision (requires ground_truth_answer in test data)
python -m rag_evaluation --metric context_precision --k 5

# Custom test data with score threshold
python -m rag_evaluation --metric recall --k 10 -t 0.7 --test-data custom_data.xlsx

# Limit queries for quick testing
python -m rag_evaluation --metric faithfulness --k 5 --limit 10 -v
```

## Programmatic Usage

### Retrieval Metrics

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

### Generation Metrics (Faithfulness)

```python
from rag_evaluation.generation_evaluator import GenerationEvaluator

# Create evaluator
evaluator = GenerationEvaluator(
    test_data_path="path/to/test_data.xlsx",
    api_base_url="http://localhost:8000",
    limit=None,
)

# Run Faithfulness evaluation
result = evaluator.run(
    metric="faithfulness",
    top_k=5,
    score_threshold=0.0,
    verbose=True,
    model_name="gpt-4o-mini",  # or "gpt-4o"
    export=True,  # Export results to Excel
)

# Access results
print(f"Faithfulness Score: {result['summary']['score']:.4f}")
print(f"Total queries: {result['summary']['total_queries']}")
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
| `Ground_truth_answer` | (Optional) Expected answer for generation metrics |
| `difficulty` | (Optional) Query difficulty level |
| `source_files` | (Optional) Source file references |

## Module Structure

```
rag_evaluation/
├── __init__.py              # Package exports
├── __main__.py              # Entry point for python -m
├── cli.py                   # Command-line interface
├── evaluator.py             # Retrieval metrics orchestrator
├── generation_evaluator.py  # Generation metrics orchestrator
├── base/
│   ├── metric_interface.py           # Abstract base for retrieval metrics
│   ├── generation_metric_interface.py # Abstract base for generation metrics
│   └── evaluation_result.py          # Result dataclasses
├── metrics/
│   ├── registry.py          # Metric registration
│   ├── hit_at_k.py          # Hit@K implementation
│   ├── recall_at_k.py       # Recall@K implementation
│   ├── precision_at_k.py    # Precision@K implementation
│   ├── f1_at_k.py           # F1@K implementation
│   ├── mrr_at_k.py          # MRR@K implementation
│   ├── faithfulness.py      # Faithfulness (RAGAS) implementation
│   ├── response_relevancy.py # Response Relevancy (RAGAS) implementation
│   └── context_precision.py # Context Precision (RAGAS) implementation
├── data/
│   ├── data_loader.py       # Test data loading
│   └── point_id_parser.py   # ID parsing utilities
├── api/
│   └── rag_api_client.py    # RAG API client (search + chat)
├── export/
│   └── excel_exporter.py    # Excel export
├── results/                 # Default output directory
└── prepare_testing_data/
    └── qr_smartphone_dataset.xlsx  # Sample test data
```

## Faithfulness Metric Details

### How It Works

Faithfulness measures how **factually consistent** the LLM response is with the retrieved context.

**Formula:**
```
Faithfulness = (Claims supported by context) / (Total claims in response)
```

**Process:**
1. Extract all factual claims from the LLM response
2. Verify each claim against the retrieved contexts
3. Calculate the ratio of supported claims

### Requirements

- **RAGAS library**: `pip install ragas`
- **OpenAI API key**: Set `OPENAI_API_KEY` environment variable
- **RAG API running**: The `/api/v1/chat/query` endpoint must be available

### Inputs (from API)

| Input | Source | Description |
|-------|--------|-------------|
| `user_input` | Test data `Query` column | The user's question |
| `response` | API `response` field | LLM-generated answer |
| `retrieved_contexts` | API `retrieved_chunks[].text` | Context texts |

**Note**: Faithfulness does NOT require `Ground_truth_answer` from test data.

## Context Precision Metric Details

### How It Works

Context Precision evaluates how well the retriever ranks **relevant chunks higher** in the retrieved results.

**Formula:**
```
Context Precision = Mean of Precision@K for each chunk
Precision@K = (Relevant chunks at rank K) / K
```

**Process:**
1. Retrieve contexts using `/rag/search` API
2. Use LLM to compare each retrieved chunk against the `Ground_truth_answer`
3. Determine if each chunk is relevant
4. Calculate weighted precision based on position

### Requirements

- **RAGAS library**: `pip install ragas`
- **OpenAI API key**: Set `OPENAI_API_KEY` environment variable
- **Ground truth answer**: Requires `Ground_truth_answer` column in test data
- **RAG API running**: The `/api/v1/rag/search` endpoint must be available

### Inputs

| Input | Source | Description |
|-------|--------|-------------|
| `user_input` | Test data `Query` column | The user's question |
| `reference` | Test data `Ground_truth_answer` column | Expected answer for relevance judgment |
| `retrieved_contexts` | API `/rag/search` response | Context texts from search results |

### Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | All relevant chunks ranked at top positions |
| 0.5 | Relevant chunks scattered throughout ranking |
| 0.0 | No relevant chunks or all at bottom |

## Output

Results are exported to Excel files in `rag_evaluation/results/`.

### Export Behavior

| Command | Output Files |
|---------|--------------|
| `--metric all` | `retrieval_combined_k5_xxx.xlsx` + `generation_combined_k5_xxx.xlsx` |
| `--metric all_retrieval` | `retrieval_combined_k5_xxx.xlsx` |
| `--metric all_generation` | `generation_combined_k5_xxx.xlsx` |
| `--metric hit` | `Hit@K_k5_xxx.xlsx` (single metric) |
| `--metric faithfulness` | `Faithfulness_k5_xxx.xlsx` (single metric) |
| `--no-export` | No files exported |

### Retrieval Metrics Output

- **Summary sheet**: Overall metric scores and configuration
- **Per-Query Results sheet**: query, ground_truth_ids, retrieved_ids, score

### Generation Metrics Output

- **Summary sheet**: Metric score, model, top_k, score statistics
- **Per-Query Results sheet**: query, response, ground_truth_answer, retrieved_contexts, score
