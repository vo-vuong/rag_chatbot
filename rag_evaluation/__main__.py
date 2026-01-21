"""
RAG Evaluation module entry point.

Enables running the evaluation framework as a module:
    python -m rag_evaluation --metric hit --k 5
"""

from rag_evaluation.cli import main

if __name__ == "__main__":
    main()
