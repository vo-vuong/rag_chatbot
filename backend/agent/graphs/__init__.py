"""
Graph factory - Central point for graph creation.
"""

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

from backend.agent.config import WorkflowType
from backend.agent.graphs.agentic_rag import create_agentic_rag_graph


def create_graph(
    workflow_type: WorkflowType,
    rag_service,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    checkpointer: Optional[BaseCheckpointSaver] = None,
    max_iterations: int = 5,
):
    """
    Factory function to create workflow graphs.

    Args:
        workflow_type: Type of workflow to create
        rag_service: Configured RAGService
        openai_api_key: OpenAI API key
        model: LLM model name
        checkpointer: Optional persistence layer
        max_iterations: Max tool iterations for agentic workflow

    Returns:
        Compiled StateGraph
    """
    # Note: checkpointer=None means stateless mode (for evaluation)
    # Do NOT create default MemorySaver here

    if workflow_type == WorkflowType.AGENTIC_RAG:
        return create_agentic_rag_graph(
            rag_service=rag_service,
            openai_api_key=openai_api_key,
            model=model,
            checkpointer=checkpointer,
            max_iterations=max_iterations,
        )

    elif workflow_type == WorkflowType.SIMPLE_RAG:
        # Future: Implement simple linear RAG graph
        raise NotImplementedError(
            "Simple RAG workflow not yet implemented. Use AGENTIC_RAG."
        )

    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")


__all__ = ["create_graph", "WorkflowType", "create_agentic_rag_graph"]
