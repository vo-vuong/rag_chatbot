"""
Tool registry for agent workflows.

Provides factory functions to create tool sets for different workflows.
"""

from typing import List

from langchain_core.tools import BaseTool

from backend.agent.tools.retrieval import create_retrieval_tool


def get_rag_tools(rag_service) -> List[BaseTool]:
    """
    Get tool set for RAG workflows.

    Args:
        rag_service: Configured RAGService instance

    Returns:
        List of tools including retrieval
    """
    return [create_retrieval_tool(rag_service)]


def get_all_tools(rag_service) -> List[BaseTool]:
    """
    Get complete tool set including future tools.

    Args:
        rag_service: Configured RAGService instance

    Returns:
        List of all available tools
    """
    tools = get_rag_tools(rag_service)
    # Future: Add web_search, calculator, etc.
    return tools
