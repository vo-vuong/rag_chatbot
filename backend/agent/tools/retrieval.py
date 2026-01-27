"""
RAG Retrieval Tool - Wraps existing RAGService for LangGraph tool use.

Provides both text and image retrieval as a single unified tool.
"""

import logging
from typing import Any, Dict

from langchain_core.tools import tool

from backend.agent.tools.base import serialize_chunk, serialize_image

logger = logging.getLogger(__name__)


def create_retrieval_tool(rag_service):
    """
    Factory function to create retrieval tool with injected RAGService.

    Args:
        rag_service: Configured RAGService instance

    Returns:
        Decorated tool function
    """

    @tool
    def retrieve_documents(
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        search_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from the knowledge base.

        Use this tool when you need information from the uploaded documents
        to answer the user's question. The tool automatically routes to
        text or image collections based on query content.

        Args:
            query: Search query derived from user's question
            top_k: Number of documents to retrieve (default: 5)
            score_threshold: Minimum relevance score (default: 0.7)
            search_type: "auto" for automatic routing, "text" for text only,
                        "image" for images only

        Returns:
            Dictionary with 'chunks', 'images', 'route', and 'reasoning'
        """
        logger.info(
            f"Retrieval tool called: query='{query}', search_type={search_type}"
        )

        if search_type == "auto":
            route, reasoning = rag_service.route_query(query)
        else:
            route = f"{search_type}_only"
            reasoning = f"Manual {search_type} search requested"

        chunks = []
        images = []

        try:
            if route == "text_only" or search_type == "text":
                chunk_results = rag_service.search_text(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                chunks = [serialize_chunk(c) for c in chunk_results]
                logger.info(f"Retrieved {len(chunks)} text chunks")

            elif route == "image_only" or search_type == "image":
                image_results = rag_service.search_images(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                images = [serialize_image(img) for img in image_results]
                logger.info(f"Retrieved {len(images)} images")

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "chunks": [],
                "images": [],
                "route": route,
                "reasoning": f"Retrieval failed: {str(e)}",
                "query": query,
                "error": str(e),
            }

        return {
            "chunks": chunks,
            "images": images,
            "route": route,
            "reasoning": reasoning,
            "query": query,
        }

    return retrieve_documents
