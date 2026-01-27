"""
LangGraph Workflow Visualization Utility.

Usage:
    # From project root
    conda activate rag_chatbot && python -m backend.agent.visualize

    # With custom output path
    conda activate rag_chatbot && python -m backend.agent.visualize --output my_graph.png

    # Show in terminal (ASCII) instead of saving PNG
    conda activate rag_chatbot && python -m backend.agent.visualize --ascii
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockRAGService:
    """Mock RAGService for graph visualization (no DB connection needed)."""

    def route_query(self, query):
        return "text_only", "mock"

    def search_text(self, query, top_k=5, score_threshold=0.7):
        return []

    def search_images(self, query, top_k=1, score_threshold=0.7):
        return []


def create_visualization(output_path: str = "agent_graph.png", ascii_mode: bool = False):
    """
    Generate LangGraph workflow visualization.

    Args:
        output_path: Path to save PNG file
        ascii_mode: If True, print ASCII representation instead of saving PNG
    """
    from api.dependencies import get_settings

    from backend.agent.config import WorkflowType
    from backend.agent.graphs import create_graph

    settings = get_settings()

    logger.info("Creating agent graph...")
    graph = create_graph(
        WorkflowType.AGENTIC_RAG,
        MockRAGService(),
        settings.openai_api_key,
    )

    if ascii_mode:
        # Print ASCII representation
        print("\n" + "=" * 50)
        print("LangGraph Workflow Structure")
        print("=" * 50)
        print(graph.get_graph().draw_ascii())
        print("=" * 50 + "\n")
    else:
        # Save as PNG
        png_data = graph.get_graph().draw_mermaid_png()
        output_file = Path(output_path)
        output_file.write_bytes(png_data)
        logger.info(f"Graph saved to: {output_file.absolute()}")
        print(f"\nGraph visualization saved to: {output_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LangGraph workflow visualization"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="agent_graph.png",
        help="Output PNG file path (default: agent_graph.png)",
    )
    parser.add_argument(
        "--ascii",
        "-a",
        action="store_true",
        help="Print ASCII representation instead of saving PNG",
    )

    args = parser.parse_args()
    create_visualization(output_path=args.output, ascii_mode=args.ascii)


if __name__ == "__main__":
    main()
