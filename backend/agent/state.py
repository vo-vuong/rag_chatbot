"""
LangGraph state schemas for RAG agent workflows.

Uses TypedDict with Annotated reducers for proper message accumulation.
"""

from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RetrievalResult(TypedDict):
    """Result from retrieval tool invocation."""

    chunks: List[Dict[str, Any]]  # Serialized ChunkElement list
    images: List[Dict[str, Any]]  # Serialized ImageElement list
    query: str
    route: str  # "text_only" or "image_only"
    reasoning: str


class AgentState(TypedDict):
    """
    Main state schema for RAG agent graph.

    Uses add_messages reducer to accumulate conversation history.
    Tool results stored separately for structured access.
    """

    # Core message history (HumanMessage, AIMessage, ToolMessage)
    messages: Annotated[Sequence[AnyMessage], add_messages]

    # Current user query (extracted for tool use)
    query: str

    # Session ID for persistence
    session_id: str

    # Retrieval results (populated by retrieval tool/node)
    retrieval_result: Optional[RetrievalResult]

    # Routing decision
    route: Optional[str]  # "text_only", "image_only", "no_retrieval"

    # Control flow
    iteration_count: int  # Prevent infinite loops


@dataclass
class AgentResponse:
    """Response container for agent output."""

    response: str
    route: Optional[str]
    route_reasoning: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    tool_calls_made: int = 0
    iterations: int = 0
