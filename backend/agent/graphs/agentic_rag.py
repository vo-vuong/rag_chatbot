"""
Agentic RAG Graph - ReAct-style agent with retrieval tool.

Best for: Complex queries where agent decides if/when to retrieve.
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.agent.nodes import create_agent_node
from backend.agent.state import AgentState
from backend.agent.tools import get_rag_tools

logger = logging.getLogger(__name__)


def create_agentic_rag_graph(
    rag_service,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    checkpointer: Optional[BaseCheckpointSaver] = None,
    max_iterations: int = 5,
):
    """
    Create agentic RAG graph with tool-calling capability.

    Architecture:
        START → agent_node → [tools_condition] → tools → agent_node → ... → END
                                    ↓
                                    END (if no tool calls)

    Args:
        rag_service: RAGService instance
        openai_api_key: OpenAI API key
        model: Model name (default: gpt-4o-mini)
        checkpointer: Optional checkpointer for persistence
        max_iterations: Max tool-calling loops (safety limit)

    Returns:
        Compiled StateGraph
    """
    logger.info(f"Creating agentic RAG graph with model={model}")

    # Initialize tools
    tools = get_rag_tools(rag_service)

    # Initialize LLM
    llm = ChatOpenAI(
        model=model,
        api_key=openai_api_key,
        temperature=0.2,
    )

    # Create agent node
    agent_node = create_agent_node(llm, tools, max_iterations)

    # Build graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools=tools))

    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,  # Built-in: checks for tool_calls in last message
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")  # Return to agent after tool execution

    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Agentic RAG graph compiled successfully")

    return graph
