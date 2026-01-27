"""
Main agent decision node for LangGraph workflows.

Handles LLM invocation with bound tools and iteration control.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agent.prompts import get_agent_system_prompt
from backend.agent.state import AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_agent_node(
    llm: ChatOpenAI,
    tools: List[Any],
    max_iterations: int = 5,
):
    """
    Factory function to create agent node with bound tools.

    Args:
        llm: ChatOpenAI instance
        tools: List of tools to bind
        max_iterations: Maximum tool-calling iterations

    Returns:
        Agent node function
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> Dict[str, Any]:
        """
        Main agent decision node.

        Invokes LLM with tools and manages iteration count.
        """
        iteration_count = state.get("iteration_count", 0)

        # Check iteration limit
        if iteration_count >= max_iterations:
            logger.warning(f"Max iterations ({max_iterations}) reached")
            return {
                "messages": [
                    AIMessage(
                        content="I've reached my search limit. Based on what I found, "
                        "here's my best answer based on available information."
                    )
                ],
                "iteration_count": iteration_count,
            }

        # Build messages with system prompt
        system_prompt = get_agent_system_prompt()
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(state["messages"])
        logger.info(f"Messages before invoke LMM with tools: {messages}")

        # Invoke LLM with tools
        try:
            response = llm_with_tools.invoke(messages)
            logger.info(
                f"Agent iteration {iteration_count + 1}: "
                f"tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0}"
            )
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return {
                "messages": [
                    AIMessage(
                        content=f"I encountered an error processing your request: {str(e)}"
                    )
                ],
                "iteration_count": iteration_count + 1,
            }

        return {
            "messages": [response],
            "iteration_count": iteration_count + 1,
        }

    return agent_node
