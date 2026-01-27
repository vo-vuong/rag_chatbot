"""
Agent Service - Main entry point for API layer integration.

Replaces ChatService with LangGraph-based orchestration.
"""

import logging
from typing import Iterator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from backend.agent.config import AgentConfig, WorkflowType
from backend.agent.graphs import create_graph
from backend.agent.state import AgentResponse
from backend.agent.tools.base import deserialize_chunk, deserialize_image
from backend.models import ChunkElement, ImageElement

logger = logging.getLogger(__name__)


class AgentService:
    """
    LangGraph-based agent service for RAG chat.

    Drop-in replacement for ChatService with enhanced capabilities.
    """

    def __init__(
        self,
        rag_service,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        workflow_type: WorkflowType = WorkflowType.AGENTIC_RAG,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize AgentService.

        Args:
            rag_service: Configured RAGService instance
            openai_api_key: OpenAI API key
            model: LLM model name
            workflow_type: Type of workflow to use
            config: Optional AgentConfig for customization
        """
        self._rag_service = rag_service
        self._openai_api_key = openai_api_key
        self._model = model
        self._workflow_type = workflow_type
        self._config = config or AgentConfig.from_env()

        # Checkpointer for conversation persistence (can be disabled for testing)
        if self._config.enable_checkpointing:
            self._checkpointer = MemorySaver()
            logger.info("Memory checkpointing enabled")
        else:
            self._checkpointer = None
            logger.info("Memory checkpointing disabled (stateless mode)")

        # Create graph
        self._graph = create_graph(
            workflow_type=workflow_type,
            rag_service=rag_service,
            openai_api_key=openai_api_key,
            model=model,
            checkpointer=self._checkpointer,
            max_iterations=self._config.max_iterations,
        )

        logger.info(f"AgentService initialized with workflow: {workflow_type.value}")

    def process_query(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> AgentResponse:
        """
        Process user query through agent graph.

        Args:
            query: User's question
            session_id: Session identifier (used as thread_id)
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score

        Returns:
            AgentResponse with answer and metadata
        """
        logger.info(f"Processing query: session={session_id}, query='{query[:50]}...'")

        # Prepare input state
        input_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "session_id": session_id,
            "iteration_count": 0,
            "retrieval_result": None,
            "route": None,
        }

        # Configure with thread_id for checkpointing
        config = {"configurable": {"thread_id": session_id}}

        # Invoke graph
        try:
            result = self._graph.invoke(input_state, config)
        except Exception as e:
            logger.error(f"Graph invocation failed: {e}")
            return AgentResponse(
                response=f"I encountered an error: {str(e)}",
                route=None,
                route_reasoning=None,
                retrieved_chunks=[],
                images=[],
                iterations=1,
            )

        # Parse result
        return self._parse_result(result)

    def process_query_stream(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> Iterator[str]:
        """
        Stream response through agent graph.

        Yields partial response chunks for real-time display.
        """
        logger.info(f"Streaming query: session={session_id}")

        input_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "session_id": session_id,
            "iteration_count": 0,
            "retrieval_result": None,
            "route": None,
        }

        config = {"configurable": {"thread_id": session_id}}

        # Stream events from graph
        try:
            for event in self._graph.stream(input_state, config, stream_mode="values"):
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        # Only yield if it's not a tool call message
                        if not (
                            hasattr(last_msg, "tool_calls") and last_msg.tool_calls
                        ):
                            yield last_msg.content
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {str(e)}"

    def _parse_result(self, result: dict) -> AgentResponse:
        """Parse graph result into AgentResponse."""
        messages = result.get("messages", [])

        # Get final AI response (non-tool-call message)
        response_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                # Skip messages that are just tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls and not msg.content:
                    continue
                response_text = msg.content
                break

        # Extract retrieval results from tool messages
        chunks: List[ChunkElement] = []
        images: List[ImageElement] = []
        route = None
        reasoning = None

        # Look for tool results in messages
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                try:
                    # Tool message content contains the result
                    import json

                    if isinstance(msg.content, str):
                        tool_result = json.loads(msg.content)
                    else:
                        tool_result = msg.content

                    if isinstance(tool_result, dict):
                        if "chunks" in tool_result:
                            for c in tool_result["chunks"]:
                                chunks.append(deserialize_chunk(c))
                        if "images" in tool_result:
                            for img in tool_result["images"]:
                                images.append(deserialize_image(img))
                        if "route" in tool_result:
                            route = tool_result["route"]
                        if "reasoning" in tool_result:
                            reasoning = tool_result["reasoning"]
                except (json.JSONDecodeError, TypeError):
                    pass

        return AgentResponse(
            response=response_text,
            route=route,
            route_reasoning=reasoning,
            retrieved_chunks=[c.__dict__ for c in chunks] if chunks else [],
            images=[img.__dict__ for img in images] if images else [],
            iterations=result.get("iteration_count", 1),
        )

    def get_conversation_history(self, session_id: str) -> List[dict]:
        """Retrieve conversation history from checkpointer."""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self._graph.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                return [
                    {"role": getattr(msg, "type", "unknown"), "content": msg.content}
                    for msg in messages
                    if hasattr(msg, "content") and msg.content
                ]
        except Exception as e:
            logger.warning(f"Failed to get conversation history: {e}")
        return []

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for session."""
        # With MemorySaver, we can't easily delete individual threads
        # For production, use a checkpointer that supports deletion
        logger.info(
            f"Session {session_id} clear requested (not implemented for MemorySaver)"
        )
