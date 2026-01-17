"""
Chat Service - Main chat orchestration for API layer.

Extracted from ChatMainUI to enable API-based chat operations.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from backend.llms.llm_strategy import LLMStrategy
from backend.models import ChunkElement, ImageElement
from backend.prompts.prompt_builder import PromptBuilder
from api.services.rag_service import RAGService
from api.services.session_service import SessionService

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Chat response container."""

    response: str
    route: Optional[str]  # "text_only", "image_only", or None (llm_only)
    route_reasoning: Optional[str]
    retrieved_chunks: List[ChunkElement]  # Typed list of chunks
    images: List[ImageElement]  # Single list of image objects (not 3 parallel lists)


class ChatService:
    """
    Chat orchestration service - extracted from ChatMainUI.

    Handles chat processing independent of Streamlit.
    """

    def __init__(
        self,
        llm: LLMStrategy,
        rag_service: RAGService,
        session_service: SessionService,
    ):
        self._llm = llm
        self._rag = rag_service
        self._sessions = session_service

    def process_query(
        self,
        query: str,
        session_id: str,
        mode: str = "rag",
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> ChatResponse:
        """
        Process chat query. Main entry point.

        Args:
            query: User's question
            session_id: Session identifier
            mode: "rag" or "llm_only"
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
        """
        # Add user message to history
        self._sessions.add_message(session_id, "user", query)

        if mode == "llm_only":
            response = self._generate_llm_only(query, session_id)
            result = ChatResponse(
                response=response,
                route=None,
                route_reasoning=None,
                retrieved_chunks=[],
                images=[],
            )
        else:
            result = self._generate_rag(query, session_id, top_k, score_threshold)

        # Add assistant response to history
        self._sessions.add_message(session_id, "assistant", result.response)

        return result

    def _generate_rag(
        self,
        query: str,
        session_id: str,
        top_k: int,
        score_threshold: float,
    ) -> ChatResponse:
        """Generate RAG response with routing."""
        try:
            route, reasoning = self._rag.route_query(query)
            logger.info(f"Query routed to: {route} (reason: {reasoning})")

            if route == "text_only":
                return self._generate_text_response(
                    query, session_id, top_k, score_threshold, reasoning
                )
            else:
                return self._generate_image_response(query, session_id, reasoning)
        except Exception as e:
            logger.error(f"Router error, falling back to text: {e}")
            return self._generate_text_response(
                query, session_id, top_k, score_threshold, "Fallback due to router error"
            )

    def _generate_text_response(
        self,
        query: str,
        session_id: str,
        top_k: int,
        score_threshold: float,
        reasoning: str,
    ) -> ChatResponse:
        """Generate response from text collection."""
        chunks = self._rag.search_text(query, top_k, score_threshold)

        if not chunks:
            response = self._generate_llm_only(query, session_id)
            return ChatResponse(
                response=response + "\n\n*Note: No relevant documents found.*",
                route="text_only",
                route_reasoning=reasoning,
                retrieved_chunks=[],
                images=[],
            )

        # Build context from typed chunks
        context = PromptBuilder.format_context(
            [c.content for c in chunks],
            [c.score for c in chunks],
            include_scores=False,
        )

        # Get chat history
        history = self._sessions.get_chat_history(session_id)
        history_str = PromptBuilder.format_chat_history(history) if history else ""

        # Build prompt and generate
        prompt = PromptBuilder.build_rag_prompt(
            query=query,
            context=context,
            chat_history=history_str,
        )
        response = self._llm.generate_content(prompt=prompt)
        response += f"\n\n---\n*📚 Answer based on {len(chunks)} text document(s)*"

        return ChatResponse(
            response=response,
            route="text_only",
            route_reasoning=reasoning,
            retrieved_chunks=chunks,
            images=[],
        )

    def _generate_image_response(
        self,
        query: str,
        session_id: str,
        reasoning: str,
    ) -> ChatResponse:
        """Generate response from image collection."""
        images = self._rag.search_images(query, top_k=1, score_threshold=0.6)

        if not images:
            return ChatResponse(
                response="❌ No relevant images found. Try rephrasing or ask about text content.",
                route="image_only",
                route_reasoning=reasoning,
                retrieved_chunks=[],
                images=[],
            )

        img = images[0]
        context = PromptBuilder.format_image_context(
            image_caption=img.content,  # Was img.caption
            page_number=img.page_number,
            source_document=img.source_file,  # Was img.source_document
            score=img.score,
        )

        history = self._sessions.get_chat_history(session_id)
        prompt = PromptBuilder.build_rag_prompt_with_images(
            query=query,
            text_context="",
            image_context=context,
            chat_history=history if history else None,
        )
        response = self._llm.generate_content(prompt=prompt)
        response += "\n\n---\n*🖼️ Answer based on 1 image*"

        return ChatResponse(
            response=response,
            route="image_only",
            route_reasoning=reasoning,
            retrieved_chunks=[],
            images=images,  # Pass typed list directly
        )

    def _generate_llm_only(self, query: str, session_id: str) -> str:
        """Generate LLM-only response."""
        history = self._sessions.get_chat_history(session_id)
        prompt = PromptBuilder.build_chat_prompt(query=query, chat_history=history)
        return self._llm.generate_content(prompt=prompt)
