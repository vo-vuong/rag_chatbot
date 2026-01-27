"""Chat endpoints - LangGraph Agent-based implementation."""

import json
import logging
from datetime import datetime
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import get_agent_service
from api.models.requests import ChatRequest
from api.models.responses import ChatResponse, ImageResult, RetrievedChunk
from backend.agent.service import AgentService
from backend.models import ChunkElement, ImageElement

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    """Process chat query through LangGraph agent."""
    try:
        result = agent_service.process_query(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )

        # Convert retrieved_chunks from dict to ChunkElement for API response
        chunks = []
        for chunk_dict in result.retrieved_chunks:
            chunk = ChunkElement(
                content=chunk_dict.get("content", ""),
                score=chunk_dict.get("score", 0.0),
                source_file=chunk_dict.get("source_file", "Unknown"),
                page_number=chunk_dict.get("page_number"),
                element_type=chunk_dict.get("element_type", "text"),
                point_id=chunk_dict.get("point_id"),
            )
            chunks.append(RetrievedChunk.from_chunk_element(chunk))

        # Convert images from dict to ImageElement for API response
        images = []
        for img_dict in result.images:
            img = ImageElement(
                content=img_dict.get("content", ""),
                image_path=img_dict.get("image_path", ""),
                score=img_dict.get("score", 0.0),
                source_file=img_dict.get("source_file", "Unknown"),
                page_number=img_dict.get("page_number"),
            )
            images.append(ImageResult.from_image_element(img))

        return ChatResponse(
            response=result.response,
            route=result.route,
            route_reasoning=result.route_reasoning,
            retrieved_chunks=chunks,
            images=images,
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Chat query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def chat_query_stream(
    request: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    """Process chat query with streaming response (SSE)."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            # Get full result first (streaming agent output is complex)
            result = agent_service.process_query(
                query=request.query,
                session_id=request.session_id,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
            )

            # Emit route info
            yield f"data: {json.dumps({'event': 'route', 'data': {'route': result.route, 'reasoning': result.route_reasoning}})}\n\n"

            # Emit context (retrieved chunks)
            if result.retrieved_chunks:
                chunks_data = [
                    {
                        "text": chunk.get("content", "")[:100],
                        "score": chunk.get("score", 0.0),
                        "source_file": chunk.get("source_file", "Unknown"),
                    }
                    for chunk in result.retrieved_chunks
                ]
                yield f"data: {json.dumps({'event': 'context', 'data': chunks_data})}\n\n"

            # Emit response (simulated streaming - split by words)
            words = result.response.split()
            buffer = ""
            for word in words:
                buffer += word + " "
                if len(buffer) > 20:
                    yield f"data: {json.dumps({'event': 'token', 'data': buffer})}\n\n"
                    buffer = ""
            if buffer:
                yield f"data: {json.dumps({'event': 'token', 'data': buffer})}\n\n"

            # Emit done
            yield f"data: {json.dumps({'event': 'done', 'data': {'timestamp': datetime.utcnow().isoformat(), 'iterations': result.iterations}})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    agent_service: AgentService = Depends(get_agent_service),
):
    """Get conversation history for a session."""
    try:
        history = agent_service.get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
