"""Chat endpoints."""

import json
import logging
from datetime import datetime
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import get_chat_service
from api.models.requests import ChatRequest
from api.models.responses import ChatResponse, ImageResult, RetrievedChunk
from api.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """Process chat query (synchronous)."""
    try:
        result = chat_service.process_query(
            query=request.query,
            session_id=request.session_id,
            mode=request.mode,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )

        return ChatResponse(
            response=result.response,
            route=result.route,
            route_reasoning=result.route_reasoning,
            retrieved_chunks=[
                RetrievedChunk(text=t, score=s, source_file=src)
                for t, s, src in result.retrieved_chunks
            ],
            images=[
                ImageResult(
                    caption=c,
                    image_path=p,
                    score=0.0,
                    page_number=None,
                    source_file=src,
                )
                for p, c, src in zip(
                    result.image_paths,
                    result.image_captions,
                    result.image_source_files or ["Unknown"] * len(result.image_paths),
                )
            ],
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Chat query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def chat_query_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """Process chat query with streaming response (SSE)."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            result = chat_service.process_query(
                query=request.query,
                session_id=request.session_id,
                mode=request.mode,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
            )

            # Emit route info
            yield f"data: {json.dumps({'event': 'route', 'data': {'route': result.route, 'reasoning': result.route_reasoning}})}\n\n"

            # Emit context (retrieved chunks)
            if result.retrieved_chunks:
                chunks_data = [
                    {"text": t[:100], "score": s, "source_file": src}
                    for t, s, src in result.retrieved_chunks
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
            yield f"data: {json.dumps({'event': 'done', 'data': {'timestamp': datetime.utcnow().isoformat()}})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
