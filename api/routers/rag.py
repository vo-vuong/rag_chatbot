"""RAG search endpoints."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_rag_service
from api.models.requests import RAGSearchRequest
from api.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGSearchResponse(BaseModel):
    """RAG search response model."""

    route: str
    reasoning: str
    results: List[dict]


@router.post("/search", response_model=RAGSearchResponse)
async def rag_search(
    request: RAGSearchRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """Search RAG collections without LLM generation."""
    try:
        route, reasoning = rag_service.route_query(request.query)

        if request.collection_type == "text" or route == "text_only":
            results = rag_service.search_text(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
            )
            return RAGSearchResponse(
                route="text_only",
                reasoning=reasoning,
                results=[{"text": t, "score": s} for t, s in results],
            )
        else:
            results = rag_service.search_images(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
            )
            return RAGSearchResponse(
                route="image_only",
                reasoning=reasoning,
                results=[
                    {
                        "caption": r.caption,
                        "image_path": r.image_path,
                        "score": r.score,
                        "page_number": r.page_number,
                    }
                    for r in results
                ],
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
