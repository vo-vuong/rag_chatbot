"""Health check endpoint."""

from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from api.config import get_settings, Settings
from api.models.responses import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """Check API and Qdrant health."""
    qdrant_connected = False
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        qdrant_connected = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        version=settings.api_version,
        qdrant_connected=qdrant_connected,
    )
