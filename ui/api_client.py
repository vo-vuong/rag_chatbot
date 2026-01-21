"""Streamlit API client for FastAPI backend."""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import httpx
import streamlit as st

from config.constants import API_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class UIChunk:
    """Retrieved text chunk for UI display."""

    text: str
    score: float
    source_file: str
    page_number: Optional[int] = None
    element_type: str = "text"
    point_id: Optional[int] = None


@dataclass
class UIImage:
    """Retrieved image for UI display."""

    caption: str
    image_path: str
    score: float
    source_file: str
    page_number: Optional[int] = None


@dataclass
class APIResponse:
    """Response from chat API."""

    response: str
    route: Optional[str]
    route_reasoning: Optional[str]
    retrieved_chunks: List[UIChunk]  # Typed list (was tuples)
    images: List[UIImage]  # Single list (was 3 parallel lists)


@dataclass
class StreamEvent:
    """SSE event from streaming API."""

    event: str  # "route", "context", "token", "done", "error"
    data: dict


@dataclass
class PreviewChunk:
    """Chunk data for preview display with metadata."""

    text: str
    source_file: str
    page_number: Optional[int]
    element_type: str
    chunk_index: int
    file_type: str
    # Extended metadata fields
    headings: List[str] = field(default_factory=list)
    source: str = "docling"
    bbox: Optional[Dict[str, float]] = None
    chunk_type: str = "hybrid"
    token_count: Optional[int] = None
    processing_strategy: Optional[str] = None
    ocr_used: bool = False


@dataclass
class PreviewImage:
    """Image data for preview display with metadata."""

    caption: str
    image_path: str
    page_number: Optional[int]
    source_file: str
    image_hash: str
    # Extended metadata fields
    image_metadata: Dict[str, Any] = field(default_factory=dict)
    bbox: Optional[Dict[str, float]] = None
    docling_caption: Optional[str] = None
    surrounding_context: Optional[str] = None
    caption_cost: float = 0.0
    file_type: str = ""
    language: str = "en"
    processing_strategy: str = "docling"


@dataclass
class PreviewResult:
    """Preview result from API - contains preview and full data for save."""

    success: bool
    file_name: str = ""
    file_type: str = ""
    preview_chunks: List[PreviewChunk] = field(default_factory=list)
    preview_images: List[PreviewImage] = field(default_factory=list)
    total_chunks_count: int = 0
    total_images_count: int = 0
    processing_time: float = 0.0
    full_chunks_data: List[Dict[str, Any]] = field(default_factory=list)
    full_images_data: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    error: Optional[str] = None


@dataclass
class SaveResult:
    """Save result from API."""

    success: bool
    chunks_count: int = 0
    images_count: int = 0
    text_collection: str = ""
    image_collection: str = ""
    error: Optional[str] = None


class StreamlitAPIClient:
    """HTTP client for FastAPI backend."""

    def __init__(self, base_url: str = API_BASE_URL):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)
        self._session_id = self._get_or_create_session_id()

    def _get_or_create_session_id(self) -> str:
        """Get session ID from st.session_state or create new one."""
        if "api_session_id" not in st.session_state:
            st.session_state.api_session_id = str(uuid.uuid4())
        return st.session_state.api_session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self._client.get(f"{self._base_url}/api/v1/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            return False

    def chat(
        self,
        query: str,
        mode: str = "rag",
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> APIResponse:
        """Send chat query to API (synchronous)."""
        response = self._client.post(
            f"{self._base_url}/api/v1/chat/query",
            json={
                "query": query,
                "session_id": self._session_id,
                "mode": mode,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        )
        response.raise_for_status()
        data = response.json()

        return APIResponse(
            response=data["response"],
            route=data.get("route"),
            route_reasoning=data.get("route_reasoning"),
            retrieved_chunks=[
                UIChunk(
                    text=c["text"],
                    score=c["score"],
                    source_file=c.get("source_file", "Unknown"),
                    page_number=c.get("page_number"),
                    element_type=c.get("element_type", "text"),
                    point_id=c.get("point_id"),
                )
                for c in data.get("retrieved_chunks", [])
            ],
            images=[
                UIImage(
                    caption=img["caption"],
                    image_path=img["image_path"],
                    score=img["score"],
                    source_file=img.get("source_file", "Unknown"),
                    page_number=img.get("page_number"),
                )
                for img in data.get("images", [])
            ],
        )

    def chat_stream(
        self,
        query: str,
        mode: str = "rag",
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> Iterator[StreamEvent]:
        """Send chat query and stream response (SSE)."""
        with self._client.stream(
            "POST",
            f"{self._base_url}/api/v1/chat/query/stream",
            json={
                "query": query,
                "session_id": self._session_id,
                "mode": mode,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                        yield StreamEvent(
                            event=payload.get("event", "unknown"),
                            data=payload.get("data", {}),
                        )
                    except json.JSONDecodeError:
                        continue

    def search(
        self,
        query: str,
        collection_type: str = "text",
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> dict:
        """Search RAG collections without generation."""
        response = self._client.post(
            f"{self._base_url}/api/v1/rag/search",
            json={
                "query": query,
                "collection_type": collection_type,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close HTTP client."""
        self._client.close()

    def preview_upload(
        self,
        file_content: bytes,
        file_name: str,
        language: str = "en",
        processing_mode: str = "fast",
        csv_columns: Optional[str] = None,
        vision_failure_mode: str = "graceful",
        timeout: float = 300.0,
    ) -> PreviewResult:
        """
        Upload file for preview processing (no save to Qdrant).

        Args:
            file_content: Raw file bytes
            file_name: Original filename
            language: Document language (en/vi)
            processing_mode: fast or ocr
            csv_columns: CSV column names (comma-separated)
            vision_failure_mode: graceful/strict/skip
            timeout: Request timeout in seconds

        Returns:
            PreviewResult with preview data and full data for later save
        """
        try:
            files = {"file": (file_name, file_content)}
            data = {
                "language": language,
                "processing_mode": processing_mode,
                "vision_failure_mode": vision_failure_mode,
            }
            if csv_columns:
                data["csv_columns"] = csv_columns

            response = self._client.post(
                f"{self._base_url}/api/v1/upload/preview",
                files=files,
                data=data,
                timeout=timeout,
            )

            if response.status_code == 200:
                result = response.json()
                # Parse preview chunks with metadata
                preview_chunks = [
                    PreviewChunk(
                        text=c["text"],
                        source_file=c["source_file"],
                        page_number=c.get("page_number"),
                        element_type=c.get("element_type", "text"),
                        chunk_index=c["chunk_index"],
                        file_type=c["file_type"],
                        # Extended metadata
                        headings=c.get("headings", []),
                        source=c.get("source", "docling"),
                        bbox=c.get("bbox"),
                        chunk_type=c.get("chunk_type", "hybrid"),
                        token_count=c.get("token_count"),
                        processing_strategy=c.get("processing_strategy"),
                        ocr_used=c.get("ocr_used", False),
                    )
                    for c in result.get("chunks", [])
                ]
                # Parse preview images with metadata
                preview_images = [
                    PreviewImage(
                        caption=img["caption"],
                        image_path=img["image_path"],
                        page_number=img.get("page_number"),
                        source_file=img["source_file"],
                        image_hash=img["image_hash"],
                        # Extended metadata
                        image_metadata=img.get("image_metadata", {}),
                        bbox=img.get("bbox"),
                        docling_caption=img.get("docling_caption"),
                        surrounding_context=img.get("surrounding_context"),
                        caption_cost=img.get("caption_cost", 0.0),
                        file_type=img.get("file_type", ""),
                        language=img.get("language", "en"),
                        processing_strategy=img.get("processing_strategy", "docling"),
                    )
                    for img in result.get("images", [])
                ]

                return PreviewResult(
                    success=True,
                    file_name=result.get("file_name", file_name),
                    file_type=result.get("file_type", ""),
                    preview_chunks=preview_chunks,
                    preview_images=preview_images,
                    total_chunks_count=result.get("total_chunks_count", 0),
                    total_images_count=result.get("total_images_count", 0),
                    processing_time=result.get("processing_time_seconds", 0.0),
                    full_chunks_data=result.get("full_chunks_data", []),
                    full_images_data=result.get("full_images_data", []),
                    language=language,
                )
            else:
                try:
                    error = response.json().get("detail", "Unknown error")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}"
                return PreviewResult(
                    success=False,
                    file_name=file_name,
                    error=error,
                )

        except httpx.TimeoutException:
            return PreviewResult(
                success=False,
                file_name=file_name,
                error="Request timeout. Processing took too long.",
            )
        except httpx.RequestError as e:
            return PreviewResult(
                success=False,
                file_name=file_name,
                error=f"Connection error: {e}",
            )

    def save_upload(
        self,
        file_name: str,
        file_type: str,
        language: str,
        chunks_data: List[Dict[str, Any]],
        images_data: List[Dict[str, Any]] = None,
        timeout: float = 300.0,
    ) -> SaveResult:
        """
        Save processed chunks and images to Qdrant.

        Args:
            file_name: Original filename
            file_type: File type (pdf, docx, csv)
            language: Document language
            chunks_data: List of chunk data dicts from preview
            images_data: List of image data dicts from preview
            timeout: Request timeout in seconds

        Returns:
            SaveResult with save status
        """
        try:
            # Build request payload
            payload = {
                "file_name": file_name,
                "file_type": file_type,
                "language": language,
                "chunks": chunks_data,
                "images": images_data or [],
            }

            response = self._client.post(
                f"{self._base_url}/api/v1/upload/save",
                json=payload,
                timeout=timeout,
            )

            if response.status_code == 200:
                result = response.json()
                return SaveResult(
                    success=True,
                    chunks_count=result.get("chunks_count", 0),
                    images_count=result.get("images_count", 0),
                    text_collection=result.get("text_collection", ""),
                    image_collection=result.get("image_collection", ""),
                )
            else:
                try:
                    error = response.json().get("detail", "Unknown error")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}"
                return SaveResult(
                    success=False,
                    error=error,
                )

        except httpx.TimeoutException:
            return SaveResult(
                success=False,
                error="Request timeout. Save took too long.",
            )
        except httpx.RequestError as e:
            return SaveResult(
                success=False,
                error=f"Connection error: {e}",
            )


def get_api_client() -> StreamlitAPIClient:
    """Get or create API client singleton."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = StreamlitAPIClient()
    return st.session_state.api_client
