"""Upload router - file upload and processing endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_upload_service
from api.models.requests import SaveUploadRequest
from api.models.responses import SaveUploadResponse, UploadPreviewResponse, UploadResponse
from api.services.upload_service import UploadService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF/DOCX/CSV)"),
    language: str = Form(default="en", description="Document language (en/vi)"),
    processing_mode: str = Form(default="fast", description="fast or ocr"),
    csv_columns: Optional[str] = Form(default=None, description="CSV columns"),
    vision_failure_mode: str = Form(
        default="graceful", description="graceful/strict/skip"
    ),
    upload_service: UploadService = Depends(get_upload_service),
) -> UploadResponse:
    """
    Upload and process a document.

    Accepts PDF, DOCX, or CSV files. Processes document, generates embeddings,
    and stores chunks in TEXT_COLLECTION_NAME (text) and IMAGE_COLLECTION_NAME (images).

    Args:
        file: Document file to upload
        language: Document language for OCR (en or vi)
        processing_mode: fast (no OCR) or ocr (with OCR)
        csv_columns: Comma-separated column names for CSV grouping
        vision_failure_mode: How to handle image captioning failures

    Returns:
        UploadResponse with processing results

    Raises:
        HTTPException 400: Invalid file or parameters
        HTTPException 500: Processing error
    """
    try:
        # Validate parameters
        if language not in {"en", "vi"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid language: {language}. Must be 'en' or 'vi'",
            )

        if processing_mode not in {"fast", "ocr"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid processing_mode: {processing_mode}. Must be 'fast' or 'ocr'",
            )

        if vision_failure_mode not in {"graceful", "strict", "skip"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vision_failure_mode: {vision_failure_mode}",
            )

        # Sanitize filename for logging (truncate, repr for safety)
        safe_filename = (file.filename or "unknown")[:100]
        logger.info(f"Upload request: {safe_filename!r}")

        result = upload_service.process_and_store(
            file=file,
            language=language,
            processing_mode=processing_mode,
            csv_columns=csv_columns,
            vision_failure_mode=vision_failure_mode,
        )

        logger.info(f"Upload success: {result.chunks_count} chunks")
        return result

    except ValueError as e:
        logger.warning(f"Upload validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Processing failed. Check server logs for details.",
        )


@router.post("/preview", response_model=UploadPreviewResponse)
async def upload_preview(
    file: UploadFile = File(..., description="Document file (PDF/DOCX/CSV)"),
    language: str = Form(default="en", description="Document language (en/vi)"),
    processing_mode: str = Form(default="fast", description="fast or ocr"),
    csv_columns: Optional[str] = Form(default=None, description="CSV columns"),
    vision_failure_mode: str = Form(
        default="graceful", description="graceful/strict/skip"
    ),
    upload_service: UploadService = Depends(get_upload_service),
) -> UploadPreviewResponse:
    """
    Process document and return preview (no save to Qdrant).

    Processes the document, extracts chunks and images, but does NOT save
    to the vector database. Returns preview chunks (max 50) and full data
    for later save operation.

    Args:
        file: Document file to upload
        language: Document language for OCR (en or vi)
        processing_mode: fast (no OCR) or ocr (with OCR)
        csv_columns: Comma-separated column names for CSV grouping
        vision_failure_mode: How to handle image captioning failures

    Returns:
        UploadPreviewResponse with preview chunks and full data for save

    Raises:
        HTTPException 400: Invalid file or parameters
        HTTPException 500: Processing error
    """
    try:
        # Validate parameters
        if language not in {"en", "vi"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid language: {language}. Must be 'en' or 'vi'",
            )

        if processing_mode not in {"fast", "ocr"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid processing_mode: {processing_mode}. Must be 'fast' or 'ocr'",
            )

        if vision_failure_mode not in {"graceful", "strict", "skip"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vision_failure_mode: {vision_failure_mode}",
            )

        safe_filename = (file.filename or "unknown")[:100]
        logger.info(f"Preview request: {safe_filename!r}")

        result = upload_service.process_preview(
            file=file,
            language=language,
            processing_mode=processing_mode,
            csv_columns=csv_columns,
            vision_failure_mode=vision_failure_mode,
        )

        logger.info(
            f"Preview success: {result.total_chunks_count} chunks, "
            f"{result.total_images_count} images"
        )
        return result

    except ValueError as e:
        logger.warning(f"Preview validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Preview error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Preview processing failed. Check server logs for details.",
        )


@router.post("/save", response_model=SaveUploadResponse)
async def save_upload(
    request: SaveUploadRequest,
    upload_service: UploadService = Depends(get_upload_service),
) -> SaveUploadResponse:
    """
    Save processed chunks and images to Qdrant.

    Takes the chunk and image data from a previous /preview call and saves
    them to the vector database with embeddings.

    Args:
        request: SaveUploadRequest with chunk and image data

    Returns:
        SaveUploadResponse with save results and collection names

    Raises:
        HTTPException 400: Invalid request data
        HTTPException 500: Save error
    """
    try:
        if not request.chunks and not request.images:
            raise HTTPException(
                status_code=400,
                detail="No chunks or images to save",
            )

        logger.info(
            f"Save request: {request.file_name}, "
            f"{len(request.chunks)} chunks, {len(request.images)} images"
        )

        result = upload_service.save_chunks(request)

        logger.info(
            f"Save success: {result.chunks_count} chunks, {result.images_count} images"
        )
        return result

    except ValueError as e:
        logger.warning(f"Save validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Save error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Save failed. Check server logs for details.",
        )
