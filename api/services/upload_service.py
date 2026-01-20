"""Upload service - orchestrates file upload and processing."""

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import UploadFile

from api.config import Settings
from api.models.requests import SaveUploadRequest
from api.models.responses import (
    FullChunkData,
    FullImageData,
    PreviewChunk,
    PreviewImage,
    SaveUploadResponse,
    UploadPreviewResponse,
    UploadResponse,
)
from backend.document_processor import DocumentProcessor
from backend.embeddings.openai_embeddings import OpenAIEmbeddingStrategy
from backend.vector_db.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)

# Preview chunk limit for UI display
PREVIEW_CHUNK_LIMIT = 50


class UploadService:
    """Service for processing and storing uploaded documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv"}

    def __init__(
        self,
        settings: Settings,
        document_processor: DocumentProcessor,
        embedding: OpenAIEmbeddingStrategy,
    ):
        self.settings = settings
        self.document_processor = document_processor
        self.embedding = embedding

    def process_and_store(
        self,
        file: UploadFile,
        language: str = "en",
        processing_mode: str = "fast",
        csv_columns: Optional[str] = None,
        vision_failure_mode: str = "graceful",
    ) -> UploadResponse:
        """Process uploaded file and store in Qdrant.

        Text chunks are stored in settings.text_collection.
        Images are stored in settings.image_collection.
        """
        start_time = time.time()
        temp_path = None

        try:
            self._validate_file(file)
            temp_path = self._save_to_temp(file)

            # Process document
            result = self.document_processor.process_document(
                file_path=temp_path,
                languages=[language],
                original_filename=file.filename,
                openai_api_key=self.settings.openai_api_key,
                caption_failure_mode=vision_failure_mode,
            )

            if not result.success:
                raise ValueError(f"Processing failed: {result.error_message}")

            # Use settings-defined collection names
            qdrant = QdrantManager(
                collection_name=self.settings.text_collection,
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )

            chunks_count = 0
            images_count = 0
            file_type = Path(file.filename).suffix.lstrip(".")

            # Process text chunks
            if result.elements:
                # Extract processing metrics for metadata
                processing_strategy = (
                    result.metrics.strategy_used
                    if result.metrics and result.metrics.strategy_used
                    else "docling"
                )
                ocr_used = (
                    result.metrics.ocr_used
                    if result.metrics
                    else False
                )

                chunks_count = self._upload_text_chunks(
                    qdrant=qdrant,
                    elements=result.elements,
                    source_file=file.filename,
                    file_type=file_type,
                    language=language,
                    processing_strategy=processing_strategy,
                    ocr_used=ocr_used,
                )

            # Process images - use settings.image_collection
            if result.image_data:
                images_count = self._upload_images(
                    image_data=result.image_data,
                    source_file=file.filename,
                    file_type=file_type,
                    language=language,
                    processing_strategy=processing_strategy,
                )

            processing_time = time.time() - start_time

            return UploadResponse(
                status="success",
                file_name=file.filename,
                file_type=file_type,
                chunks_count=chunks_count,
                images_count=images_count,
                message=f"Successfully processed. Text: '{self.settings.text_collection}', Images: '{self.settings.image_collection}'",
                processing_time_seconds=round(processing_time, 2),
            )

        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
            raise

        finally:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except Exception as cleanup_err:
                    logger.warning(f"Temp cleanup failed: {cleanup_err}")

    def _upload_text_chunks(
        self,
        qdrant: QdrantManager,
        elements: list,
        source_file: str,
        file_type: str,
        language: str,
        processing_strategy: str = "docling",
        ocr_used: bool = False,
    ) -> int:
        """Upload text chunks to Qdrant collection.

        Extracts and flattens all metadata from chunk elements for full traceability.
        """
        # Build DataFrame with chunk data including all metadata
        chunks_data = []
        for idx, elem in enumerate(elements):
            text = elem.text if hasattr(elem, "text") else str(elem)

            # Extract metadata dict from element
            metadata = {}
            if hasattr(elem, "metadata"):
                if isinstance(elem.metadata, dict):
                    metadata = elem.metadata
                elif hasattr(elem.metadata, "get"):
                    metadata = dict(elem.metadata)

            # Extract individual fields from metadata
            page_num = metadata.get("page_number")
            elem_type = metadata.get("element_type", "text")
            headings = metadata.get("headings", [])
            source = metadata.get("source", "docling")
            bbox = metadata.get("bbox")
            chunk_type = metadata.get("chunk_type", "hybrid")
            token_count = metadata.get("token_count")

            chunks_data.append(
                {
                    "chunk": text,
                    "page_number": page_num,
                    "element_type": elem_type,
                    "chunk_index": idx,
                    "file_type": file_type,
                    "language": language,
                    # Extended metadata fields (flattened)
                    "headings": headings,
                    "source": source,
                    "bbox": bbox,
                    "chunk_type": chunk_type,
                    "token_count": token_count,
                    "processing_strategy": processing_strategy,
                    "ocr_used": ocr_used,
                }
            )

        chunks_df = pd.DataFrame(chunks_data)

        # Generate embeddings
        texts = [row["chunk"] for row in chunks_data]
        embeddings = self.embedding.embed_texts(texts)

        # Ensure collection exists
        vector_dim = len(embeddings[0])
        qdrant.ensure_collection(dimension=vector_dim)

        # Upload to Qdrant
        success = qdrant.add_documents(
            chunks_df=chunks_df,
            embeddings=embeddings,
            source_file=source_file,
            language=language,
        )

        if not success:
            raise ValueError("Failed to upload text chunks to Qdrant")

        logger.info(f"Uploaded {len(chunks_df)} text chunks")
        return len(chunks_df)

    def _upload_images(
        self,
        image_data: list,
        source_file: str,
        file_type: str = "",
        language: str = "en",
        processing_strategy: str = "docling",
    ) -> int:
        """Upload image captions to settings.image_collection.

        Includes all metadata fields from image extraction for full traceability.
        """
        try:
            image_qdrant = QdrantManager(
                collection_name=self.settings.image_collection,
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )

            # Build image DataFrame with all metadata
            images_df = pd.DataFrame(
                [
                    {
                        "chunk": img["caption"],
                        "image_path": img["image_path"],
                        "image_hash": img["image_hash"],
                        "page_number": img["page_number"],
                        "source_file": img.get("source_file", source_file),
                        # Image dimensions from image_metadata
                        "width": img["image_metadata"]["width"],
                        "height": img["image_metadata"]["height"],
                        "format": img["image_metadata"]["format"],
                        "optimized_size_bytes": img["image_metadata"].get(
                            "optimized_size_bytes"
                        ),
                        # Extended metadata fields
                        "bbox": img.get("bbox"),
                        "docling_caption": img.get("docling_caption"),
                        "surrounding_context": img.get("surrounding_context"),
                        "headings": img.get("headings", []),
                        "caption_cost": img.get("cost", 0.0),
                        "file_type": file_type,
                        "language": language,
                        "processing_strategy": processing_strategy,
                    }
                    for img in image_data
                ]
            )

            # Embed captions
            captions = [img["caption"] for img in image_data]
            embeddings = self.embedding.embed_texts(captions)

            # Ensure collection exists
            vector_dim = len(embeddings[0])
            image_qdrant.ensure_collection(dimension=vector_dim)

            # Upload
            success = image_qdrant.add_documents(
                chunks_df=images_df,
                embeddings=embeddings,
                source_file=source_file,
            )

            if success:
                logger.info(f"Uploaded {len(images_df)} images to '{self.settings.image_collection}'")
                return len(images_df)

        except Exception as e:
            logger.warning(f"Image upload failed (continuing): {e}")

        return 0

    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file."""
        if not file.filename:
            raise ValueError("No filename provided")

        ext = Path(file.filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

    def _save_to_temp(self, file: UploadFile) -> str:
        """Save uploaded file to temp directory with size validation."""
        suffix = Path(file.filename).suffix
        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Read in chunks to validate size without loading entire file
            total_size = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_bytes:
                    # Clean up partial file
                    temp_file.close()
                    Path(temp_file.name).unlink()
                    raise ValueError(
                        f"File exceeds max size: {self.settings.max_upload_size_mb}MB"
                    )
                temp_file.write(chunk)

            return temp_file.name

    def process_preview(
        self,
        file: UploadFile,
        language: str = "en",
        processing_mode: str = "fast",
        csv_columns: Optional[str] = None,
        vision_failure_mode: str = "graceful",
    ) -> UploadPreviewResponse:
        """Process file and return preview data without saving to Qdrant.

        Args:
            file: Uploaded file
            language: Document language (en/vi)
            processing_mode: fast or ocr
            csv_columns: CSV column names for grouping
            vision_failure_mode: How to handle image captioning failures

        Returns:
            UploadPreviewResponse with chunks for preview and full data for save
        """
        start_time = time.time()
        temp_path = None

        try:
            self._validate_file(file)
            temp_path = self._save_to_temp(file)
            file_type = Path(file.filename).suffix.lstrip(".")

            # Process document
            result = self.document_processor.process_document(
                file_path=temp_path,
                languages=[language],
                original_filename=file.filename,
                openai_api_key=self.settings.openai_api_key,
                caption_failure_mode=vision_failure_mode,
            )

            if not result.success:
                raise ValueError(f"Processing failed: {result.error_message}")

            # Extract processing metrics for metadata
            processing_strategy = (
                result.metrics.strategy_used
                if result.metrics and result.metrics.strategy_used
                else "docling"
            )
            ocr_used = (
                result.metrics.ocr_used
                if result.metrics
                else False
            )

            # Build full chunks data (all chunks) with all metadata
            full_chunks_data = []
            for idx, elem in enumerate(result.elements):
                text = elem.text if hasattr(elem, "text") else str(elem)

                # Extract metadata dict from element
                metadata = {}
                if hasattr(elem, "metadata"):
                    if isinstance(elem.metadata, dict):
                        metadata = elem.metadata
                    elif hasattr(elem.metadata, "get"):
                        metadata = dict(elem.metadata)

                # Extract individual fields from metadata
                page_num = metadata.get("page_number")
                elem_type = metadata.get("element_type", "text")
                headings = metadata.get("headings", [])
                source = metadata.get("source", "docling")
                bbox = metadata.get("bbox")
                chunk_type = metadata.get("chunk_type", "hybrid")
                token_count = metadata.get("token_count")

                full_chunks_data.append(
                    FullChunkData(
                        text=text,
                        source_file=file.filename,
                        page_number=page_num,
                        element_type=elem_type,
                        chunk_index=idx,
                        file_type=file_type,
                        language=language,
                        # Extended metadata fields
                        headings=headings,
                        source=source,
                        bbox=bbox,
                        chunk_type=chunk_type,
                        token_count=token_count,
                        processing_strategy=processing_strategy,
                        ocr_used=ocr_used,
                    )
                )

            # Build preview chunks (max PREVIEW_CHUNK_LIMIT) with metadata
            preview_chunks = [
                PreviewChunk(
                    text=chunk.text,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    element_type=chunk.element_type,
                    chunk_index=chunk.chunk_index,
                    file_type=chunk.file_type,
                    # Extended metadata for preview display
                    headings=chunk.headings,
                    source=chunk.source,
                    bbox=chunk.bbox,
                    chunk_type=chunk.chunk_type,
                    token_count=chunk.token_count,
                    processing_strategy=chunk.processing_strategy,
                    ocr_used=chunk.ocr_used,
                )
                for chunk in full_chunks_data[:PREVIEW_CHUNK_LIMIT]
            ]

            # Build full images data with all metadata
            full_images_data = []
            preview_images = []

            if result.image_data:
                for img in result.image_data:
                    full_img = FullImageData(
                        caption=img["caption"],
                        image_path=img["image_path"],
                        page_number=img.get("page_number"),
                        source_file=img.get("source_file", file.filename),
                        image_hash=img["image_hash"],
                        image_metadata=img["image_metadata"],
                        # Extended metadata fields
                        bbox=img.get("bbox"),
                        docling_caption=img.get("docling_caption"),
                        surrounding_context=img.get("surrounding_context"),
                        headings=img.get("headings", []),
                        caption_cost=img.get("cost", 0.0),
                        file_type=file_type,
                        language=language,
                        processing_strategy=processing_strategy,
                    )
                    full_images_data.append(full_img)

                    preview_images.append(
                        PreviewImage(
                            caption=img["caption"],
                            image_path=img["image_path"],
                            page_number=img.get("page_number"),
                            source_file=img.get("source_file", file.filename),
                            image_hash=img["image_hash"],
                            # Extended metadata for preview display
                            image_metadata=img.get("image_metadata", {}),
                            bbox=img.get("bbox"),
                            docling_caption=img.get("docling_caption"),
                            surrounding_context=img.get("surrounding_context"),
                            headings=img.get("headings", []),
                            caption_cost=img.get("cost", 0.0),
                            file_type=file_type,
                            language=language,
                            processing_strategy=processing_strategy,
                        )
                    )

            processing_time = time.time() - start_time

            return UploadPreviewResponse(
                status="success",
                file_name=file.filename,
                file_type=file_type,
                chunks=preview_chunks,
                images=preview_images,
                total_chunks_count=len(full_chunks_data),
                total_images_count=len(full_images_data),
                processing_time_seconds=round(processing_time, 2),
                full_chunks_data=full_chunks_data,
                full_images_data=full_images_data,
            )

        except Exception as e:
            logger.error(f"Preview processing failed: {e}", exc_info=True)
            raise

        finally:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except Exception as cleanup_err:
                    logger.warning(f"Temp cleanup failed: {cleanup_err}")

    def save_chunks(self, request: SaveUploadRequest) -> SaveUploadResponse:
        """Save processed chunks and images to Qdrant.

        Args:
            request: SaveUploadRequest with chunk and image data

        Returns:
            SaveUploadResponse with save results
        """
        try:
            # Initialize QdrantManager with settings
            qdrant = QdrantManager(
                collection_name=self.settings.text_collection,
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )

            chunks_count = 0
            images_count = 0

            # Save text chunks
            if request.chunks:
                chunks_count = self._save_text_chunks(
                    qdrant=qdrant,
                    chunks=request.chunks,
                    source_file=request.file_name,
                    file_type=request.file_type,
                    language=request.language,
                )

            # Save images
            if request.images:
                images_count = self._save_images(
                    images=request.images,
                    source_file=request.file_name,
                )

            return SaveUploadResponse(
                status="success",
                file_name=request.file_name,
                chunks_count=chunks_count,
                images_count=images_count,
                message=f"Successfully saved {chunks_count} chunks and {images_count} images",
                text_collection=self.settings.text_collection,
                image_collection=self.settings.image_collection,
            )

        except Exception as e:
            logger.error(f"Save failed: {e}", exc_info=True)
            raise

    def _save_text_chunks(
        self,
        qdrant: QdrantManager,
        chunks: list,
        source_file: str,
        file_type: str,
        language: str,
    ) -> int:
        """Save text chunks to Qdrant with embeddings.

        Args:
            qdrant: QdrantManager instance
            chunks: List of SaveChunkData
            source_file: Source file name
            file_type: File type (pdf, docx, csv)
            language: Document language

        Returns:
            Number of chunks saved
        """
        # Build DataFrame with chunk data including all metadata
        chunks_data = []
        for chunk in chunks:
            chunks_data.append(
                {
                    "chunk": chunk.text,
                    "page_number": chunk.page_number,
                    "element_type": chunk.element_type,
                    "chunk_index": chunk.chunk_index,
                    "file_type": file_type,
                    "language": language,
                    # Extended metadata fields (flattened)
                    "headings": chunk.headings,
                    "source": chunk.source,
                    "bbox": chunk.bbox,
                    "chunk_type": chunk.chunk_type,
                    "token_count": chunk.token_count,
                    "processing_strategy": chunk.processing_strategy,
                    "ocr_used": chunk.ocr_used,
                }
            )

        chunks_df = pd.DataFrame(chunks_data)

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding.embed_texts(texts)

        # Ensure collection exists
        vector_dim = len(embeddings[0])
        qdrant.ensure_collection(dimension=vector_dim)

        # Upload to Qdrant
        success = qdrant.add_documents(
            chunks_df=chunks_df,
            embeddings=embeddings,
            source_file=source_file,
            language=language,
        )

        if not success:
            raise ValueError("Failed to upload text chunks to Qdrant")

        logger.info(f"Saved {len(chunks_df)} text chunks to '{qdrant.collection_name}'")
        return len(chunks_df)

    def _save_images(
        self,
        images: list,
        source_file: str,
    ) -> int:
        """Save image captions to image collection.

        Args:
            images: List of SaveImageData
            source_file: Source file name

        Returns:
            Number of images saved
        """
        try:
            image_qdrant = QdrantManager(
                collection_name=self.settings.image_collection,
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
            )

            # Build image DataFrame with all metadata
            images_df = pd.DataFrame(
                [
                    {
                        "chunk": img.caption,
                        "image_path": img.image_path,
                        "image_hash": img.image_hash,
                        "page_number": img.page_number,
                        "source_file": img.source_file or source_file,
                        # Image dimensions from image_metadata
                        "width": img.image_metadata.get("width", 0),
                        "height": img.image_metadata.get("height", 0),
                        "format": img.image_metadata.get("format", "unknown"),
                        "optimized_size_bytes": img.image_metadata.get(
                            "optimized_size_bytes"
                        ),
                        # Extended metadata fields
                        "bbox": img.bbox,
                        "docling_caption": img.docling_caption,
                        "surrounding_context": img.surrounding_context,
                        "headings": img.headings,
                        "caption_cost": img.caption_cost,
                        "file_type": img.file_type,
                        "language": img.language,
                        "processing_strategy": img.processing_strategy,
                    }
                    for img in images
                ]
            )

            # Embed captions
            captions = [img.caption for img in images]
            embeddings = self.embedding.embed_texts(captions)

            # Ensure collection exists
            vector_dim = len(embeddings[0])
            image_qdrant.ensure_collection(dimension=vector_dim)

            # Upload
            success = image_qdrant.add_documents(
                chunks_df=images_df,
                embeddings=embeddings,
                source_file=source_file,
            )

            if success:
                logger.info(
                    f"Saved {len(images_df)} images to '{self.settings.image_collection}'"
                )
                return len(images_df)

        except Exception as e:
            logger.warning(f"Image save failed (continuing): {e}")

        return 0
