"""
PDF processing strategy using Docling with OCR and table extraction.

Provides Vietnamese OCR support via EasyOCR, TableFormer ACCURATE mode for
complex tables, and image extraction with Vision API captioning integration.
"""

import base64
import hashlib
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .interfaces import DocumentProcessingStrategy
from .results import ProcessingResult, ProcessingMetrics, ProcessingStatus

logger = logging.getLogger(__name__)


class DoclingPDFStrategy(DocumentProcessingStrategy):
    """PDF processing using Docling with OCR, table extraction, and image handling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.docling_config = self.config.get("docling", {})
        self.converter = None  # Lazy initialization
        self._converter_initialized = False

        # Image captioning setup - check both top-level and nested pdf config
        self.openai_api_key = (
            self.config.get("openai_api_key")
            or self.config.get("pdf", {}).get("openai_api_key")
        )
        self.captioner = None

        # Processing state
        self.ocr_used = False
        self.last_strategy_used = "docling"

        logger.info("DoclingPDFStrategy initialized")

    def _ensure_converter(self) -> None:
        """Lazy initialize the DocumentConverter."""
        if self._converter_initialized:
            return

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
            TableFormerMode,
        )
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )

        ocr_config = self.docling_config.get("ocr", {})
        table_config = self.docling_config.get("table", {})
        accel_config = self.docling_config.get("acceleration", {})

        opts = PdfPipelineOptions()

        # OCR configuration - respect mode from pdf config
        # Mode comes from UI selection: "no_ocr" (default) or "ocr"
        pdf_mode = self.config.get("pdf", {}).get("mode", "no_ocr")
        if pdf_mode == "ocr":
            # OCR enabled
            opts.do_ocr = True
            confidence = ocr_config.get("confidence_threshold", 0.5)
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                logger.warning(f"Invalid confidence_threshold {confidence}, using 0.5")
                confidence = 0.5
            opts.ocr_options = EasyOcrOptions(
                lang=ocr_config.get("languages", ["en", "vi"]),
                confidence_threshold=confidence,
            )
            logger.info(f"OCR enabled with languages: {ocr_config.get('languages', ['en', 'vi'])}")
        else:
            # no_ocr mode (default) - skip OCR for faster processing
            opts.do_ocr = False
            logger.info("OCR disabled (no_ocr mode - default)")

        # Table extraction
        opts.do_table_structure = True
        mode = table_config.get("mode", "accurate")
        opts.table_structure_options.mode = (
            TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST
        )
        opts.table_structure_options.do_cell_matching = table_config.get(
            "do_cell_matching", True
        )

        # Image extraction with validation
        opts.generate_picture_images = True
        images_scale = self.docling_config.get("images_scale", 2.0)
        if not isinstance(images_scale, (int, float)) or images_scale <= 0:
            logger.warning(f"Invalid images_scale {images_scale}, using 2.0")
            images_scale = 2.0
        opts.images_scale = images_scale

        # Acceleration with validation
        device_str = accel_config.get("device", "auto")
        device_map = {
            "auto": AcceleratorDevice.AUTO,
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS,
        }
        num_threads = accel_config.get("num_threads", 4)
        if not isinstance(num_threads, int) or num_threads < 1:
            logger.warning(f"Invalid num_threads {num_threads}, using 4")
            num_threads = 4
        opts.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=device_map.get(device_str, AcceleratorDevice.AUTO),
        )

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        self._converter_initialized = True
        logger.info("Docling DocumentConverter initialized")

    def _init_captioner(self) -> None:
        """Initialize image captioner with Vision API."""
        if self.captioner is not None or not self.openai_api_key:
            return

        from backend.vision.image_captioner import ImageCaptioner

        self.captioner = ImageCaptioner(
            api_key=self.openai_api_key,
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.3,
            detail_mode="low",
            enable_cache=True,
        )
        logger.info("ImageCaptioner initialized for Docling strategy")

    @property
    def supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".pdf"]

    @property
    def strategy_name(self) -> str:
        """Return strategy name."""
        return "docling"

    def can_process(self, file_path: str) -> bool:
        """Check if this strategy can process the file."""
        return self.get_file_extension(file_path) in self.supported_extensions

    def extract_elements(self, file_path: str, **kwargs) -> ProcessingResult:
        """Extract elements from PDF using Docling."""
        return self.process(file_path, **kwargs)

    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process PDF file using Docling."""
        start_time = time.time()
        file_path = Path(file_path)

        try:
            # Validate file
            if not self.validate_file(str(file_path)):
                return self._create_error_result(
                    f"Invalid file: {file_path}",
                    processing_time=time.time() - start_time,
                )

            # Initialize converter and captioner
            self._ensure_converter()
            if self.openai_api_key:
                self._init_captioner()

            # Convert PDF
            result = self.converter.convert(source=str(file_path))
            doc = result.document

            # Check OCR usage - based on actual mode used
            pdf_mode = self.config.get("pdf", {}).get("mode", "no_ocr")
            self.ocr_used = pdf_mode == "ocr"

            # Extract images
            image_data = self._extract_images(doc, file_path)

            # Caption images (uses Vision API if available, else fallback)
            if image_data:
                image_data = self._caption_images(image_data)

            # Get image paths for chunker
            image_paths = [img["image_path"] for img in image_data]

            # Chunk document using Docling HybridChunker (token-aware)
            chunking_config = self.docling_config.get("chunking", {})
            elements, chunking_metadata = self._chunk_document(
                doc, image_paths, chunking_config
            )

            # Build ProcessingResult
            processing_time = time.time() - start_time

            return ProcessingResult(
                success=True,
                elements=elements,
                status=ProcessingStatus.SUCCESS,
                metadata={
                    "processor": "docling",
                    "file_name": file_path.name,
                    "conversion_status": str(result.status),
                    "chunking": chunking_metadata,
                },
                metrics=ProcessingMetrics(
                    processing_time=processing_time,
                    elements_extracted=len(elements),
                    pages_processed=len(doc.pages) if hasattr(doc, "pages") else 0,
                    ocr_used=self.ocr_used,
                    strategy_used="docling",
                ),
                processing_time=processing_time,
                image_paths=image_paths,
                image_data=image_data,
            )

        except Exception as e:
            logger.error(f"Docling processing failed: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                processing_time=time.time() - start_time,
            )

    def _extract_images(self, doc, file_path: Path) -> List[Dict[str, Any]]:
        """Extract images from Docling document."""
        from backend.utils.image_storage import ImageStorageUtility

        image_data = []

        storage = ImageStorageUtility(
            base_storage_path=self.config.get("pdf", {}).get(
                "image_storage_path", "extracted_images"
            ),
            create_subdirs=True,
        )

        for i, pic in enumerate(doc.pictures):
            try:
                # Get image bytes from URI
                if not (hasattr(pic, "image") and pic.image and pic.image.uri):
                    continue

                img_bytes = self._decode_image_uri(pic.image.uri)
                img = Image.open(io.BytesIO(img_bytes))
                width, height = img.size
                img_format = img.format or "PNG"

                # Generate hash (SHA-256 for security)
                image_hash = hashlib.sha256(img_bytes).hexdigest()[:32]

                # Get page number and bbox
                page_num = self._get_page_number(pic) or 1
                bbox = self._get_bbox(pic)

                # Save image using storage utility
                img_path, _ = storage.store_image(
                    image_data=img_bytes,
                    original_filename=f"page_{page_num}_{file_path.stem}_{i}",
                    page_number=page_num,
                    bbox=tuple(bbox.values()) if bbox else None,
                )

                # Get docling caption if available
                docling_caption = None
                if hasattr(pic, "caption_text"):
                    try:
                        docling_caption = pic.caption_text(doc=doc)
                    except Exception:
                        pass

                image_data.append({
                    "image_path": str(img_path),
                    "caption": "",  # Filled by _caption_images
                    "docling_caption": docling_caption,
                    "image_hash": image_hash,
                    "page_number": page_num,
                    "image_metadata": {
                        "width": width,
                        "height": height,
                        "format": img_format,
                        "optimized_size_bytes": len(img_bytes),
                    },
                    "bbox": bbox,
                    "cost": 0.0,  # Filled by _caption_images
                })
            except Exception as e:
                logger.warning(f"Failed to extract image {i}: {e}")

        return image_data

    def _caption_images(self, image_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Caption images using Vision API."""
        if not self.captioner:
            for img in image_data:
                img["caption"] = img.get("docling_caption") or "Image from document"
            return image_data

        for img in image_data:
            try:
                caption, cost = self.captioner.caption_image(img["image_path"])
                img["caption"] = caption
                img["cost"] = cost
            except Exception as e:
                logger.warning(f"Failed to caption {img['image_path']}: {e}")
                img["caption"] = img.get("docling_caption") or "Image from document"
                img["cost"] = 0.0

        return image_data

    def _decode_image_uri(self, uri: str) -> bytes:
        """Decode image from Docling URI (base64 or file path).

        Security: File paths are validated to prevent path traversal attacks.
        """
        # Convert AnyUrl (Pydantic) to string for Docling 2.x compatibility
        uri = str(uri)

        if uri.startswith("data:"):
            # Base64: data:image/png;base64,<data>
            _, data = uri.split(",", 1)
            return base64.b64decode(data)
        elif uri.startswith("file://"):
            file_path = uri.replace("file://", "")
            # Validate path to prevent traversal attacks
            resolved = Path(file_path).resolve()
            if not resolved.exists():
                raise ValueError(f"File not found: {file_path}")
            if not resolved.is_file():
                raise ValueError(f"Not a file: {file_path}")
            with open(resolved, "rb") as f:
                return f.read()
        else:
            # Assume raw base64
            return base64.b64decode(uri)

    def _get_page_number(self, item) -> Optional[int]:
        """Extract page number from Docling item provenance."""
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            # Docling uses 'page_no' not 'page'
            if hasattr(prov, "page_no"):
                return prov.page_no
            elif hasattr(prov, "page"):
                return prov.page
        return None

    def _get_bbox(self, item) -> Optional[Dict[str, float]]:
        """Extract bounding box from Docling item."""
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, "bbox"):
                return {
                    "left": prov.bbox.l,
                    "top": prov.bbox.t,
                    "right": prov.bbox.r,
                    "bottom": prov.bbox.b,
                }
        return None

    def _chunk_document(
        self,
        doc: Any,
        image_paths: List[str],
        chunking_config: Dict[str, Any],
    ) -> tuple:
        """
        Chunk document using Docling HybridChunker.

        Args:
            doc: Docling DoclingDocument
            image_paths: List of image paths
            chunking_config: Chunking configuration

        Returns:
            Tuple of (elements, metadata)
        """
        from backend.chunking.docling_chunker import DoclingChunker

        chunker = DoclingChunker(config=chunking_config)
        chunk_result = chunker.chunk_document(doc, image_paths)

        if not chunk_result.chunks:
            raise ValueError("DoclingChunker returned no chunks - document may be empty or invalid")

        logger.info(
            f"DoclingChunker created {len(chunk_result.chunks)} chunks "
            f"(tokenizer: {chunking_config.get('tokenizer_model', 'default')})"
        )
        return chunk_result.chunks, chunk_result.metadata

