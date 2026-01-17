"""
Document processing strategy using Docling for PDF and DOCX formats.

Provides:
- PDF: Vietnamese OCR support via EasyOCR, TableFormer ACCURATE mode for
  complex tables, image extraction with Vision API captioning integration.
- DOCX: Text extraction, table extraction, image handling without OCR.
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

# Logger configured by api/main.py at startup
logger = logging.getLogger(__name__)


class DoclingDocumentStrategy(DocumentProcessingStrategy):
    """Document processing using Docling for PDF and DOCX formats.

    Supports:
    - PDF: OCR, table extraction, image handling
    - DOCX: Text extraction, table extraction, image handling
    """

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

        logger.info("DoclingDocumentStrategy initialized")

    def _ensure_converter(self) -> None:
        """Lazy initialize the DocumentConverter for PDF and DOCX."""
        if self._converter_initialized:
            return

        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
            WordFormatOption,
        )
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
        from docling.pipeline.simple_pipeline import SimplePipeline

        ocr_config = self.docling_config.get("ocr", {})
        table_config = self.docling_config.get("table", {})
        accel_config = self.docling_config.get("acceleration", {})

        # --- PDF Pipeline Options ---
        pdf_opts = PdfPipelineOptions()

        # OCR configuration - respect mode from pdf config (PDF only)
        # Mode comes from UI selection: "no_ocr" (default) or "ocr"
        pdf_mode = self.config.get("pdf", {}).get("mode", "no_ocr")
        if pdf_mode == "ocr":
            # OCR enabled
            pdf_opts.do_ocr = True
            confidence = ocr_config.get("confidence_threshold", 0.5)
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                logger.warning(f"Invalid confidence_threshold {confidence}, using 0.5")
                confidence = 0.5
            pdf_opts.ocr_options = EasyOcrOptions(
                lang=ocr_config.get("languages", ["en", "vi"]),
                confidence_threshold=confidence,
            )
            logger.info(f"OCR enabled with languages: {ocr_config.get('languages', ['en', 'vi'])}")
        else:
            # no_ocr mode (default) - skip OCR for faster processing
            pdf_opts.do_ocr = False
            logger.info("OCR disabled (no_ocr mode - default)")

        # Table extraction
        pdf_opts.do_table_structure = True
        mode = table_config.get("mode", "accurate")
        pdf_opts.table_structure_options.mode = (
            TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST
        )
        pdf_opts.table_structure_options.do_cell_matching = table_config.get(
            "do_cell_matching", True
        )

        # Image extraction with validation
        pdf_opts.generate_picture_images = True
        images_scale = self.docling_config.get("images_scale", 2.0)
        if not isinstance(images_scale, (int, float)) or images_scale <= 0:
            logger.warning(f"Invalid images_scale {images_scale}, using 2.0")
            images_scale = 2.0
        pdf_opts.images_scale = images_scale

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
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=device_map.get(device_str, AcceleratorDevice.AUTO),
        )

        # --- Create Multi-Format Converter ---
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
            }
        )
        self._converter_initialized = True
        logger.info("Docling DocumentConverter initialized for PDF and DOCX")

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
        return [".pdf", ".docx"]

    @property
    def strategy_name(self) -> str:
        """Return strategy name."""
        return "docling"

    def can_process(self, file_path: str) -> bool:
        """Check if this strategy can process the file."""
        return self.get_file_extension(file_path) in self.supported_extensions

    def extract_elements(self, file_path: str, **kwargs) -> ProcessingResult:
        """Extract elements from PDF or DOCX using Docling."""
        return self.process(file_path, **kwargs)

    def process(self, file_path: str, **kwargs) -> ProcessingResult:
        """Process PDF or DOCX file using Docling."""
        start_time = time.time()
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

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

            # Convert document (Docling auto-detects format)
            result = self.converter.convert(source=str(file_path))
            doc = result.document

            # Log conversion details
            logger.info(
                f"Docling conversion completed: "
                f"status={result.status}, "
                f"pages={len(result.pages) if hasattr(result, 'pages') and result.pages else 0}, "
                f"source={file_path.name}, "
                f"tables={len(doc.tables) if hasattr(doc, 'tables') else 0}, "
                f"pictures={len(doc.pictures) if hasattr(doc, 'pictures') else 0}, "
                f"texts={len(list(doc.texts)) if hasattr(doc, 'texts') else 0}"
            )

            # Log document headings from texts collection
            # from docling_core.types.doc import DocItemLabel
            # headers = [
            #     (item.text, getattr(item, 'level', 0))
            #     for item in doc.texts
            #     if hasattr(item, 'label') and item.label == DocItemLabel.SECTION_HEADER
            # ]
            # if headers:
            #     logger.info(f"Document headings: {headers}")
            # else:
            #     logger.info("Document headings: None found")

            # OCR tracking - only applies to PDF
            pdf_mode = self.config.get("pdf", {}).get("mode", "no_ocr")
            self.ocr_used = file_ext == ".pdf" and pdf_mode == "ocr"

            # Get original filename from kwargs (UI passes this for uploaded files)
            original_filename = kwargs.get("original_filename")

            # Extract images
            image_data = self._extract_images(doc, file_path, original_filename)

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

    def _extract_surrounding_context(
        self, doc, pic, all_items: list, pic_index: int, max_tokens: int = 200
    ) -> str:
        """Extract surrounding text context for image captioning.

        Args:
            doc: Docling document
            pic: PictureItem being processed
            all_items: List of (item, level) tuples from doc.iterate_items()
            pic_index: Index of pic in all_items
            max_tokens: Maximum tokens for context (default: 200)

        Returns:
            Combined context string (preceding text + figure caption)
        """
        from docling_core.types.doc import TextItem

        # Get preceding text items (up to 2)
        preceding_text = []
        for j in range(max(0, pic_index - 3), pic_index):
            prev_item, _ = all_items[j]
            if isinstance(prev_item, TextItem) and hasattr(prev_item, "text"):
                preceding_text.append(prev_item.text)
                if len(preceding_text) >= 2:
                    break

        # Get figure caption from PDF structure
        docling_caption = ""
        if hasattr(pic, "caption_text"):
            try:
                docling_caption = pic.caption_text(doc=doc) or ""
            except Exception:
                pass

        # Combine context
        context_parts = preceding_text
        if docling_caption:
            context_parts.append(f"Figure caption: {docling_caption}")

        full_context = " ".join(context_parts)

        # Truncate to max_tokens (approximate: 1 token ≈ 4 chars for English)
        max_chars = max_tokens * 4
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "..."

        return full_context.strip()

    def _extract_images(
        self, doc, file_path: Path, original_filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract images from Docling document.

        Args:
            doc: Docling document object
            file_path: Path to the document file (may be temp file)
            original_filename: Original filename from upload (preferred for metadata)
        """
        from backend.utils.image_storage import ImageStorageUtility

        image_data = []

        # Use original filename if available, otherwise fall back to file_path
        display_name = original_filename or file_path.name
        display_stem = Path(display_name).stem if original_filename else file_path.stem

        storage = ImageStorageUtility(
            base_storage_path=self.config.get("pdf", {}).get(
                "image_storage_path", "extracted_images"
            ),
            create_subdirs=True,
        )

        # Build item index for context extraction
        all_items = list(doc.iterate_items())
        pic_to_index = {}
        for idx, (item, level) in enumerate(all_items):
            if hasattr(item, "image") and item.image:
                pic_to_index[id(item)] = idx

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
                    original_filename=f"page_{page_num}_{display_stem}_{i}",
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

                # Extract surrounding context for Vision API
                pic_index = pic_to_index.get(id(pic), -1)
                surrounding_context = ""
                if pic_index >= 0:
                    surrounding_context = self._extract_surrounding_context(
                        doc, pic, all_items, pic_index, max_tokens=200
                    )

                image_data.append({
                    "image_path": str(img_path),
                    "caption": "",  # Filled by _caption_images
                    "docling_caption": docling_caption,
                    "surrounding_context": surrounding_context,  # NEW: for Vision API
                    "image_hash": image_hash,
                    "page_number": page_num,
                    "source_file": display_name,  # Track source document (original filename)
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
        """Caption images using Vision API with surrounding context."""
        if not self.captioner:
            for img in image_data:
                img["caption"] = img.get("docling_caption") or "Image from document"
            return image_data

        for img in image_data:
            try:
                # Build context-aware prompt if context available
                custom_prompt = None
                context = img.get("surrounding_context", "")
                docling_cap = img.get("docling_caption", "")

                if context or docling_cap:
                    context_lines = []
                    if context:
                        context_lines.append(f"Document context: {context}")
                    if docling_cap:
                        context_lines.append(f"Figure caption: {docling_cap}")

                    custom_prompt = (
                        self.captioner.PRODUCT_CAPTION_PROMPT
                        + "\n\n"
                        + "\n".join(context_lines)
                    )

                caption, cost = self.captioner.caption_image(
                    img["image_path"], custom_prompt=custom_prompt
                )
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
