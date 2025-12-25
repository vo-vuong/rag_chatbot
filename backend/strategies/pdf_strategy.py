"""
PDF processing strategy using unstructured with OCR integration.

This module provides comprehensive PDF processing capabilities with multi-tier fallback
system, OCR integration, and semantic chunking support.
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pikepdf
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

from ..chunking.semantic_chunker import SemanticChunker
from ..ocr.tesseract_ocr import get_tesseract_ocr
from ..utils.image_storage import ImageStorageUtility
from .interfaces import DocumentProcessingStrategy
from .results import ProcessingResult

# Configure logging
logger = logging.getLogger(__name__)

# PDF processing strategies
PDF_PROCESSING_STRATEGIES = {
    "auto": "auto",  # Auto-detect best strategy
    "fast": "fast",  # Fast processing without OCR
    "hi_res": "hi_res",  # High-resolution processing with OCR
    "ocr_only": "ocr_only",  # Force OCR processing
    "fallback": "fallback",  # Use fallback strategies
}

# Default configuration
DEFAULT_PDF_STRATEGY = "auto"
DEFAULT_INFER_TABLE_STRUCTURE = True
DEFAULT_EXTRACT_IMAGES = True
DEFAULT_IMAGE_STORAGE_PATH = "extracted_images"


class PDFProcessingStrategy(DocumentProcessingStrategy):
    """
    PDF processing strategy with multi-tier fallback system and OCR integration.

    This strategy provides comprehensive PDF processing capabilities including:
    - Primary processing with unstructured hi_res mode and OCR
    - Fast processing fallback for text-based PDFs
    - Final fallback to pdfplumber for basic extraction
    - Cross-platform OCR integration
    - Semantic chunking support
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        ocr_languages: Optional[List[str]] = None,
        chunker: Optional[SemanticChunker] = None,
    ):
        """
        Initialize PDF processing strategy.

        Args:
            config: Configuration dictionary for processing parameters
            ocr_languages: List of languages for OCR (e.g., ['en', 'vi'])
            chunker: Optional semantic chunker instance
        """
        super().__init__(config)

        # Initialize OCR
        self.ocr = get_tesseract_ocr(languages=ocr_languages)

        # Initialize chunker
        self.chunker = chunker or SemanticChunker()

        # Set processing parameters from config
        self.processing_strategy = self.config.get("strategy", DEFAULT_PDF_STRATEGY)
        self.infer_table_structure = self.config.get(
            "infer_table_structure", DEFAULT_INFER_TABLE_STRUCTURE
        )
        self.extract_images = self.config.get("extract_images", DEFAULT_EXTRACT_IMAGES)
        self.chunk_after_extraction = self.config.get("chunk_after_extraction", True)

        # Initialize image storage
        image_storage_path = self.config.get(
            "image_storage_path", DEFAULT_IMAGE_STORAGE_PATH
        )
        self.image_storage = ImageStorageUtility(
            base_storage_path=image_storage_path,
            enable_optimization=True,
            create_subdirs=True,
        )

        # Processing state
        self.last_processing_strategy_used = None
        self.ocr_used = False

        # Initialize ImageCaptioner if API key available
        self.openai_api_key = self.config.get("openai_api_key")
        self.captioner = None
        self.caption_failure_mode = self.config.get("caption_failure_mode", "graceful")

        if self.openai_api_key and self.extract_images:
            try:
                from backend.vision.image_captioner import ImageCaptioner
                self.captioner = ImageCaptioner(
                    api_key=self.openai_api_key,
                    model=self.config.get("vision_model", "gpt-4o-mini"),
                    max_tokens=self.config.get("vision_max_tokens", 100),
                    temperature=self.config.get("vision_temperature", 0.3),
                    detail_mode=self.config.get("vision_detail_mode", "low")
                )
                self._logger.info("ImageCaptioner initialized for PDF processing")
            except Exception as e:
                self._logger.warning(f"ImageCaptioner not available: {e}")
                self.captioner = None
        else:
            self._logger.info("Image captioning disabled (no API key or extract_images=False)")

        self._logger.info(
            f"PDF processing strategy initialized with OCR available: {self.ocr.is_configured}"
        )

    def _get_document_name(self, filename: str) -> str:
        """
        Extract document name without extension for folder naming.

        Args:
            filename: Original filename or file path

        Returns:
            Document name suitable for folder creation
        """
        # Extract filename from path if needed
        filename = Path(filename).name
        # Remove extension
        document_name = Path(filename).stem
        # Sanitize document name (remove problematic characters)
        document_name = "".join(
            c for c in document_name if c.isalnum() or c in (' ', '-', '_')
        )
        # Limit length to avoid filesystem issues
        return document_name[:100] or "unnamed_document"

    def _cleanup_figures_folder(self, base_path: Optional[str] = None) -> None:
        """
        Clean up the figures folder created by unstructured library.

        The unstructured library automatically creates a 'figures' folder when
        extract_images_in_pdf=True. Since we store images in our own organized
        structure, we clean up the redundant figures folder.

        Args:
            base_path: Base path where figures folder might be created.
                      Defaults to current working directory.
        """
        if not base_path:
            base_path = os.getcwd()

        figures_path = Path(base_path) / "figures"

        try:
            if figures_path.exists() and figures_path.is_dir():
                # Check if it contains image files
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']:
                    image_files.extend(figures_path.glob(ext))
                    image_files.extend(figures_path.glob(ext.upper()))

                if image_files:
                    self._logger.info(
                        f"Cleaning up unstructured figures folder: {figures_path}"
                    )
                    shutil.rmtree(figures_path)
                    self._logger.info(
                        f"Removed {len(image_files)} image files from figures folder"
                    )
                else:
                    self._logger.debug(
                        f"Figures folder {figures_path} is empty or doesn't contain images"
                    )
            else:
                self._logger.debug(f"Figures folder {figures_path} does not exist")

        except Exception as e:
            self._logger.warning(
                f"Failed to clean up figures folder {figures_path}: {e}"
            )

    def _process_extracted_images(
        self, elements: List[Any], document_name: str
    ) -> Dict[str, Any]:
        """
        Process and store images extracted from PDF elements.

        Args:
            elements: List of elements from unstructured processing
            document_name: Name of the document for organizing images

        Returns:
            Dictionary with image processing statistics and metadata
        """
        import base64
        import io

        image_stats = {
            "total_images": 0,
            "stored_images": 0,
            "failed_images": 0,
            "image_paths": [],
            "document_name": document_name,
            "processing_errors": [],
        }

        if not self.extract_images or not elements:
            return image_stats

        # Create document-specific storage directory
        doc_image_path = Path(self.image_storage.base_path) / document_name

        # Create a document-specific image storage instance
        doc_image_storage = ImageStorageUtility(
            base_storage_path=str(doc_image_path),
            enable_optimization=self.image_storage.enable_optimization,
            create_subdirs=False,  # Don't create date subdirs for document-specific storage
        )

        self._logger.info(
            f"Processing extracted images for document: {document_name} -> {doc_image_path}"
        )

        for idx, element in enumerate(elements):
            try:
                # Check if element is an image type
                if hasattr(element, 'category') and element.category == 'Image':
                    image_stats["total_images"] += 1

                    # Get image metadata
                    element_metadata = getattr(element, 'metadata', None)
                    if not element_metadata:
                        self._logger.warning(f"Image element {idx} has no metadata")
                        continue

                    image_bytes = None
                    image_source = ""

                    # Try to get image data from file path first (unstructured approach)
                    image_path = getattr(element_metadata, 'image_path', None)
                    if image_path and Path(image_path).exists():
                        try:
                            with open(image_path, 'rb') as f:
                                image_bytes = f.read()
                            image_source = f"file: {image_path}"
                            self._logger.info(f"Loading image from file: {image_path}")
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to read image file {image_path}: {e}"
                            )

                    # Fallback to Base64 if available
                    if not image_bytes:
                        image_base64 = getattr(element_metadata, 'image_base64', None)
                        if image_base64:
                            try:
                                image_bytes = base64.b64decode(image_base64)
                                image_source = "base64 data"
                                self._logger.info(f"Loading image from base64 data")
                            except Exception as e:
                                error_msg = (
                                    f"Failed to decode base64 for image {idx}: {str(e)}"
                                )
                                self._logger.error(error_msg)
                                image_stats["processing_errors"].append(error_msg)
                                image_stats["failed_images"] += 1
                                continue

                    # If no image data found, skip
                    if not image_bytes:
                        self._logger.warning(
                            f"Image element {idx} has no accessible image data"
                        )
                        continue

                    # Get additional metadata
                    page_number = getattr(element_metadata, 'page_number', idx + 1)

                    # Convert coordinates to JSON-serializable tuple
                    raw_bbox = getattr(element_metadata, 'coordinates', None)
                    if raw_bbox is not None:
                        # Handle PixelSpace object from unstructured
                        if hasattr(raw_bbox, 'points'):
                            # Extract bounding box coordinates from PixelSpace
                            points = raw_bbox.points
                            if len(points) >= 2:
                                # Get min/max x,y from points
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                            else:
                                bbox = None
                        elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                            # Already a tuple/list of 4 coordinates
                            bbox = tuple(raw_bbox)
                        else:
                            bbox = None
                    else:
                        bbox = None

                    original_filename = getattr(
                        element_metadata, 'filename', f"image_{idx}"
                    )

                    # Store the image using document-specific ImageStorage utility
                    try:
                        # Store image in document-specific folder
                        storage_path, metadata = doc_image_storage.store_image(
                            image_data=image_bytes,
                            original_filename=f"page_{page_number}_{original_filename}",
                            page_number=page_number,
                            bbox=bbox,
                            target_format='PNG',
                        )

                        # Update element metadata with stored image path
                        element_metadata.image_path = str(storage_path)
                        element_metadata.storage_metadata = metadata

                        # Track successful storage
                        image_stats["stored_images"] += 1
                        image_stats["image_paths"].append(str(storage_path))

                        self._logger.info(f"Stored image {idx}: {storage_path}")

                    except Exception as e:
                        error_msg = f"Failed to store image {idx}: {str(e)}"
                        self._logger.error(error_msg)
                        image_stats["processing_errors"].append(error_msg)
                        image_stats["failed_images"] += 1

            except Exception as e:
                error_msg = f"Unexpected error processing element {idx}: {str(e)}"
                self._logger.error(error_msg)
                image_stats["processing_errors"].append(error_msg)

        # Log processing summary
        self._logger.info(
            f"Image processing completed for {document_name}: "
            f"{image_stats['stored_images']}/{image_stats['total_images']} images stored "
            f"in {doc_image_path}"
        )

        return image_stats

    def _caption_extracted_images(
        self, image_paths: List[str], document_name: str
    ) -> List[Dict[str, Any]]:
        """
        Generate captions for extracted images using Vision API.

        Args:
            image_paths: List of paths to extracted images
            document_name: Name of the document for context

        Returns:
            List of dictionaries containing image data with captions
        """
        from backend.vision.image_captioner import ImageCaptioningError
        import hashlib

        image_data = []

        if not self.captioner or not image_paths:
            self._logger.info("Skipping image captioning (captioner not available or no images)")
            return image_data

        self._logger.info(f"Starting caption generation for {len(image_paths)} images...")

        for idx, image_path in enumerate(image_paths):
            try:
                # Generate image hash for deduplication
                with open(image_path, 'rb') as f:
                    image_hash = hashlib.md5(f.read()).hexdigest()

                # Get image metadata from path
                from PIL import Image
                img = Image.open(image_path)
                width, height = img.size
                format_name = img.format

                # Extract page number from path or metadata
                page_number = idx + 1  # Default fallback

                # Caption the image
                caption, cost = self.captioner.caption_image(image_path)

                image_data.append({
                    "image_path": str(image_path),
                    "caption": caption,
                    "image_hash": image_hash,
                    "page_number": page_number,
                    "image_metadata": {
                        "width": width,
                        "height": height,
                        "format": format_name or "PNG",
                        "optimized_size_bytes": Path(image_path).stat().st_size
                    },
                    "cost": cost
                })

                self._logger.debug(
                    f"Captioned image {idx+1}/{len(image_paths)}: "
                    f"'{caption[:50]}...' (cost: ${cost:.6f})"
                )

            except ImageCaptioningError as caption_error:
                # Handle caption failure based on mode
                if self.caption_failure_mode == "strict":
                    self._logger.error(f"STRICT MODE: Caption failed: {caption_error}")
                    raise Exception(
                        f"Failed to caption image {image_path}. "
                        f"Upload aborted (strict mode). Error: {str(caption_error)}"
                    )
                elif self.caption_failure_mode == "graceful":
                    self._logger.warning(
                        f"GRACEFUL MODE: Using fallback caption for {image_path}: {caption_error}"
                    )
                    # Use fallback caption
                    try:
                        with open(image_path, 'rb') as f:
                            image_hash = hashlib.md5(f.read()).hexdigest()
                        from PIL import Image
                        img = Image.open(image_path)
                        width, height = img.size

                        image_data.append({
                            "image_path": str(image_path),
                            "caption": "Image (caption unavailable)",
                            "image_hash": image_hash,
                            "page_number": idx + 1,
                            "image_metadata": {
                                "width": width,
                                "height": height,
                                "format": img.format or "PNG",
                                "optimized_size_bytes": Path(image_path).stat().st_size
                            },
                            "cost": 0.0
                        })
                    except Exception as e:
                        self._logger.error(f"Failed to create fallback caption: {e}")
                elif self.caption_failure_mode == "skip":
                    self._logger.info(
                        f"SKIP MODE: Skipping image {image_path} due to caption failure"
                    )
                    # Don't add to image_data
                else:
                    raise ValueError(f"Invalid caption_failure_mode: {self.caption_failure_mode}")

            except Exception as e:
                self._logger.error(f"Failed to process/caption image {image_path}: {e}")
                # Handle based on failure mode
                if self.caption_failure_mode == "strict":
                    raise
                # For graceful/skip modes, continue to next image
                continue

        total_cost = sum(img["cost"] for img in image_data)
        self._logger.info(
            f"Caption generation completed: {len(image_data)}/{len(image_paths)} images captioned, "
            f"total cost: ${total_cost:.4f}"
        )

        return image_data

    @property
    def supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".pdf"]

    @property
    def strategy_name(self) -> str:
        """Get strategy name."""
        return "PDF Processing Strategy"

    def can_process(self, file_path: str) -> bool:
        """
        Check if this strategy can process the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this strategy can process the file, False otherwise
        """
        if not self.validate_file(file_path):
            return False

        extension = self.get_file_extension(file_path)
        return extension.lower() in self.supported_extensions

    def extract_elements(
        self,
        file_path: str,
        languages: Optional[List[str]] = None,
        original_filename: Optional[str] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Extract structural elements from the PDF document.

        Args:
            file_path: Path to the PDF file
            languages: List of languages for processing
            original_filename: Original filename to override metadata
            **kwargs: Additional parameters for extraction

        Returns:
            ProcessingResult containing extracted elements and metadata
        """
        start_time = time.time()

        try:
            self._logger.info(f"Starting PDF processing for: {file_path}")

            # Reset processing state
            self.last_processing_strategy_used = None
            self.ocr_used = False

            # Validate file
            if not self.validate_file(file_path):
                return ProcessingResult(
                    success=False, error_message=f"Invalid file: {file_path}"
                )

            # Determine processing strategy
            strategy = kwargs.get("strategy", self.processing_strategy)

            # Map legacy strategy names to current ones
            strategy_mapping = {"ocr": "ocr_only"}  # Map old "ocr" to new "ocr_only"
            strategy = strategy_mapping.get(strategy, strategy)

            # Apply multi-tier processing based on strategy
            if strategy == "auto":
                elements, processing_info = self._process_auto_strategy(
                    file_path, languages
                )
            elif strategy == "fast":
                elements, processing_info = self._process_fast_strategy(
                    file_path, languages
                )
            elif strategy == "hi_res":
                elements, processing_info = self._process_hires_strategy(
                    file_path, languages
                )
            elif strategy == "ocr_only":
                elements, processing_info = self._process_ocr_only_strategy(
                    file_path, languages
                )
            elif strategy == "fallback":
                elements, processing_info = self._process_with_fallback(
                    file_path, languages
                )
            else:
                self._logger.warning(f"Unknown strategy: {strategy}, using auto")
                elements, processing_info = self._process_auto_strategy(
                    file_path, languages
                )

            # Override element metadata with original filename if provided
            if original_filename and elements:
                self._override_element_metadata(elements, original_filename)

            # Process extracted images if enabled
            image_stats = {}
            image_data = []
            if self.extract_images and elements:
                # Get document name for organizing images
                doc_name = self._get_document_name(original_filename or file_path)

                # Process and store images
                image_stats = self._process_extracted_images(elements, doc_name)

                # Add image statistics to processing info
                processing_info["image_extraction"] = image_stats
                image_paths = image_stats.get("image_paths", [])

                # Caption extracted images if captioner available
                if image_paths:
                    image_data = self._caption_extracted_images(image_paths, doc_name)
                    self._logger.info(
                        f"Captioned {len(image_data)} out of {len(image_paths)} extracted images"
                    )
            else:
                image_paths = []

            # Apply semantic chunking if requested
            if self.chunk_after_extraction and elements:
                chunk_result = self.chunker.chunk_elements(
                    elements, languages, image_paths=image_paths
                )
                elements = chunk_result.chunks
                processing_info["chunking_stats"] = chunk_result.stats

            # Calculate processing time
            processing_time = time.time() - start_time

            # Extract total pages from elements
            total_pages = self._extract_total_pages(elements)

            # Calculate caption costs
            total_caption_cost = sum(img["cost"] for img in image_data)
            captioned_count = len(image_data)

            # Prepare metadata
            metadata = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "processing_time": processing_time,
                "strategy_used": self.last_processing_strategy_used,
                "ocr_used": self.ocr_used,
                "ocr_available": self.ocr.is_configured,
                "elements_extracted": len(elements),
                "total_pages": total_pages,
                "languages": languages,
                "extracted_images": image_stats.get("total_images", 0),
                "stored_images": image_stats.get("stored_images", 0),
                "failed_images": image_stats.get("failed_images", 0),
                "captioned_count": captioned_count,
                "caption_total_cost": total_caption_cost,
                "image_paths": image_stats.get("image_paths", []),
                "document_name": image_stats.get("document_name", ""),
                "image_extraction_enabled": self.extract_images,
                **processing_info,
            }

            self._logger.info(
                f"PDF processing completed in {processing_time:.2f}s: "
                f"{len(elements)} elements extracted using {self.last_processing_strategy_used}"
            )

            # Clean up the figures folder created by unstructured
            if self.extract_images:
                self._cleanup_figures_folder()

            return ProcessingResult(
                success=True,
                elements=elements,
                metadata=metadata,
                processing_time=processing_time,
                image_paths=image_stats.get("image_paths", []),
                image_data=image_data
            )

        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            self._logger.error(error_msg, exc_info=True)

            # Clean up the figures folder even if processing failed
            if self.extract_images:
                self._cleanup_figures_folder()

            return ProcessingResult(
                success=False,
                error_message=error_msg,
                processing_time=time.time() - start_time,
            )

    def _process_auto_strategy(
        self, file_path: str, languages: Optional[List[str]] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Auto-detect and apply the best processing strategy.

        Args:
            file_path: Path to the PDF file
            languages: List of languages for processing

        Returns:
            Tuple of (elements, processing_info)
        """
        # First, try to determine if PDF is text-based or image-based
        is_text_based = self._is_text_based_pdf(file_path)

        if is_text_based:
            self._logger.info("Detected text-based PDF, trying fast processing first")
            elements, processing_info = self._process_fast_strategy(
                file_path, languages
            )

            # Check if fast processing extracted meaningful content
            if elements and self._has_meaningful_content(elements):
                self.last_processing_strategy_used = "fast"
                return elements, processing_info

        # Fall back to high-resolution processing with OCR
        self._logger.info("Using high-resolution processing with OCR")
        elements, processing_info = self._process_hires_strategy(file_path, languages)

        if elements and self._has_meaningful_content(elements):
            self.last_processing_strategy_used = "hi_res"
            return elements, processing_info

        # Final fallback to pdfplumber
        self._logger.info("Falling back to pdfplumber processing")
        elements, processing_info = self._process_pdfplumber_fallback(
            file_path, languages
        )

        self.last_processing_strategy_used = "pdfplumber_fallback"
        return elements, processing_info

    def _process_fast_strategy(
        self, file_path: str, languages: Optional[List[str]] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process PDF using fast strategy without OCR.

        Args:
            file_path: Path to the PDF file
            languages: List of languages for processing

        Returns:
            Tuple of (elements, processing_info)
        """
        try:
            self._logger.info("Processing PDF with fast strategy")

            elements = partition_pdf(
                filename=file_path,
                strategy=PartitionStrategy.FAST,
                infer_table_structure=self.infer_table_structure,
                languages=languages,
                extract_images_in_pdf=self.extract_images,
                include_page_breaks=True,
            )

            processing_info = {
                "strategy": "fast",
                "ocr_enabled": False,
                "table_detection": self.infer_table_structure,
                "success": True,
            }

            self.ocr_used = False
            self.last_processing_strategy_used = "fast"
            return elements, processing_info

        except Exception as e:
            self._logger.warning(f"Fast processing failed: {e}")
            return [], {"strategy": "fast", "success": False, "error": str(e)}

    def _process_hires_strategy(
        self, file_path: str, languages: Optional[List[str]] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process PDF using high-resolution strategy with OCR.

        Args:
            file_path: Path to the PDF file
            languages: List of languages for processing

        Returns:
            Tuple of (elements, processing_info)
        """
        try:
            self._logger.info("Processing PDF with high-resolution strategy")

            # Configure OCR if available
            ocr_enabled = self.ocr.is_configured and self.ocr.test_ocr()
            ocr_languages = languages or ["en", "vi"]

            elements = partition_pdf(
                filename=file_path,
                strategy=PartitionStrategy.HI_RES,
                infer_table_structure=self.infer_table_structure,
                extract_images_in_pdf=self.extract_images,
                languages=ocr_languages if ocr_enabled else None,
                include_page_breaks=True,
            )

            processing_info = {
                "strategy": "hi_res",
                "ocr_enabled": ocr_enabled,
                "ocr_languages": ocr_languages if ocr_enabled else None,
                "table_detection": self.infer_table_structure,
                "image_extraction": self.extract_images,
                "success": True,
            }

            self.ocr_used = ocr_enabled
            self.last_processing_strategy_used = "hi_res"
            return elements, processing_info

        except Exception as e:
            self._logger.warning(f"High-resolution processing failed: {e}")
            return [], {"strategy": "hi_res", "success": False, "error": str(e)}

    def _process_ocr_only_strategy(
        self, file_path: str, languages: Optional[List[str]] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process PDF using OCR-only strategy.

        Args:
            file_path: Path to the PDF file
            languages: List of languages for processing

        Returns:
            Tuple of (elements, processing_info)
        """
        if not self.ocr.is_configured:
            self._logger.error(
                "OCR-only processing requested but OCR is not configured"
            )
            return [], {
                "strategy": "ocr_only",
                "success": False,
                "error": "OCR not configured",
            }

        try:
            self._logger.info("Processing PDF with OCR-only strategy")

            ocr_languages = languages or ["en", "vi"]

            elements = partition_pdf(
                filename=file_path,
                strategy=PartitionStrategy.HI_RES,
                infer_table_structure=False,  # Disable table detection for OCR
                extract_images_in_pdf=True,
                languages=ocr_languages,
                include_page_breaks=True,
                # Use high resolution strategy to force OCR processing
                # Modern unstructured automatically applies OCR when needed
            )

            processing_info = {
                "strategy": "ocr_only",
                "ocr_enabled": True,
                "ocr_languages": ocr_languages,
                "table_detection": False,
                "success": True,
            }

            self.ocr_used = True
            self.last_processing_strategy_used = "ocr_only"
            return elements, processing_info

        except Exception as e:
            self._logger.error(f"OCR-only processing failed: {e}")
            return [], {"strategy": "ocr_only", "success": False, "error": str(e)}

    def _process_with_fallback(
        self, file_path: str, languages: Optional[List[str]] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process PDF with multi-tier fallback system.

        Args:
            file_path: Path to the PDF file
            languages: List of languages for processing

        Returns:
            Tuple of (elements, processing_info)
        """
        # Try hi_res first
        elements, info = self._process_hires_strategy(file_path, languages)
        if elements and self._has_meaningful_content(elements):
            return elements, info

        # Try fast processing
        elements, info = self._process_fast_strategy(file_path, languages)
        if elements and self._has_meaningful_content(elements):
            return elements, info

        # Final fallback
        return self._process_pdfplumber_fallback(file_path, languages)

    def _process_pdfplumber_fallback(
        self, file_path: str, languages: Optional[List[str]] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Final fallback processing using pdfplumber.

        Args:
            file_path: Path to the PDF file
            languages: List of languages for processing

        Returns:
            Tuple of (elements, processing_info)
        """
        try:
            self._logger.info("Processing PDF with pdfplumber fallback")

            elements = partition_pdf(
                filename=file_path,
                strategy=PartitionStrategy.FAST,
                infer_table_structure=self.infer_table_structure,
                languages=languages,
                include_page_breaks=True,
                # Use pdfplumber explicitly
                pdf_processing_mode="pdfplumber",
            )

            processing_info = {
                "strategy": "pdfplumber_fallback",
                "ocr_enabled": False,
                "table_detection": self.infer_table_structure,
                "success": True,
            }

            self.ocr_used = False
            self.last_processing_strategy_used = "pdfplumber_fallback"
            return elements, processing_info

        except Exception as e:
            self._logger.error(f"pdfplumber fallback failed: {e}")
            return [], {
                "strategy": "pdfplumber_fallback",
                "success": False,
                "error": str(e),
            }

    def _extract_total_pages(self, elements: List[Any]) -> int:
        """
        Extract total page count from processed elements.

        Args:
            elements: List of document elements from unstructured

        Returns:
            Total number of pages found in the document
        """
        if not elements:
            return 0

        page_numbers = set()
        for element in elements:
            try:
                # Get page number from element metadata
                if hasattr(element, 'metadata') and hasattr(
                    element.metadata, 'page_number'
                ):
                    page_num = element.metadata.page_number
                    if page_num is not None:
                        page_numbers.add(int(page_num))

                # Also check for page_number attribute directly on element
                elif hasattr(element, 'page_number'):
                    page_num = element.page_number
                    if page_num is not None:
                        page_numbers.add(int(page_num))
            except (ValueError, TypeError, AttributeError) as e:
                self._logger.debug(f"Could not extract page number from element: {e}")
                continue

        if page_numbers:
            total_pages = max(page_numbers)
            self._logger.info(
                f"Extracted {len(page_numbers)} unique page numbers, total pages: {total_pages}"
            )
            return total_pages
        else:
            # Fallback: try to count page breaks
            page_break_elements = [
                elem
                for elem in elements
                if hasattr(elem, 'category') and elem.category == 'PageBreak'
            ]
            if page_break_elements:
                total_pages = len(page_break_elements) + 1
                self._logger.info(
                    f"Counted {len(page_break_elements)} page breaks, total pages: {total_pages}"
                )
                return total_pages
            else:
                self._logger.warning(
                    "Could not determine total page count from elements"
                )
                return 0

    def _is_text_based_pdf(self, file_path: str) -> bool:
        """
        Determine if PDF is text-based or image-based.

        Args:
            file_path: Path to the PDF file

        Returns:
            True if PDF is text-based, False if image-based
        """
        try:
            with pikepdf.open(file_path) as pdf:
                for page in pdf.pages:
                    # Check if page has text content
                    if "/Text" in page.get("/Resources", {}):
                        return True

                    # Try to extract some text
                    try:
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:  # Reasonable text threshold
                            return True
                    except Exception:
                        continue

            return False

        except Exception as e:
            self._logger.debug(f"Error determining PDF type: {e}")
            return False  # Assume image-based if we can't determine

    def _has_meaningful_content(self, elements: List[Any]) -> bool:
        """
        Check if extracted elements contain meaningful content.

        Args:
            elements: List of extracted elements

        Returns:
            True if elements contain meaningful content
        """
        if not elements:
            return False

        total_text = ""
        for element in elements:
            if hasattr(element, 'text') and element.text:
                total_text += element.text + " "

        # Check if we have sufficient text content
        clean_text = " ".join(total_text.split())
        return len(clean_text) > 100  # Minimum threshold for meaningful content

    def get_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about the PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing PDF information
        """
        if not self.validate_file(file_path):
            return {"error": "Invalid file"}

        try:
            info = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "file_extension": self.get_file_extension(file_path),
                "ocr_available": self.ocr.is_configured,
                "ocr_info": self.ocr.get_ocr_info(),
            }

            # Try to get PDF metadata
            try:
                with pikepdf.open(file_path) as pdf:
                    pdf_info = pdf.docinfo
                    if pdf_info:
                        info.update(
                            {
                                "title": str(pdf_info.get("/Title", "")),
                                "author": str(pdf_info.get("/Author", "")),
                                "creator": str(pdf_info.get("/Creator", "")),
                                "producer": str(pdf_info.get("/Producer", "")),
                                "creation_date": str(pdf_info.get("/CreationDate", "")),
                            }
                        )

                    info["page_count"] = len(pdf.pages)

                    # Check if PDF is text-based
                    info["is_text_based"] = self._is_text_based_pdf(file_path)

            except Exception as e:
                self._logger.debug(f"Error extracting PDF metadata: {e}")
                info["metadata_error"] = str(e)

            return info

        except Exception as e:
            return {"error": f"Error reading PDF: {e}"}

    def _override_element_metadata(
        self, elements: List[Any], original_filename: str
    ) -> None:
        """
        Override element metadata with original filename to fix temporary filename issue.

        Args:
            elements: List of document elements from unstructured
            original_filename: Original filename to use instead of temporary filename
        """
        if not original_filename:
            self._logger.warning("No original filename provided for metadata override")
            return

        self._logger.info(
            f"Overriding metadata for {len(elements)} elements with original filename: {original_filename}"
        )

        overridden_count = 0
        for i, element in enumerate(elements):
            if hasattr(element, 'metadata'):
                try:
                    old_filename = getattr(element.metadata, 'filename', None)
                    # Override the filename in element metadata
                    element.metadata.filename = original_filename
                    # Also update file_directory if it contains temp path info
                    element.metadata.file_directory = None  # Clear temp directory info

                    if old_filename != original_filename:
                        self._logger.debug(
                            f"Element {i}: Overrode filename from '{old_filename}' to '{original_filename}'"
                        )
                        overridden_count += 1

                except Exception as e:
                    self._logger.error(
                        f"Could not override metadata for element {i}: {e}"
                    )
            else:
                self._logger.debug(f"Element {i} has no metadata: {type(element)}")

        self._logger.info(
            f"Metadata override completed. Overrode {overridden_count} elements."
        )
