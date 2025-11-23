"""
PDF processing strategy using unstructured with OCR integration.

This module provides comprehensive PDF processing capabilities with multi-tier fallback
system, OCR integration, and semantic chunking support.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pikepdf
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

from ..chunking.semantic_chunker import SemanticChunker
from ..ocr.tesseract_ocr import get_tesseract_ocr
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

        # Processing state
        self.last_processing_strategy_used = None
        self.ocr_used = False

        self._logger.info(
            f"PDF processing strategy initialized with OCR available: {self.ocr.is_configured}"
        )

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

            # Apply semantic chunking if requested
            if self.chunk_after_extraction and elements:
                chunk_result = self.chunker.chunk_elements(elements, languages)
                elements = chunk_result.chunks
                processing_info["chunking_stats"] = chunk_result.stats

            # Calculate processing time
            processing_time = time.time() - start_time

            # Extract total pages from elements
            total_pages = self._extract_total_pages(elements)

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
                **processing_info,
            }

            self._logger.info(
                f"PDF processing completed in {processing_time:.2f}s: "
                f"{len(elements)} elements extracted using {self.last_processing_strategy_used}"
            )

            return ProcessingResult(
                success=True,
                elements=elements,
                metadata=metadata,
                processing_time=processing_time,
            )

        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            self._logger.error(error_msg, exc_info=True)

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
