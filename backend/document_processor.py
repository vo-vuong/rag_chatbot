"""
Main document processor orchestrator.

This module provides the main interface for document processing, coordinating
different processing strategies and managing the overall processing pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config.constants import DOCLING_CONFIG

from .strategies.csv_strategy import CSVProcessingStrategy
from .strategies.docling_document_strategy import DoclingDocumentStrategy
from .strategies.interfaces import DocumentProcessingStrategy
from .strategies.results import ProcessingResult

# Configure logging
logger = logging.getLogger(__name__)

# Supported file types and their corresponding strategies
SUPPORTED_FILE_TYPES: Dict[str, type[DocumentProcessingStrategy]] = {
    ".pdf": DoclingDocumentStrategy,
    ".docx": DoclingDocumentStrategy,
}

# Default processing configurations
DEFAULT_CONFIG = {
    "pdf": {
        "strategy": "auto",  # auto, fast, hi_res, ocr_only, fallback
        "infer_table_structure": True,
        "extract_images": True,
        "chunk_after_extraction": True,
        "image_storage_path": "extracted_images",  # Path for storing extracted images
    },
    "ocr": {
        "languages": ["en", "vi"],  # Default OCR languages
    },
    "docling": DOCLING_CONFIG,  # Docling PDF processing and chunking config
}


class DocumentProcessor:
    """
    Main document processor that orchestrates different processing strategies.

    This class provides a unified interface for processing different document types
    using appropriate strategies, with support for fallback mechanisms, error handling,
    and integration with the existing RAG chatbot architecture.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document processor.

        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or DEFAULT_CONFIG
        self.strategies: Dict[str, DocumentProcessingStrategy] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize processing strategies
        self._initialize_strategies()

        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "strategies_used": {},
        }

        self.logger.info("Document processor initialized")

    def _initialize_strategies(self) -> None:
        """Initialize document processing strategies."""
        try:
            pdf_config = self.config.get("pdf", {})
            ocr_config = self.config.get("ocr", {})

            # Initialize Docling strategy
            self._init_docling_strategy(pdf_config, ocr_config)

            # Initialize CSV strategy
            self._init_csv_strategy()

        except Exception as e:
            self.logger.error(f"Failed to initialize strategies: {e}")
            raise

    def _init_docling_strategy(self, pdf_config: dict, ocr_config: dict) -> None:
        """Initialize Docling document strategy for PDF and DOCX."""
        # Merge config with defaults
        docling_config = {**DOCLING_CONFIG, **self.config.get("docling", {})}

        # Override OCR languages from ocr_config if provided
        if "languages" in ocr_config:
            docling_config["ocr"]["languages"] = ocr_config["languages"]

        # Create Docling strategy
        docling_strategy = DoclingDocumentStrategy(
            config={
                **self.config,
                "docling": docling_config,
            }
        )

        # Register for both PDF and DOCX
        self.strategies[".pdf"] = docling_strategy
        self.strategies[".docx"] = docling_strategy
        self.logger.info("Docling document strategy initialized for PDF and DOCX")

    def _init_csv_strategy(self) -> None:
        """Initialize CSV strategy."""
        try:
            csv_config = self.config.get("csv", {})
            self.strategies[".csv"] = CSVProcessingStrategy(config=csv_config)
            SUPPORTED_FILE_TYPES[".csv"] = CSVProcessingStrategy
            self.logger.info("CSV strategy initialized")
        except Exception as e:
            self.logger.warning(f"CSV strategy unavailable: {e}")

    def process_document(
        self,
        file_path: Union[str, Path],
        strategy_name: Optional[str] = None,
        languages: Optional[List[str]] = None,
        original_filename: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process a document using the appropriate strategy.

        Args:
            file_path: Path to the document file
            strategy_name: Optional specific strategy to use
            languages: List of languages for processing
            original_filename: Original filename to override metadata
            openai_api_key: OpenAI API key for embeddings/vision
            **kwargs: Additional parameters for processing

        Returns:
            ProcessingResult containing processed elements and metadata
        """
        file_path = str(file_path)
        file_extension = self._get_file_extension(file_path).lower()

        self.logger.info(f"Processing document: {file_path}")

        try:
            # Update processing statistics
            self.processing_stats["total_processed"] += 1

            # Validate file
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                self.logger.error(error_msg)
                self.processing_stats["failed_processed"] += 1
                return ProcessingResult(success=False, error_message=error_msg)

            if not os.path.isfile(file_path):
                error_msg = f"Path is not a file: {file_path}"
                self.logger.error(error_msg)
                self.processing_stats["failed_processed"] += 1
                return ProcessingResult(success=False, error_message=error_msg)

            # Find appropriate strategy
            strategy = self._get_strategy(file_extension, strategy_name)
            if not strategy:
                error_msg = (
                    f"No processing strategy available for file type: {file_extension}"
                )
                self.logger.error(error_msg)
                self.processing_stats["failed_processed"] += 1
                return ProcessingResult(success=False, error_message=error_msg)

            # Pass openai_api_key to strategy if needed (for re-initialization with API key)
            if openai_api_key and file_extension in [".pdf", ".docx"]:
                # Update config with API key
                updated_config = {**self.config, "openai_api_key": openai_api_key}
                updated_config["pdf"] = {
                    **updated_config.get("pdf", {}),
                    "openai_api_key": openai_api_key,
                }

                # Merge Docling config
                docling_config = {**DOCLING_CONFIG, **self.config.get("docling", {})}
                ocr_config = self.config.get("ocr", {})
                if "languages" in ocr_config:
                    docling_config["ocr"]["languages"] = ocr_config["languages"]

                updated_config["docling"] = docling_config

                # Create fresh Docling strategy with API key
                strategy = DoclingDocumentStrategy(config=updated_config)

            # Process document with strategy
            result = strategy.extract_elements(
                file_path,
                languages=languages,
                original_filename=original_filename,
                **kwargs,
            )

            # Update statistics
            if result.success:
                self.processing_stats["successful_processed"] += 1
                strategy_name = strategy.__class__.__name__
                self.processing_stats["strategies_used"][strategy_name] = (
                    self.processing_stats["strategies_used"].get(strategy_name, 0) + 1
                )

                # Add processor metadata
                result.metadata["processor"] = {
                    "file_extension": file_extension,
                    "strategy_used": strategy.__class__.__name__,
                    "processor_version": "1.0.0",
                    "supported_file_types": list(SUPPORTED_FILE_TYPES.keys()),
                }

                self.logger.info(
                    f"Document processed successfully: {len(result.elements)} elements extracted"
                )
            else:
                self.processing_stats["failed_processed"] += 1
                self.logger.error(f"Document processing failed: {result.error_message}")

            return result

        except Exception as e:
            error_msg = f"Unexpected error processing document: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.processing_stats["failed_processed"] += 1
            return ProcessingResult(success=False, error_message=error_msg)

    def get_supported_file_types(self) -> List[str]:
        """
        Get list of supported file types.

        Returns:
            List of supported file extensions
        """
        return list(SUPPORTED_FILE_TYPES.keys())

    def can_process_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the processor can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file can be processed, False otherwise
        """
        file_path = str(file_path)
        file_extension = self._get_file_extension(file_path).lower()

        # Check if file type is supported
        if file_extension not in SUPPORTED_FILE_TYPES:
            return False

        # Check if strategy is available and can process the file
        strategy = self.strategies.get(file_extension)
        if strategy:
            return strategy.can_process(file_path)

        return False

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file information
        """
        file_path = str(file_path)
        file_extension = self._get_file_extension(file_path).lower()

        # Basic file information
        info = {
            "file_path": file_path,
            "file_extension": file_extension,
            "is_supported": self.can_process_file(file_path),
            "file_size": None,
            "file_exists": os.path.exists(file_path),
        }

        if info["file_exists"] and os.path.isfile(file_path):
            info["file_size"] = os.path.getsize(file_path)

            # Get strategy-specific information
            strategy = self.strategies.get(file_extension)
            if strategy and hasattr(strategy, 'get_pdf_info'):
                try:
                    strategy_info = getattr(strategy, 'get_pdf_info')(file_path)
                    info.update(strategy_info)
                except Exception as e:
                    info["strategy_info_error"] = str(e)

        return info

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            **self.processing_stats,
            "success_rate": (
                self.processing_stats["successful_processed"]
                / max(1, self.processing_stats["total_processed"])
            )
            * 100,
            "configured_strategies": list(self.strategies.keys()),
        }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "strategies_used": {},
        }
        self.logger.info("Processing statistics reset")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update processor configuration.

        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)

        # Reinitialize strategies if configuration changed significantly
        if any(key in new_config for key in ["pdf", "ocr", "chunking"]):
            self.logger.info("Configuration updated, reinitializing strategies")
            self._initialize_strategies()

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get current configuration information.

        Returns:
            Dictionary containing configuration details
        """
        config_info = {
            "supported_file_types": self.get_supported_file_types(),
            "current_config": self.config,
            "strategies": {
                ext: strategy.get_strategy_info()
                for ext, strategy in self.strategies.items()
            },
        }

        # OCR is handled by Docling's EasyOCR - no separate OCR config needed
        config_info["ocr_info"] = {"engine": "easyocr", "provider": "docling"}

        return config_info

    def _get_file_extension(self, file_path: str) -> str:
        """
        Get file extension in lowercase.

        Args:
            file_path: Path to the file

        Returns:
            File extension including the dot
        """
        return os.path.splitext(file_path)[1].lower()

    def _get_strategy(
        self, file_extension: str, strategy_name: Optional[str] = None
    ) -> Optional[DocumentProcessingStrategy]:
        """
        Get processing strategy for the given file type.

        Args:
            file_extension: File extension
            strategy_name: Optional specific strategy name

        Returns:
            Processing strategy or None if not available
        """
        # If specific strategy name is provided, try to use it
        if strategy_name:
            for strategy in self.strategies.values():
                if (
                    hasattr(strategy, 'strategy_name')
                    and strategy.strategy_name == strategy_name
                ):
                    return strategy

        # Return default strategy for file type
        return self.strategies.get(file_extension)
