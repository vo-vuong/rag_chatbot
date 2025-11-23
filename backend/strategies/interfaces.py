"""
Strategy interfaces and abstract base classes for document processing.

This module defines the contracts that all document processing strategies must implement,
providing a clean, extensible architecture for document type support.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .results import ProcessingResult


class DocumentProcessingStrategy(ABC):
    """
    Abstract base class for document processing strategies.

    Implements the Strategy pattern to provide a common interface for
    processing different document types (PDF, CSV, images, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processing strategy.

        Args:
            config: Configuration dictionary for the strategy
        """
        self.config = config or {}
        self._logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the strategy instance."""
        return logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """
        Check if this strategy can process the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this strategy can process the file, False otherwise
        """

    @abstractmethod
    def extract_elements(self, file_path: str, **kwargs) -> ProcessingResult:
        """
        Extract structural elements from the document.

        Args:
            file_path: Path to the document file
            **kwargs: Additional parameters for extraction

        Returns:
            ProcessingResult containing extracted elements and metadata
        """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of supported file extensions (including the dot)
        """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        Get the name of this processing strategy.

        Returns:
            Human-readable name of the strategy
        """

    @property
    def strategy_version(self) -> str:
        """
        Get the version of this processing strategy.

        Returns:
            Version string for compatibility tracking
        """
        return "1.0.0"

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this processing strategy.

        Returns:
            Dictionary containing strategy information
        """
        return {
            "name": self.strategy_name,
            "version": self.strategy_version,
            "supported_extensions": self.supported_extensions,
            "config": self.config,
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
        }

    def validate_file(self, file_path: str) -> bool:
        """
        Validate that the file exists and is accessible.

        Args:
            file_path: Path to the file to validate

        Returns:
            True if file is valid and accessible, False otherwise
        """
        if not os.path.exists(file_path):
            self._logger.error(f"File does not exist: {file_path}")
            return False

        if not os.path.isfile(file_path):
            self._logger.error(f"Path is not a file: {file_path}")
            return False

        if not os.access(file_path, os.R_OK):
            self._logger.error(f"File is not readable: {file_path}")
            return False

        return True

    def get_file_extension(self, file_path: str) -> str:
        """
        Get file extension in lowercase.

        Args:
            file_path: Path to the file

        Returns:
            File extension in lowercase (including the dot)
        """
        return os.path.splitext(file_path)[1].lower()

    def supports_file(self, file_path: str) -> bool:
        """
        Check if the strategy supports the given file type.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file type is supported, False otherwise
        """
        extension = self.get_file_extension(file_path)
        return extension in self.supported_extensions

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file information
        """
        if not self.validate_file(file_path):
            return {"error": "Invalid file"}

        try:
            file_path_obj = Path(file_path)
            return {
                "file_path": file_path,
                "file_name": file_path_obj.name,
                "file_size": file_path_obj.stat().st_size,
                "file_extension": self.get_file_extension(file_path),
                "is_supported": self.supports_file(file_path),
                "is_readable": os.access(file_path, os.R_OK),
                "strategy_info": self.get_strategy_info(),
            }
        except Exception as e:
            return {"error": f"Error getting file info: {e}"}

    def _create_error_result(
        self,
        message: str,
        error_type: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None,
    ) -> ProcessingResult:
        """
        Create a standardized error result.

        Args:
            message: Error message
            error_type: Type of error
            error_details: Additional error details
            processing_time: Processing time before error

        Returns:
            ProcessingResult with error information
        """
        result = ProcessingResult(
            success=False,
            error_message=message,
            error_type=error_type or type(self).__name__,
            error_details=error_details,
            processing_time=processing_time,
            strategy_version=self.strategy_version,
        )

        result.metadata["strategy"] = self.strategy_name
        result.metadata["strategy_class"] = self.__class__.__name__

        return result

    def _create_success_result(
        self,
        elements: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None,
        **additional_metadata,
    ) -> ProcessingResult:
        """
        Create a standardized success result.

        Args:
            elements: Extracted elements
            metadata: Additional metadata
            processing_time: Processing time
            **additional_metadata: Additional metadata key-value pairs

        Returns:
            ProcessingResult with extracted elements
        """
        # Prepare metadata
        final_metadata = {
            "strategy": self.strategy_name,
            "strategy_class": self.__class__.__name__,
            "strategy_version": self.strategy_version,
            "elements_extracted": len(elements),
        }

        if metadata:
            final_metadata.update(metadata)

        if additional_metadata:
            final_metadata.update(additional_metadata)

        result = ProcessingResult(
            success=True,
            elements=elements,
            metadata=final_metadata,
            processing_time=processing_time,
            strategy_version=self.strategy_version,
        )

        # Update metrics
        result.update_metrics(
            elements_extracted=len(elements), strategy_used=self.strategy_name
        )

        return result
