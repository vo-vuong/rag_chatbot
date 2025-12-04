"""
Backend error handling modules.

This package contains error handling utilities for different types of processing
operations in the backend system.
"""

from .csv_errors import (
    CSVColumnError,
    CSVConfigurationError,
    CSVDataValidationError,
    CSVDuplicateColumnError,
    CSVEmptyFileError,
    CSVEncodingError,
    CSVErrorRecovery,
    CSVFileTooLargeError,
    CSVMemoryError,
    CSVParsingError,
    CSVProcessingError,
    CSVStreamingError,
    create_csv_error,
)

__all__ = [
    "CSVProcessingError",
    "CSVFileTooLargeError",
    "CSVMemoryError",
    "CSVParsingError",
    "CSVColumnError",
    "CSVConfigurationError",
    "CSVEncodingError",
    "CSVEmptyFileError",
    "CSVStreamingError",
    "CSVDuplicateColumnError",
    "CSVDataValidationError",
    "CSVErrorRecovery",
    "create_csv_error",
]
