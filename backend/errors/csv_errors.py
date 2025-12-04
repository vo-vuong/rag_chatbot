"""
CSV processing error handling with specific error types and recovery strategies.

This module provides comprehensive error handling for CSV processing operations,
including detailed error messages, actionable suggestions, and recovery strategies.
"""

from typing import Any, Dict, List, Optional


class CSVProcessingError(Exception):
    """Base exception for CSV processing errors."""

    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.suggestions = suggestions or []
        self.error_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error_type": self.error_type,
            "message": str(self),
            "suggestions": self.suggestions,
        }


class CSVFileTooLargeError(CSVProcessingError):
    """Raised when CSV file exceeds size limits."""

    def __init__(self, file_size_mb: float, max_size_mb: float):
        message = f"CSV file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
        suggestions = [
            f"Split the file into smaller files (max {max_size_mb}MB each)",
            "Remove unnecessary columns to reduce file size",
            "Use data compression if possible",
            "Consider using streaming mode for large datasets",
        ]
        super().__init__(message, suggestions)
        self.file_size_mb = file_size_mb
        self.max_size_mb = max_size_mb


class CSVMemoryError(CSVProcessingError):
    """Raised when CSV processing runs out of memory."""

    def __init__(
        self, file_size_mb: float, available_memory_mb: Optional[float] = None
    ):
        message = f"Insufficient memory to process CSV file ({file_size_mb:.1f}MB)"
        if available_memory_mb:
            message += f" (available: {available_memory_mb:.1f}MB)"

        suggestions = [
            "Close other applications to free memory",
            "Try processing a smaller subset of the data",
            "Restart the application to clear memory",
            "Use streaming processing mode for large files",
            "Increase system RAM if possible",
        ]
        super().__init__(message, suggestions)
        self.file_size_mb = file_size_mb
        self.available_memory_mb = available_memory_mb


class CSVParsingError(CSVProcessingError):
    """Raised when CSV file cannot be parsed."""

    def __init__(self, parsing_error: str, line_number: Optional[int] = None):
        message = f"CSV parsing error: {parsing_error}"
        if line_number:
            message += f" (around line {line_number})"

        suggestions = [
            "Check if the file is a valid CSV format",
            "Verify the delimiter character (comma, semicolon, tab)",
            "Try opening the file in a spreadsheet application first",
            "Check for special characters or encoding issues",
            "Ensure consistent quote usage throughout the file",
            "Validate that all rows have the same number of columns",
        ]
        super().__init__(message, suggestions)
        self.parsing_error = parsing_error
        self.line_number = line_number


class CSVColumnError(CSVProcessingError):
    """Raised when selected columns are invalid."""

    def __init__(self, invalid_columns: List[str], available_columns: List[str]):
        message = f"Invalid columns selected: {', '.join(invalid_columns)}"
        suggestions = [
            f"Available columns: {', '.join(available_columns)}",
            "Check for exact column name spelling",
            "Remove any extra spaces from column names",
            "Ensure column names are case-sensitive",
            "Verify that column names don't contain special characters",
        ]
        super().__init__(message, suggestions)
        self.invalid_columns = invalid_columns
        self.available_columns = available_columns


class CSVConfigurationError(CSVProcessingError):
    """Raised when processing configuration is invalid."""

    def __init__(self, config_issue: str, config_value: Any = None):
        message = f"Invalid configuration: {config_issue}"
        if config_value is not None:
            message += f" (value: {config_value})"

        suggestions = [
            "Check processing parameter values",
            "Ensure numeric values are within reasonable ranges",
            "Verify column names exist in the CSV file",
            "Review the configuration documentation",
            "Use default values if unsure",
        ]
        super().__init__(message, suggestions)
        self.config_issue = config_issue
        self.config_value = config_value


class CSVEncodingError(CSVProcessingError):
    """Raised when CSV file encoding cannot be detected or processed."""

    def __init__(self, file_path: str, attempted_encodings: Optional[List[str]] = None):
        message = f"Unable to read CSV file with supported encodings: {file_path}"
        if attempted_encodings:
            message += f" (attempted: {', '.join(attempted_encodings)})"

        suggestions = [
            "Save the file with UTF-8 encoding",
            "Try converting the file to a different encoding",
            "Check for special characters in the file",
            "Use a text editor to verify file encoding",
            "Remove BOM (Byte Order Mark) if present",
        ]
        super().__init__(message, suggestions)
        self.file_path = file_path
        self.attempted_encodings = attempted_encodings or []


class CSVEmptyFileError(CSVProcessingError):
    """Raised when CSV file is empty or contains no data."""

    def __init__(self, file_path: str):
        message = f"CSV file is empty or contains no data: {file_path}"
        suggestions = [
            "Verify that the file contains data",
            "Check if the file has headers but no data rows",
            "Ensure the file is not corrupted",
            "Try opening the file in a spreadsheet application",
        ]
        super().__init__(message, suggestions)
        self.file_path = file_path


class CSVStreamingError(CSVProcessingError):
    """Raised when streaming processing fails."""

    def __init__(self, streaming_error: str, file_size_mb: Optional[float] = None):
        message = f"Streaming processing failed: {streaming_error}"
        if file_size_mb:
            message += f" (file: {file_size_mb:.1f}MB)"

        suggestions = [
            "Try using standard processing mode instead",
            "Reduce chunk size for streaming",
            "Check available disk space for temporary files",
            "Verify file integrity and permissions",
            "Process a smaller subset of the data",
        ]
        super().__init__(message, suggestions)
        self.streaming_error = streaming_error
        self.file_size_mb = file_size_mb


class CSVDuplicateColumnError(CSVProcessingError):
    """Raised when CSV contains duplicate column names."""

    def __init__(self, duplicate_columns: List[str]):
        message = f"Duplicate columns found: {', '.join(duplicate_columns)}"
        suggestions = [
            "Rename duplicate columns in the source file",
            "Use a spreadsheet application to fix column names",
            "Add prefixes or suffixes to make columns unique",
            "Remove duplicate columns if they contain the same data",
        ]
        super().__init__(message, suggestions)
        self.duplicate_columns = duplicate_columns


class CSVDataValidationError(CSVProcessingError):
    """Raised when CSV data validation fails."""

    def __init__(self, validation_errors: List[str]):
        message = f"Data validation failed: {'; '.join(validation_errors[:3])}"
        if len(validation_errors) > 3:
            message += f" (and {len(validation_errors) - 3} more issues)"

        suggestions = [
            "Clean the data before processing",
            "Remove or correct invalid values",
            "Ensure consistent data formats",
            "Handle missing values appropriately",
            "Validate data against expected schema",
        ]
        super().__init__(message, suggestions)
        self.validation_errors = validation_errors


class CSVErrorRecovery:
    """Provides recovery strategies for CSV processing errors."""

    @staticmethod
    def get_recovery_strategy(error: CSVProcessingError) -> Dict[str, Any]:
        """
        Get recovery strategy for a specific CSV error.

        Returns:
            Dictionary containing recovery actions and fallback options
        """
        recovery_strategies = {
            CSVFileTooLargeError: {
                "primary_action": "reduce_file_size",
                "fallback_actions": ["enable_streaming", "sample_data"],
                "auto_recovery": False,
                "user_intervention_required": True,
            },
            CSVMemoryError: {
                "primary_action": "enable_streaming",
                "fallback_actions": ["reduce_chunk_size", "sample_data"],
                "auto_recovery": True,
                "user_intervention_required": False,
            },
            CSVParsingError: {
                "primary_action": "try_different_delimiter",
                "fallback_actions": ["skip_problem_rows", "manual_cleaning"],
                "auto_recovery": True,
                "user_intervention_required": False,
            },
            CSVColumnError: {
                "primary_action": "validate_columns",
                "fallback_actions": ["use_available_columns", "auto_correct_names"],
                "auto_recovery": True,
                "user_intervention_required": False,
            },
            CSVEncodingError: {
                "primary_action": "try_encodings",
                "fallback_actions": ["user_specified_encoding", "file_conversion"],
                "auto_recovery": True,
                "user_intervention_required": True,
            },
        }

        error_type = type(error)
        strategy = recovery_strategies.get(
            error_type,
            {
                "primary_action": "generic_error_handling",
                "fallback_actions": ["log_and_continue"],
                "auto_recovery": False,
                "user_intervention_required": True,
            },
        )

        return {
            **strategy,
            "error_type": error.error_type,
            "error_message": str(error),
            "suggestions": error.suggestions,
        }

    @staticmethod
    def can_auto_recover(error: CSVProcessingError) -> bool:
        """Check if an error can be automatically recovered."""
        strategy = CSVErrorRecovery.get_recovery_strategy(error)
        return strategy["auto_recovery"]

    @staticmethod
    def requires_user_intervention(error: CSVProcessingError) -> bool:
        """Check if an error requires user intervention."""
        strategy = CSVErrorRecovery.get_recovery_strategy(error)
        return strategy["user_intervention_required"]


def create_csv_error(error_type: str, **kwargs) -> CSVProcessingError:
    """
    Factory function to create CSV errors with proper parameters.

    Args:
        error_type: Type of error to create
        **kwargs: Error-specific parameters

    Returns:
        CSVProcessingError instance
    """
    error_classes = {
        "file_too_large": CSVFileTooLargeError,
        "memory_error": CSVMemoryError,
        "parsing_error": CSVParsingError,
        "column_error": CSVColumnError,
        "configuration_error": CSVConfigurationError,
        "encoding_error": CSVEncodingError,
        "empty_file": CSVEmptyFileError,
        "streaming_error": CSVStreamingError,
        "duplicate_column": CSVDuplicateColumnError,
        "data_validation": CSVDataValidationError,
    }

    error_class = error_classes.get(error_type, CSVProcessingError)
    return error_class(**kwargs)
