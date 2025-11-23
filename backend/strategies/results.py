"""
Result classes and data structures for document processing.

This module provides standardized result objects that encapsulate the outcomes
of document processing operations, including success status, extracted data,
metadata, and error information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessingStatus(Enum):
    """Enumeration of processing status types."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class ProcessingMetrics:
    """Processing performance and quality metrics."""

    processing_time: float = 0.0
    elements_extracted: int = 0
    characters_extracted: int = 0
    pages_processed: int = 0
    ocr_used: bool = False
    strategy_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "processing_time": self.processing_time,
            "elements_extracted": self.elements_extracted,
            "characters_extracted": self.characters_extracted,
            "pages_processed": self.pages_processed,
            "ocr_used": self.ocr_used,
            "strategy_used": self.strategy_used,
        }


@dataclass
class ProcessingResult:
    """
    Standardized result object for document processing operations.

    Encapsulates all outcomes of a processing operation including
    extracted elements, metadata, metrics, and error information.
    """

    # Core result data
    success: bool
    elements: List[Any] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.SUCCESS

    # Metadata and metrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)

    # Error handling
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Timing and versioning
    processing_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_version: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.error_message:
            self.status = ProcessingStatus.FAILED
            self.success = False
        elif not self.success:
            self.status = ProcessingStatus.FAILED

        # Update processing time in metrics
        if self.processing_time is not None:
            self.metrics.processing_time = self.processing_time

    @property
    def element_count(self) -> int:
        """Get number of extracted elements."""
        return len(self.elements)

    @property
    def has_content(self) -> bool:
        """Check if result contains meaningful content."""
        return self.success and self.element_count > 0

    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.status == ProcessingStatus.FAILED

    def add_error(self, message: str, error_type: str = None, details: Dict = None):
        """Add error information to the result."""
        self.error_message = message
        self.error_type = error_type or type(self).__name__
        self.error_details = details or {}
        self.success = False
        self.status = ProcessingStatus.FAILED

    def update_metrics(self, **kwargs):
        """Update processing metrics."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format for serialization."""
        return {
            "success": self.success,
            "status": self.status.value,
            "element_count": self.element_count,
            "has_content": self.has_content,
            "metadata": self.metadata,
            "metrics": self.metrics.to_dict(),
            "error_message": self.error_message,
            "error_type": self.error_type,
            "error_details": self.error_details,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
            "strategy_version": self.strategy_version,
        }

    def __repr__(self) -> str:
        """String representation of the processing result."""
        status = self.status.value.upper()
        elements_count = len(self.elements)
        processing_time = (
            f"{self.processing_time:.2f}s" if self.processing_time else "N/A"
        )

        return (
            f"ProcessingResult(status={status}, "
            f"elements_count={elements_count}, "
            f"processing_time={processing_time})"
        )
