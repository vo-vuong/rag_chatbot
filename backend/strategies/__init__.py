"""
Document processing strategies package.

This package provides a pluggable architecture for processing different
document types with standardized interfaces and result handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Type

from .interfaces import DocumentProcessingStrategy
from .results import ProcessingMetrics, ProcessingResult, ProcessingStatus

__all__ = [
    # Interfaces
    "DocumentProcessingStrategy",
    # Results and data structures
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingMetrics",
]

# Strategy registry for dynamic loading
STRATEGY_REGISTRY: Dict[str, Type] = {}

# Try to import PDF strategy (may fail if dependencies missing)
try:
    from .pdf_strategy import PDFProcessingStrategy

    STRATEGY_REGISTRY[".pdf"] = PDFProcessingStrategy
    __all__.append("PDFProcessingStrategy")
except ImportError as e:
    # PDF strategy not available due to missing dependencies
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"PDF strategy not available: {e}")
    PDFProcessingStrategy = None


def get_strategy_for_file(
    file_path: str, config: Dict = None
) -> Optional[DocumentProcessingStrategy]:
    """Get appropriate strategy for a given file."""

    extension = Path(file_path).suffix.lower()
    strategy_class = STRATEGY_REGISTRY.get(extension)

    if strategy_class:
        return strategy_class(config=config)

    return None


def get_supported_extensions() -> List[str]:
    """Get list of all supported file extensions."""
    return list(STRATEGY_REGISTRY.keys())


def register_strategy(extension: str, strategy_class: Type):
    """Register a new strategy for a file extension."""
    STRATEGY_REGISTRY[extension.lower()] = strategy_class
