"""Document processing strategies package."""

from pathlib import Path
from typing import Dict, List, Optional, Type

from .csv_strategy import CSVProcessingStrategy
from .interfaces import DocumentProcessingStrategy
from .results import ProcessingMetrics, ProcessingResult, ProcessingStatus

__all__ = [
    # Interfaces
    "DocumentProcessingStrategy",
    # Strategies
    "CSVProcessingStrategy",
    # Results and data structures
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingMetrics",
]

# Strategy registry for dynamic loading
STRATEGY_REGISTRY: Dict[str, Type] = {
    ".csv": CSVProcessingStrategy,
}

# Import Docling PDF strategy (primary)
try:
    from .docling_pdf_strategy import DoclingPDFStrategy

    STRATEGY_REGISTRY[".pdf"] = DoclingPDFStrategy
    __all__.append("DoclingPDFStrategy")
except ImportError as e:
    # Docling strategy not available due to missing dependencies
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Docling PDF strategy not available: {e}")
    DoclingPDFStrategy = None


def get_strategy_for_file(
    file_path: str, config: Optional[Dict] = None
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
