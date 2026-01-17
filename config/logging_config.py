"""Logging configuration for RAG Chatbot."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Track if logging has been set up to prevent duplicate handlers
_logging_configured = False


def setup_logging(
    module_name: str,
    log_dir: str = "logs",
    log_file: str = "app.log",
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Configure logging with file rotation and console output.

    Args:
        module_name: Subdirectory for logs (e.g., "api", "ui")
        log_dir: Base logs directory
        log_file: Log file name
        level: Logging level
        max_bytes: Max file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured root logger
    """
    global _logging_configured

    root_logger = logging.getLogger()

    # Prevent adding duplicate handlers on repeated calls
    if _logging_configured:
        return root_logger

    # Create log directory
    log_path = Path(log_dir) / module_name
    log_path.mkdir(parents=True, exist_ok=True)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler with rotation (UTF-8 encoding for international characters)
    file_handler = RotatingFileHandler(
        log_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler with UTF-8 support for Windows
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # On Windows, configure stdout for Unicode
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _logging_configured = True
    return root_logger
