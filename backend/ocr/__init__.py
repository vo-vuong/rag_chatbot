"""
OCR (Optical Character Recognition) integration package.

This package provides OCR functionality with cross-platform support and
automatic configuration detection.
"""

from .tesseract_ocr import TesseractOCR, is_ocr_available, get_tesseract_ocr

__all__ = [
    "TesseractOCR",
    "is_ocr_available",
    "get_tesseract_ocr",
]