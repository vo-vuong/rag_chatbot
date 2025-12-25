"""
Vision module for image captioning using GPT-4o Mini Vision API.

Provides ImageCaptioner service for generating brief, semantic descriptions
of images extracted from PDFs. Supports parallel processing, retry logic,
and cost tracking.
"""

from .image_captioner import ImageCaptioner, ImageCaptioningError

__all__ = ["ImageCaptioner", "ImageCaptioningError"]
