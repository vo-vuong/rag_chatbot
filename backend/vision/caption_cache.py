"""
Caption caching to avoid re-captioning duplicate images.

Uses in-memory dict cache with MD5 hash keys. Optionally can be
extended to use SQLite or Redis for persistence.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class CaptionCache:
    """In-memory cache for image captions keyed by MD5 hash."""

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[str, Dict[str, any]] = {}
        logger.info("CaptionCache initialized (in-memory)")

    def get(self, image_path: str) -> Optional[str]:
        """
        Get cached caption for image.

        Args:
            image_path: Path to image file

        Returns:
            Cached caption or None if not found
        """
        image_hash = self._compute_hash(image_path)
        if image_hash in self._cache:
            cached = self._cache[image_hash]
            logger.info(f"Cache HIT for {Path(image_path).name} (hash: {image_hash[:8]})")
            return cached["caption"]
        logger.debug(f"Cache MISS for {Path(image_path).name}")
        return None

    def set(self, image_path: str, caption: str, cost: float = 0.0) -> None:
        """
        Cache caption for image.

        Args:
            image_path: Path to image file
            caption: Generated caption
            cost: API cost for this caption
        """
        image_hash = self._compute_hash(image_path)
        self._cache[image_hash] = {
            "caption": caption,
            "cost": cost,
            "image_path": image_path
        }
        logger.info(f"Cached caption for {Path(image_path).name} (hash: {image_hash[:8]})")

    def clear(self) -> None:
        """Clear all cached captions."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared ({count} entries removed)")

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self._cache),
            "total_cost_saved": sum(
                entry["cost"] for entry in self._cache.values()
            )
        }

    def _compute_hash(self, image_path: str) -> str:
        """Compute MD5 hash of image file."""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {image_path}: {e}")
            return ""
