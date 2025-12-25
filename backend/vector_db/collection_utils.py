"""Utilities for managing Qdrant collections."""

import logging
from typing import Dict, Optional

from backend.vector_db.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)


def cascade_delete_collections(
    text_collection: str,
    image_collection: str,
    host: str = "localhost",
    port: int = 6333
) -> bool:
    """
    Delete both text and image collections together.

    Args:
        text_collection: Name of text collection
        image_collection: Name of image collection
        host: Qdrant host
        port: Qdrant port

    Returns:
        True if both deleted successfully
    """
    try:
        text_manager = QdrantManager(
            collection_name=text_collection,
            host=host,
            port=port
        )

        image_manager = QdrantManager(
            collection_name=image_collection,
            host=host,
            port=port
        )

        # Delete text collection
        if text_manager.collection_exists():
            text_manager.delete_collection()
            logger.info(f"Deleted text collection: {text_collection}")

        # Delete image collection
        if image_manager.collection_exists():
            image_manager.delete_collection()
            logger.info(f"Deleted image collection: {image_collection}")

        return True

    except Exception as e:
        logger.error(f"Failed to cascade delete collections: {e}")
        return False


def get_collection_stats(
    collection_name: str,
    host: str = "localhost",
    port: int = 6333
) -> Optional[Dict[str, any]]:
    """
    Get statistics for a collection.

    Returns:
        Dict with point_count, vector_size, etc.
    """
    try:
        manager = QdrantManager(
            collection_name=collection_name,
            host=host,
            port=port
        )

        if not manager.collection_exists():
            return None

        # Get collection info from Qdrant
        info = manager.client.get_collection(collection_name)

        return {
            "name": collection_name,
            "point_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance.name
        }

    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return None
