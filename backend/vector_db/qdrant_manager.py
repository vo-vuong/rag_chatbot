"""
Qdrant Vector Database Manager.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config.constants import TEXT_COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantManager:
    """Manager for Qdrant vector database operations."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = TEXT_COLLECTION_NAME,
        timeout: int = 60,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name

        try:
            clean_host = self.host.replace("http://", "").replace("https://", "")

            self.client = QdrantClient(host=clean_host, port=self.port, timeout=timeout)
            logger.info(
                f"Qdrant client initialized: {clean_host}:{self.port}, "
                f"collection: {self.collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

    def is_healthy(self) -> bool:
        """Check if Qdrant server is healthy."""
        try:
            self.client.get_collections()
            logger.info("Qdrant server is healthy")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)
            logger.debug(f"Collection {self.collection_name} exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get collection information."""
        if not self.collection_exists():
            return None

        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection.points_count,
                "status": collection.status,
                "config": collection.config,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return None

    def ensure_collection(self, dimension: int, distance: str = "Cosine") -> bool:
        """Ensure collection exists with correct configuration."""
        try:
            if self.collection_exists():
                info = self.get_collection_info()
                if info:
                    current_dim = info['config'].params.vectors.size

                    if current_dim != dimension:
                        logger.warning(
                            f"Collection exists with dimension {current_dim}, "
                            f"requested {dimension}. Using existing collection."
                        )
                    else:
                        logger.info(
                            f"Collection {self.collection_name} exists "
                            f"with correct dimension {dimension}"
                        )
                return True

            # Create new collection
            logger.info(
                f"Creating collection {self.collection_name} with dimension {dimension}"
            )

            distance_map = {
                "Cosine": models.Distance.COSINE,
                "Euclid": models.Distance.EUCLID,
                "Dot": models.Distance.DOT,
            }
            distance_metric = distance_map.get(distance, models.Distance.COSINE)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=distance_metric
                ),
            )

            logger.info(f"Collection {self.collection_name} created successfully")
            return True

        except Exception as e:
            logger.error(f"Error ensuring collection: {str(e)}")
            return False

    def add_documents(
        self,
        chunks_df: pd.DataFrame,
        embeddings: List[List[float]],
        language: Optional[str] = None,
        source_file: str = "unknown",
        batch_size: int = 100,
    ) -> bool:
        """
        Add documents to Qdrant (no session_id).

        Args:
            chunks_df: DataFrame containing chunks and metadata
            embeddings: List of embedding vectors
            language: Language code ("en" or "vi") - optional
            source_file: Source file name
            batch_size: Batch size for uploading

        Returns:
            True if successful, False otherwise
        """
        if len(chunks_df) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks_df)} chunks but {len(embeddings)} embeddings"
            )

        try:
            logger.info(f"Adding {len(chunks_df)} documents to {self.collection_name}")

            points = []
            timestamp = datetime.now().isoformat()

            for idx, (_, row) in enumerate(chunks_df.iterrows()):
                point_id = str(uuid.uuid4())

                # Simple metadata (no session_id)
                payload = {
                    "chunk": row.get("chunk", ""),
                    "source_file": row.get(
                        "source_file", source_file
                    ),  # Use individual chunk source_file, fallback to parameter
                    "timestamp": timestamp,
                    "chunk_index": idx,
                }

                # Add language if provided
                if language:
                    payload["language"] = language

                # Add all other columns from dataframe
                for col in chunks_df.columns:
                    if col not in payload and col != "chunk":
                        value = row[col]
                        # Handle NaN values - use try/except for array ambiguity
                        try:
                            if pd.isna(value):
                                payload[col] = None
                                continue
                        except (ValueError, TypeError):
                            # Handle cases where pd.isna() can't evaluate the value (e.g., numpy arrays)
                            if value is None:
                                payload[col] = None
                                continue

                        if isinstance(value, (int, float, str, bool)):
                            payload[col] = value
                        elif isinstance(value, list):
                            # Handle lists like image_paths
                            payload[col] = value if value else []
                        else:
                            payload[col] = str(value)

                point = models.PointStruct(
                    id=point_id, vector=embeddings[idx], payload=payload
                )
                points.append(point)

            # Upload in batches
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                logger.debug(
                    f"Uploaded batch {i // batch_size + 1}: {len(batch)} points"
                )

            logger.info(f"Successfully added {len(points)} documents to Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {str(e)}")
            return False

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            language: Optional language filter (not used by default)

        Returns:
            List of search results with payload and score
        """
        try:
            logger.info(f"Searching in {self.collection_name} (top_k: {top_k})")

            # No filter by default (search all documents)
            search_filter = None

            # Optional: Can enable language filter if needed
            # if language:
            #     search_filter = models.Filter(
            #         must=[models.FieldCondition(
            #             key="language",
            #             match=models.MatchValue(value=language)
            #         )]
            #     )

            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=score_threshold,
            ).points

            results = []
            for hit in search_results:
                results.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "payload": hit.payload,
                    }
                )

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            return []

    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            if not self.collection_exists():
                logger.warning(f"Collection {self.collection_name} does not exist")
                return True

            logger.warning(f"Deleting collection: {self.collection_name}")
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
            return True

        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            if not self.collection_exists():
                return {"error": "Collection does not exist"}

            info = self.get_collection_info()
            if info is None:
                return {"error": "Failed to retrieve collection info"}

            return {
                "collection_name": self.collection_name,
                "total_documents": info.get("points_count", 0),
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with basic information."""
        try:
            collections_response = self.client.get_collections()
            collections_info = []

            for collection in collections_response.collections:
                try:
                    # Get detailed info for each collection
                    collection_info = self.client.get_collection(collection.name)
                    collections_info.append(
                        {
                            "name": collection.name,
                            "points_count": collection_info.points_count,
                            "status": (
                                str(collection_info.status)
                                if collection_info.status
                                else "unknown"
                            ),
                            "vectors_count": collection_info.vectors_count,
                            "segments_count": collection_info.segments_count,
                            "indexed_vectors_count": collection_info.indexed_vectors_count,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Error getting details for collection {collection.name}: {str(e)}"
                    )
                    collections_info.append(
                        {
                            "name": collection.name,
                            "points_count": 0,
                            "status": "unknown",
                            "error": str(e),
                            "vectors_count": 0,
                            "segments_count": 0,
                            "indexed_vectors_count": 0,
                        }
                    )

            logger.info(f"Found {len(collections_info)} collections")
            return collections_info

        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []

    def get_detailed_collection_info(
        self, collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific collection."""
        try:
            collection_info = self.client.get_collection(collection_name)

            # Extract vector configuration
            vector_config = {}
            if hasattr(collection_info.config, 'params') and hasattr(
                collection_info.config.params, 'vectors'
            ):
                if isinstance(collection_info.config.params.vectors, dict):
                    for name, config in collection_info.config.params.vectors.items():
                        vector_config[name] = {
                            "size": config.size,
                            "distance": (
                                config.distance.value
                                if hasattr(config.distance, 'value')
                                else str(config.distance)
                            ),
                        }
                else:
                    # Single vector configuration
                    vector_config["default"] = {
                        "size": collection_info.config.params.vectors.size,
                        "distance": (
                            collection_info.config.params.vectors.distance.value
                            if hasattr(
                                collection_info.config.params.vectors.distance, 'value'
                            )
                            else str(collection_info.config.params.vectors.distance)
                        ),
                    }

            return {
                "name": collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "segments_count": collection_info.segments_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": (
                    str(collection_info.status) if collection_info.status else "unknown"
                ),
                "vector_config": vector_config,
                "config": collection_info.config,
            }

        except Exception as e:
            logger.error(
                f"Error getting detailed collection info for {collection_name}: {str(e)}"
            )
            return None

    def create_collection(
        self, collection_name: str, dimension: int, distance: str = "Cosine"
    ) -> bool:
        """Create a new collection with specified configuration."""
        try:
            # Check if collection already exists
            if collection_name in [col["name"] for col in self.list_collections()]:
                logger.warning(f"Collection {collection_name} already exists")
                return False

            distance_map = {
                "Cosine": models.Distance.COSINE,
                "Euclid": models.Distance.EUCLID,
                "Dot": models.Distance.DOT,
            }
            distance_metric = distance_map.get(distance, models.Distance.COSINE)

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=distance_metric
                ),
            )

            logger.info(f"Collection {collection_name} created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {str(e)}")
            return False

    def delete_collection_by_name(self, collection_name: str) -> bool:
        """Delete a specific collection by name."""
        try:
            if collection_name not in [col["name"] for col in self.list_collections()]:
                logger.warning(f"Collection {collection_name} does not exist")
                return False

            logger.warning(f"Deleting collection: {collection_name}")
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False

    def get_collection_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int = 0,
        with_payload: bool = True,
        with_vectors: bool = False,
        start_from_offset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve points from a collection with pagination.

        Args:
            collection_name: Name of the collection
            limit: Maximum number of points to return
            offset: Number of points to skip (for simple pagination)
            with_payload: Whether to include payload data
            with_vectors: Whether to include vector data
            start_from_offset: Use Qdrant's scroll offset for proper pagination

        Returns:
            Dictionary containing points and metadata
        """
        try:
            logger.info(
                f"Retrieving points from {collection_name} (limit={limit}, offset={offset})"
            )

            # Use scroll to get points with pagination
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=start_from_offset,  # Use the scroll offset if provided
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            points = scroll_result[
                0
            ]  # scroll_result is a tuple (points, next_page_offset)
            next_page_offset = scroll_result[1]

            # Convert points to more readable format
            formatted_points = []
            for point in points:
                point_data = {
                    "id": str(point.id),
                    "payload": point.payload if with_payload else {},
                }

                if with_vectors and point.vector is not None:
                    point_data["vector_size"] = (
                        len(point.vector)
                        if isinstance(point.vector, list)
                        else "unknown"
                    )
                else:
                    point_data["vector_size"] = None

                formatted_points.append(point_data)

            result = {
                "points": formatted_points,
                "count": len(formatted_points),
                "has_more": next_page_offset is not None,
                "next_page_offset": next_page_offset,
                "collection_name": collection_name,
            }

            logger.info(
                f"Retrieved {len(formatted_points)} points from {collection_name}"
            )
            return result

        except Exception as e:
            logger.error(f"Error retrieving points from {collection_name}: {str(e)}")
            return {
                "points": [],
                "count": 0,
                "has_more": False,
                "next_page_offset": None,
                "collection_name": collection_name,
                "error": str(e),
            }

    def search_all_points(
        self,
        collection_name: str,
        limit: int = 1000,
        search_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for all points in a collection (useful for getting all data).

        Args:
            collection_name: Name of the collection
            limit: Maximum number of points to return
            search_filter: Optional filter to apply

        Returns:
            List of all points found
        """
        try:
            logger.info(f"Searching all points in {collection_name}")

            # Get collection info to determine vector dimension
            collection_info = self.get_detailed_collection_info(collection_name)
            if not collection_info:
                logger.error(f"Could not get collection info for {collection_name}")
                return []

            # Determine vector dimension from config
            vector_dim = 384  # default
            vector_config = collection_info.get('vector_config', {})
            if vector_config and 'default' in vector_config:
                vector_dim = vector_config['default'].get('size', 384)
            elif vector_config:
                # Use the first available vector config
                first_config = list(vector_config.values())[0]
                vector_dim = first_config.get('size', 384)

            # Use a simple search with a dummy vector to get all points
            dummy_vector = [0.1] * vector_dim
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=dummy_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,  # Don't need vectors for data table
                score_threshold=0.0,  # Get all points regardless of similarity
            ).points

            points = []
            for hit in search_result:
                point_data = {
                    "id": str(hit.id),
                    "payload": hit.payload,
                    "score": hit.score,
                }
                points.append(point_data)

            logger.info(f"Found {len(points)} points in {collection_name}")
            return points

        except Exception as e:
            logger.error(f"Error searching all points in {collection_name}: {str(e)}")
            return []
