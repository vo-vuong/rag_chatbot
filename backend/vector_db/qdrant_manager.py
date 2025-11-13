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

from config.constants import QDRANT_COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantManager:
    """Manager for Qdrant vector database operations."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = QDRANT_COLLECTION_NAME,
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
                    "source_file": source_file,
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
                        if pd.isna(value):
                            payload[col] = None
                        elif isinstance(value, (int, float, str, bool)):
                            payload[col] = value
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

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )

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
