import logging
from typing import List

import cohere
from cohere import ClassifyExample

from backend.models import ChunkElement
from backend.reranking.base import RerankerStrategy

logger = logging.getLogger(__name__)


class CohereReranker(RerankerStrategy):
    """
    Reranker implementation using Cohere API.
    """

    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0"):
        """
        Initialize Cohere Reranker.

        Args:
            api_key: Cohere API Key.
            model: Model name to use (default: rerank-multilingual-v3.0).
        """
        if not api_key:
            raise ValueError("Cohere API Key is required for CohereReranker")
        
        self.client = cohere.Client(api_key=api_key)
        self.model = model
        logger.info(f"Initialized CohereReranker with model: {model}")

    def rerank(
        self, query: str, documents: List[ChunkElement], top_k: int
    ) -> List[ChunkElement]:
        """
        Rerank documents using Cohere API.
        """
        if not documents:
            return []

        # Prepare documents for Cohere (list of strings)
        doc_contents = [doc.content for doc in documents]

        try:
            # Call Cohere Rerank API
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_contents,
                top_n=top_k,
            )

            # Map results back to ChunkElement objects
            reranked_docs = []
            for result in response.results:
                # result.index is the index in the original list
                original_doc = documents[result.index]
                
                # Update score with rerank score
                reranked_doc = ChunkElement(
                    content=original_doc.content,
                    score=result.relevance_score,  # Update with Cohere score
                    source_file=original_doc.source_file,
                    page_number=original_doc.page_number,
                    element_type=original_doc.element_type,
                    metadata=original_doc.metadata,
                    point_id=original_doc.point_id,
                )
                reranked_docs.append(reranked_doc)

            logger.info(f"Reranked {len(documents)} documents to {len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"Cohere Rerank API failed: {e}")
            # Fallback strategy: Raise error as requested by user
            raise RuntimeError(f"Reranking failed: {e}")
