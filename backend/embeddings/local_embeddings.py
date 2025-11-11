"""
Local Embeddings Strategy - PLACEHOLDER.

This module provides a placeholder for local embedding implementations
using Sentence Transformers or other local models. This allows the codebase
to be structured properly for future implementation.

TODO: Implement actual local embedding functionality when needed.

Future Usage:
    from backend.embeddings.local_embeddings import LocalEmbeddingStrategy

    strategy = LocalEmbeddingStrategy(model_name="all-MiniLM-L6-v2")
    embeddings = strategy.embed_texts(["Hello", "World"])
"""

import logging
from typing import List

from backend.embeddings.embedding_strategy import EmbeddingStrategy

# from config.constants import LOCAL_EMBEDDING_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalEmbeddingStrategy(EmbeddingStrategy):
    """
    Local embedding strategy - PLACEHOLDER.

    This class provides a structure for local embeddings but does not
    implement the actual functionality yet. It will raise NotImplementedError
    when methods are called.

    Future Implementation:
        - Use sentence-transformers library
        - Support CPU and GPU inference
        - Support multiple models (MiniLM, MPNet, multilingual, etc.)
        - Batch processing optimization
        - Model caching

    Attributes:
        model_name: Name of the local model to use
        device: Device to run model on ('cpu' or 'cuda')
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize local embedding strategy (placeholder).

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self._model = None

        logger.warning(
            "LocalEmbeddingStrategy initialized as PLACEHOLDER for model: %s",
            model_name,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts - NOT IMPLEMENTED.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "Local embeddings are not yet implemented. "
            "Please use OpenAI embeddings for now. "
            "\n\nTo implement this feature, you need to:"
            "\n1. Install sentence-transformers: pip install sentence-transformers"
            "\n2. Implement model loading and inference logic"
            "\n3. Add GPU support if needed"
            f"\n\nModel requested: {self.model_name}"
        )

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query - NOT IMPLEMENTED.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "Local embeddings are not yet implemented. "
            "Please use OpenAI embeddings for now."
        )

    def get_dimension(self) -> int:
        """
        Get embedding dimension - PLACEHOLDER.

        Returns:
            Expected dimension for the model (from constants)

        Raises:
            NotImplementedError: If model is not in known configs
        """
        # Return expected dimensions for known models
        dimension_map = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "keepitreal/vietnamese-sbert": 768,
        }

        if self.model_name in dimension_map:
            return dimension_map[self.model_name]

        raise NotImplementedError(
            f"Unknown dimension for model: {self.model_name}. "
            "Local embeddings need to be implemented first."
        )

    def is_available(self) -> bool:
        """
        Check if local embeddings are available.

        Returns:
            Always False (not implemented)
        """
        logger.warning("Local embeddings are not available (not implemented)")
        return False

    def get_model_name(self) -> str:
        """
        Get the model name.

        Returns:
            Model name string
        """
        return self.model_name


# ============================================================
# IMPLEMENTATION GUIDE for Future Developers
# ============================================================
"""
To implement local embeddings, follow these steps:

1. Install dependencies:
   pip install sentence-transformers torch

2. Implement _load_model method:
   def _load_model(self):
       from sentence_transformers import SentenceTransformer
       self._model = SentenceTransformer(self.model_name, device=self.device)
       
3. Implement embed_texts:
   def embed_texts(self, texts: List[str]) -> List[List[float]]:
       if self._model is None:
           self._load_model()
       embeddings = self._model.encode(texts, convert_to_numpy=True)
       return embeddings.tolist()
       
4. Implement embed_query:
   def embed_query(self, query: str) -> List[float]:
       return self.embed_texts([query])[0]

5. Add caching for better performance:
   - Cache loaded models in memory
   - Implement batch processing
   - Add progress bars for large batches

6. Add GPU support:
   - Auto-detect CUDA availability
   - Allow device configuration
   - Handle memory management

7. Add model validation:
   - Check if model exists on HuggingFace
   - Validate model compatibility
   - Handle download errors gracefully

Example complete implementation:

class LocalEmbeddingStrategy(EmbeddingStrategy):
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        
    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading model {self.model_name} on {self.device}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            self._load_model()
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.tolist()
        
    # ... implement other methods
"""
