"""
Session state management for the RAG chatbot application.
Implements Singleton pattern for centralized state management.
"""

import logging
from typing import Any, Dict

import pandas as pd
import streamlit as st

from config.constants import DEFAULT_SYSTEM_PROMPT, PAGE_CHAT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionManager:
    """Singleton class to manage Streamlit session state."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialize_defaults()
            self._initialized = True
            logger.info("SessionManager initialized with default values")

    def _initialize_defaults(self):
        """Set default values for all session state variables."""
        defaults = {
            # ============================================================
            # PAGE NAVIGATION
            # ============================================================
            'current_page': PAGE_CHAT,  # Current page: 'chat' or 'upload'
            # ============================================================
            # LANGUAGE & EMBEDDING CONFIGURATION
            # ============================================================
            'language': None,  # 'en', 'vi', or None
            # Embedding Strategy
            'embedding_strategy': None,  # EmbeddingStrategy instance
            'embedding_provider': None,  # "openai" or "local"
            'embedding_model_name': None,  # Model identifier
            'embedding_api_key': None,  # API key for online providers
            'embedding_dimension': None,  # Vector dimension
            # ============================================================
            # LLM CONFIGURATION
            # ============================================================
            'llm_type': None,  # 'online_llm' or 'local_llm'
            'llm_name': None,  # Model name
            'online_llms': None,  # Online LLM instance
            'llm_api_key': None,  # API key for online LLM
            'llm_choice': None,  # 'Online' or 'Local (Ollama)'
            'local_llms': None,  # Local LLM instance
            'model_version': None,  # Specific model version
            'selected_model_display': None,  # Display name
            'llm_model': None,  # OpenAILLM instance
            'llm_model_name': None,  # Model name (e.g., "gpt-4o-mini")
            'system_prompt': DEFAULT_SYSTEM_PROMPT,  # System prompt
            'temperature': 0.7,  # LLM temperature
            # ============================================================
            # VECTOR DATABASE (QDRANT)
            # ============================================================
            'qdrant_manager': None,  # QdrantManager instance
            'collection_name': None,  # Collection name
            # ============================================================
            # DATA MANAGEMENT
            # ============================================================
            'chunks_df': pd.DataFrame(),  # DataFrame with chunks
            'doc_ids': [],  # List of document IDs
            'data_saved_success': False,  # Data saved to vector DB
            'source_data': 'UPLOAD',  # 'UPLOAD' or 'DB'
            # ============================================================
            # CHUNKING CONFIGURATION
            # ============================================================
            'chunk_size': 200,  # Chunk size in tokens
            'chunk_overlap': 20,  # Overlap between chunks
            'chunkOption': None,  # Chunking strategy
            'semantic_embedding_option': 'TF-IDF',
            # ============================================================
            # SEARCH & RETRIEVAL
            # ============================================================
            'search_option': 'Vector Search',
            'number_docs_retrieval': 3,
            'score_threshold': 0.5,  # Minimum similarity score
            # ============================================================
            # CHAT HISTORY
            # ============================================================
            'chat_history': [],
            # ============================================================
            # UI STATE
            # ============================================================
            'open_dialog': None,
            'show_settings': False,  # Show advanced settings
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                logger.debug("Initialized '%s' with default value", key)

    # ============================================================
    # BASIC GETTERS & SETTERS
    # ============================================================

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session state."""
        value = st.session_state.get(key, default)
        logger.debug("Getting '%s' = %s", key, value)
        return value

    def set(self, key: str, value: Any) -> None:
        """Set value in session state."""
        logger.debug("Setting '%s' = %s", key, value)
        st.session_state[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple session state values."""
        logger.debug("Updating %s values", len(updates))
        for key, value in updates.items():
            st.session_state[key] = value
            logger.debug("Updated '%s' = %s", key, value)

    def delete(self, key: str) -> None:
        """Delete a key from session state."""
        if key in st.session_state:
            del st.session_state[key]
            logger.debug("Deleted '%s' from session state", key)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in st.session_state

    def clear(self) -> None:
        """Clear all session state and reinitialize."""
        logger.info("Clearing all session state")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_defaults()
        logger.info("Session state cleared and reinitialized")

    # ============================================================
    # SPECIFIC CLEAR METHODS
    # ============================================================

    def clear_chat_history(self) -> None:
        """Clear only chat history."""
        logger.info("Clearing chat history")
        st.session_state['chat_history'] = []

    def clear_data(self) -> None:
        """Clear data-related state."""
        logger.info("Clearing data-related state")
        self.update(
            {
                'chunks_df': pd.DataFrame(),
                'doc_ids': [],
                'collection': None,
                'collection_name': None,
                'data_saved_success': False,
            }
        )

    # ============================================================
    # VALIDATION & CHECK METHODS
    # ============================================================

    def is_embedding_configured(self) -> bool:
        """Check if embedding strategy is configured."""
        embedding_strategy = self.get('embedding_strategy')
        is_configured = embedding_strategy is not None
        logger.debug("Embedding configured: %s", is_configured)
        return is_configured

    def is_llm_configured(self) -> bool:
        """Check if LLM is configured."""
        llm_model = self.get('llm_model')
        is_configured = llm_model is not None
        # online_llms = self.get('online_llms')
        # local_llms = self.get('local_llms')
        # is_configured = online_llms is not None or local_llms is not None
        logger.debug("LLM configured: %s", is_configured)
        return is_configured

    def is_qdrant_connected(self) -> bool:
        """Check if Qdrant is connected."""
        qdrant_manager = self.get('qdrant_manager')
        is_connected = qdrant_manager is not None
        logger.debug("Qdrant connected: %s", is_connected)
        return is_connected

    def has_documents(self) -> bool:
        """Check if there are documents in Qdrant."""
        if not self.is_qdrant_connected():
            return False

        qdrant_manager = self.get('qdrant_manager')
        try:
            stats = qdrant_manager.get_statistics()
            doc_count = stats.get('total_documents', 0)
            return doc_count > 0
        except Exception:
            return False

    def is_ready_for_chat(self) -> bool:
        """Check if system is ready for chat (with or without RAG)."""
        # Minimum: Need LLM configured
        llm_ok = self.is_llm_configured()
        embedding_ok = self.is_embedding_configured()

        # Can chat without documents (LLM only mode)
        is_ready = llm_ok and embedding_ok

        logger.debug(
            "Ready for chat: %s (llm:%s, embedding:%s)",
            is_ready,
            llm_ok,
            embedding_ok,
        )
        return is_ready

    # ============================================================
    # CONFIGURATION SUMMARY
    # ============================================================

    def get_config_summary(self) -> Dict[str, str]:
        """Get configuration summary for display."""
        qdrant_manager = self.get('qdrant_manager')

        if qdrant_manager:
            try:
                stats = qdrant_manager.get_statistics()
                doc_count = stats.get('total_documents', 0)
                collection_name = stats.get('collection_name', 'N/A')
            except Exception:
                doc_count = 0
                collection_name = 'N/A'
        else:
            doc_count = 0
            collection_name = 'Not connected'

        # check llm_model
        llm_model = self.get('llm_model')
        llm_name = llm_model.get_model_name() if llm_model else 'Not configured'

        embedding_strategy = self.get('embedding_strategy')
        embedding_name = (
            embedding_strategy.get_model_name()
            if embedding_strategy
            else 'Not configured'
        )

        return {
            'llm_model': llm_name,
            'embedding_model': embedding_name,
            'collection': collection_name,
            'total_docs': str(doc_count),
            'num_retrieval': str(self.get('number_docs_retrieval', 3)),
            'temperature': str(self.get('temperature', 0.7)),
            'score_threshold': str(self.get('score_threshold', 0.5)),
        }

    def get_status_summary(self) -> Dict[str, bool]:
        """Get boolean status of major components."""
        return {
            'Embedding': self.is_embedding_configured(),
            'LLM': self.is_llm_configured(),
            'Qdrant': self.is_qdrant_connected(),
            'Documents': self.has_documents(),
            'Ready': self.is_ready_for_chat(),
        }


def get_session_manager() -> SessionManager:
    """Convenience function to get SessionManager instance."""
    return SessionManager()
