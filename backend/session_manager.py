"""
Session state management for the RAG chatbot application.
Implements Singleton pattern for centralized state management.
"""

import logging
from typing import Any, Dict

import pandas as pd
import streamlit as st

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
            # LANGUAGE & EMBEDDING CONFIGURATION
            # ============================================================
            'language': None,  # 'en', 'vi', or None
            # Embedding Strategy
            'embedding_strategy': None,  # EmbeddingStrategy instance
            'embedding_provider': None,  # "openai" or "local"
            'embedding_model_name': None,  # Model identifier
            'embedding_api_key': None,  # API key for online providers
            'embedding_dimension': None,  # Vector dimension
            # Deprecated (keep for backward compatibility)
            'embedding_model': None,  # Old SentenceTransformer instance
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
            # ============================================================
            # VECTOR DATABASE (QDRANT)
            # ============================================================
            'qdrant_client': None,  # QdrantManager instance
            'qdrant_manager': None,  # QdrantManager instance
            'collection': None,  # Deprecated
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
            # ============================================================
            # CHAT HISTORY
            # ============================================================
            'chat_history': [],
            # ============================================================
            # UI STATE
            # ============================================================
            'open_dialog': None,
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
        logger.debug("Updating %s value", len(updates))
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

    def is_language_configured(self) -> bool:
        """Check if language is configured."""
        language = self.get('language')
        is_configured = language is not None
        logger.debug("Language configured: %s", is_configured)
        return is_configured

    def is_embedding_configured(self) -> bool:
        """Check if embedding strategy is configured."""
        embedding_strategy = self.get('embedding_strategy')
        is_configured = embedding_strategy is not None
        logger.debug("Embedding configured: %s", is_configured)
        return is_configured

    def is_llm_configured(self) -> bool:
        """Check if LLM is configured."""
        online_llms = self.get('online_llms')
        local_llms = self.get('local_llms')
        is_configured = online_llms is not None or local_llms is not None
        logger.debug("LLM configured: %s", is_configured)
        return is_configured

    def is_data_loaded(self) -> bool:
        """Check if data is loaded to vector DB."""
        data_saved = self.get('data_saved_success', False)
        qdrant_manager = self.get('qdrant_manager')
        is_loaded = data_saved and qdrant_manager is not None
        logger.debug("Data loaded: %s", is_loaded)
        return is_loaded

    def is_ready_for_chat(self) -> bool:
        """Check if all prerequisites for chat are met."""
        language_ok = self.is_language_configured()
        embedding_ok = self.is_embedding_configured()
        llm_ok = self.is_llm_configured()
        data_ok = self.is_data_loaded()

        is_ready = (
            language_ok and embedding_ok and llm_ok and data_ok
        )
        logger.debug(
            "Ready for chat: %s " "(lang:%s, emb:%s, llm:%s, " "data:%s)",
            is_ready,
            language_ok,
            embedding_ok,
            llm_ok,
            data_ok
        )
        return is_ready

    # ============================================================
    # CONFIGURATION SUMMARY
    # ============================================================

    def get_config_summary(self) -> Dict[str, str]:
        """Get configuration summary for display."""
        collection_name = self.get('collection_name', 'Not set')
        if collection_name and len(collection_name) > 30:
            collection_name = collection_name[:27] + "..."

        embedding_strategy = self.get('embedding_strategy')
        embedding_info = 'Not configured'
        if embedding_strategy:
            provider = self.get('embedding_provider', 'Unknown')
            model = self.get('embedding_model_name', 'Unknown')
            embedding_info = f"{provider.upper()}: {model}"

        return {
            'collection_name': collection_name,
            'llm_model': self.get('llm_name', 'Not selected'),
            'llm_type': self.get('llm_type', 'Not specified'),
            'language': self.get('language', 'Not selected'),
            'embedding': embedding_info,
            'chunk_size': str(self.get('chunk_size', 'Not set')),
            'num_retrieval': str(self.get('number_docs_retrieval', 'Not set')),
            'data_saved': 'Yes' if self.get('data_saved_success') else 'No',
            'chunking_option': self.get('chunkOption', 'Not selected'),
        }

    def get_status_summary(self) -> Dict[str, bool]:
        """Get boolean status of major components."""
        return {
            'Language': self.is_language_configured(),
            'Embedding': self.is_embedding_configured(),
            'LLM': self.is_llm_configured(),
            'Data': self.is_data_loaded(),
            'Ready for Chat': self.is_ready_for_chat(),
        }


def get_session_manager() -> SessionManager:
    """Convenience function to get SessionManager instance."""
    return SessionManager()
