"""
Session state management for the RAG chatbot application.
Implements Singleton pattern for centralized state management.

This module provides a centralized way to manage Streamlit's session state,
ensuring consistent state management across all UI components and preventing
common issues with direct session state access.

Usage:
    from backend.session_manager import SessionManager

    session_manager = SessionManager()
    session_manager.set("language", "en")
    language = session_manager.get("language")
"""

import logging
from typing import Any, Dict

import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionManager:
    """
    Singleton class to manage Streamlit session state.

    This class provides centralized access and initialization of session variables,
    ensuring consistent state management throughout the application. It implements
    the Singleton pattern to guarantee only one instance exists.

    Key Features:
        - Automatic initialization of default values
        - Type-safe access to session state
        - Helper methods for common state checks
        - Centralized state management
        - Debug logging support

    Example:
        >>> session_manager = SessionManager()
        >>> session_manager.set("language", "en")
        >>> language = session_manager.get("language")
        >>> if session_manager.is_data_loaded():
        ...     print("Data is ready!")
    """

    _instance = None

    def __new__(cls):
        """
        Implement Singleton pattern.

        Ensures only one instance of SessionManager exists throughout
        the application lifecycle.

        Returns:
            SessionManager: The single instance of SessionManager
        """
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            logger.info("SessionManager instance created")
        return cls._instance

    def __init__(self):
        """
        Initialize session state with default values.

        This method is called every time SessionManager() is instantiated,
        but actual initialization only happens once due to the _initialized flag.
        """
        if not hasattr(self, '_initialized'):
            self._initialize_defaults()
            self._initialized = True
            logger.info("SessionManager initialized with default values")

    def _initialize_defaults(self):
        """
        Set default values for all session state variables.

        This method defines all variables used throughout the application
        with their default values, preventing KeyError and ensuring
        consistent initial state.
        """
        defaults = {
            # ============================================================
            # LANGUAGE & EMBEDDING CONFIGURATION
            # ============================================================
            'language': None,  # Selected language: 'en', 'vi', or None
            'embedding_model': None,  # SentenceTransformer model instance
            'embedding_model_name': None,  # Name of the embedding model
            # ============================================================
            # LLM CONFIGURATION
            # ============================================================
            'llm_type': None,  # Type: 'online_llm' or 'local_llm'
            'llm_name': None,  # Model name: 'Gemini', 'OpenAI', or model identifier
            'online_llms': None,  # Online LLM model instance
            'llm_api_key': None,  # API key for online LLM
            'llm_choice': None,  # User's choice: 'Online' or 'Local (Ollama)'
            'local_llms': None,  # Local LLM model instance (Ollama)
            'model_version': None,  # Specific model version (e.g., 'gpt-4o')
            'selected_model_display': None,  # Display name of selected model
            # ============================================================
            # VECTOR DATABASE CONFIGURATION
            # ============================================================
            'qdrant_client': None,  # Qdrant client instance
            'collection': None,  # Current Qdrant collection object
            'collection_name': None,  # Name of the current collection
            'random_collection_name': None,  # Auto-generated collection name
            # ============================================================
            # DATA MANAGEMENT
            # ============================================================
            'chunks_df': pd.DataFrame(),  # DataFrame containing text chunks
            'doc_ids': [],  # List of document IDs
            'columns_to_answer': [],  # Columns selected for chatbot responses
            'data_saved_success': False,  # Flag: data saved to vector DB
            'source_data': 'UPLOAD',  # Data source: 'UPLOAD' or 'DB'
            # ============================================================
            # CHUNKING CONFIGURATION
            # ============================================================
            'chunk_size': 200,  # Size of each text chunk in tokens
            'chunk_overlap': 20,  # Overlap between consecutive chunks
            'chunkOption': None,  # Selected chunking strategy
            'semantic_embedding_option': 'TF-IDF',  # Embedding for semantic chunker
            # ============================================================
            # SEARCH & RETRIEVAL CONFIGURATION
            # ============================================================
            'search_option': 'Vector Search',  # Selected search strategy
            'number_docs_retrieval': 3,  # Number of documents to retrieve
            # ============================================================
            # CHAT HISTORY
            # ============================================================
            'chat_history': [],  # List of chat messages: [{'role': ..., 'content': ...}]
            # ============================================================
            # UI STATE
            # ============================================================
            'open_dialog': None,  # Currently open dialog identifier
        }

        # Initialize all defaults in session state
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                logger.debug("Initialized '%s' with default value", key)

    # ============================================================
    # BASIC GETTERS & SETTERS
    # ============================================================

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from session state with optional default.

        This method provides safe access to session state variables,
        returning a default value if the key doesn't exist.

        Args:
            key: Session state key to retrieve
            default: Default value to return if key doesn't exist

        Returns:
            Value from session state or default value

        Example:
            >>> language = session_manager.get("language", "en")
            >>> chunk_size = session_manager.get("chunk_size", 200)
        """
        value = st.session_state.get(key, default)
        logger.debug("Getting '%s' = %s", key, value)
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set value in session state.

        This method provides a centralized way to update session state,
        with logging for debugging purposes.

        Args:
            key: Session state key to set
            value: Value to store

        Example:
            >>> session_manager.set("language", "en")
            >>> session_manager.set("chunk_size", 300)
        """
        logger.debug("Setting '%s' = %s", key, value)
        st.session_state[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple session state values at once.

        This method is useful when you need to update several related
        variables together, ensuring atomic updates.

        Args:
            updates: Dictionary of key-value pairs to update

        Example:
            >>> session_manager.update({
            ...     "language": "en",
            ...     "embedding_model": model,
            ...     "embedding_model_name": "all-MiniLM-L6-v2"
            ... })
        """
        logger.debug("Updating %s value", len(updates))
        for key, value in updates.items():
            st.session_state[key] = value
            logger.debug("Updated '%s' = %s", key, value)

    def delete(self, key: str) -> None:
        """
        Delete a key from session state.

        Args:
            key: Session state key to delete

        Example:
            >>> session_manager.delete("temp_data")
        """
        if key in st.session_state:
            del st.session_state[key]
            logger.debug("Deleted '%s' from session state", key)

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in session state.

        Args:
            key: Session state key to check

        Returns:
            True if key exists, False otherwise

        Example:
            >>> if session_manager.exists("language"):
            ...     print("Language is configured")
        """
        return key in st.session_state

    def clear(self) -> None:
        """
        Clear all session state and reinitialize with defaults.

        This method is useful for "reset" functionality, clearing all
        user data and returning to initial state.

        Warning:
            This will delete ALL session state data!

        Example:
            >>> if st.button("Reset All"):
            ...     session_manager.clear()
            ...     st.rerun()
        """
        logger.info("Clearing all session state")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_defaults()
        logger.info("Session state cleared and reinitialized")

    # ============================================================
    # SPECIFIC CLEAR METHODS
    # ============================================================

    def clear_chat_history(self) -> None:
        """
        Clear only chat history while preserving other state.

        This is useful for "New Chat" functionality without losing
        configuration settings.

        Example:
            >>> if st.button("New Chat"):
            ...     session_manager.clear_chat_history()
            ...     st.rerun()
        """
        logger.info("Clearing chat history")
        st.session_state['chat_history'] = []

    def clear_data(self) -> None:
        """
        Clear data-related state (chunks, collection, etc.).

        This is useful when user wants to load new data without
        reconfiguring LLM and other settings.

        Example:
            >>> if st.button("Load New Data"):
            ...     session_manager.clear_data()
        """
        logger.info("Clearing data-related state")
        self.update(
            {
                'chunks_df': pd.DataFrame(),
                'doc_ids': [],
                'collection': None,
                'collection_name': None,
                'data_saved_success': False,
                'columns_to_answer': [],
            }
        )

    # ============================================================
    # VALIDATION & CHECK METHODS
    # ============================================================

    def is_language_configured(self) -> bool:
        """
        Check if language and embedding model are properly configured.

        Returns:
            True if both language and embedding model are set, False otherwise

        Example:
            >>> if not session_manager.is_language_configured():
            ...     st.warning("Please select a language first")
        """
        language = self.get('language')
        embedding_model = self.get('embedding_model')
        is_configured = language is not None and embedding_model is not None
        logger.debug("Language configured: %s", is_configured)
        return is_configured

    def is_llm_configured(self) -> bool:
        """
        Check if LLM is properly configured (online or local).

        Returns:
            True if either online or local LLM is configured, False otherwise

        Example:
            >>> if not session_manager.is_llm_configured():
            ...     st.warning("Please configure LLM first")
        """
        online_llms = self.get('online_llms')
        local_llms = self.get('local_llms')
        is_configured = online_llms is not None or local_llms is not None
        logger.debug("LLM configured: %s", is_configured)
        return is_configured

    def is_data_loaded(self) -> bool:
        """
        Check if data is loaded and saved to vector database.

        This method checks multiple conditions to ensure data is
        properly loaded and ready for use.

        Returns:
            True if data is loaded and saved successfully, False otherwise

        Example:
            >>> if session_manager.is_data_loaded():
            ...     st.success("Data is ready for chatbot")
            ... else:
            ...     st.warning("Please load data first")
        """
        data_saved = self.get('data_saved_success', False)
        collection = self.get('collection')
        is_loaded = data_saved and collection is not None
        logger.debug("Data loaded: %s", is_loaded)
        return is_loaded

    def is_ready_for_chat(self) -> bool:
        """
        Check if all prerequisites for chatbot are met.

        This includes: language configured, LLM configured, data loaded,
        and columns selected for answers.

        Returns:
            True if all prerequisites are met, False otherwise

        Example:
            >>> if session_manager.is_ready_for_chat():
            ...     # Show chat interface
            ...     render_chat()
            ... else:
            ...     st.warning("Complete setup first")
        """
        language_ok = self.is_language_configured()
        llm_ok = self.is_llm_configured()
        data_ok = self.is_data_loaded()
        columns_selected = len(self.get('columns_to_answer', [])) > 0

        is_ready = language_ok and llm_ok and data_ok and columns_selected
        logger.debug(
            "Ready for chat: %s (lang:%s, llm:%s, data:%s, cols:%s)",
            is_ready,
            language_ok,
            llm_ok,
            data_ok,
            columns_selected,
        )
        return is_ready

    # ============================================================
    # CONFIGURATION SUMMARY
    # ============================================================

    def get_config_summary(self) -> Dict[str, str]:
        """
        Get summary of current configuration for display.

        This method returns a formatted dictionary containing all
        important configuration values, useful for displaying in
        sidebar or status sections.

        Returns:
            Dictionary with configuration details formatted as strings

        Example:
            >>> config = session_manager.get_config_summary()
            >>> for key, value in config.items():
            ...     st.sidebar.markdown(f"**{key}:** {value}")
        """
        collection_name = self.get('collection_name', 'No collection')
        if collection_name and len(collection_name) > 30:
            collection_name = collection_name[:27] + "..."

        embedding_model = self.get('embedding_model')
        embedding_name = (
            embedding_model.__class__.__name__ if embedding_model else 'None'
        )

        return {
            'collection_name': collection_name,
            'llm_model': self.get('llm_name', 'Not selected'),
            'llm_type': self.get('llm_type', 'Not specified'),
            'language': self.get('language', 'Not selected'),
            'embedding_model': embedding_name,
            'chunk_size': str(self.get('chunk_size', 'Not set')),
            'num_retrieval': str(self.get('number_docs_retrieval', 'Not set')),
            'data_saved': 'Yes' if self.get('data_saved_success') else 'No',
            'api_key_set': 'Yes' if self.get('llm_api_key') else 'No',
            'chunking_option': self.get('chunkOption', 'Not selected'),
        }

    def get_status_summary(self) -> Dict[str, bool]:
        """
        Get boolean status of major components.

        This is useful for showing status indicators (✅/❌) in UI.

        Returns:
            Dictionary with component names and their ready status

        Example:
            >>> status = session_manager.get_status_summary()
            >>> for component, ready in status.items():
            ...     icon = "✅" if ready else "❌"
            ...     st.write(f"{icon} {component}")
        """
        return {
            'Language': self.is_language_configured(),
            'LLM': self.is_llm_configured(),
            # 'Data': self.is_data_loaded(),
            'Columns': len(self.get('columns_to_answer', [])) > 0,
            'Ready for Chat': self.is_ready_for_chat(),
        }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def get_session_manager() -> SessionManager:
    """
    Convenience function to get SessionManager instance.

    This function is useful when you want to get the session manager
    without having to remember to instantiate it.

    Returns:
        SessionManager instance (always the same singleton instance)

    Example:
        >>> from backend.session_manager import get_session_manager
        >>> sm = get_session_manager()
        >>> sm.set("language", "en")
    """
    return SessionManager()
