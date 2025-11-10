import streamlit as st
from sentence_transformers import SentenceTransformer

from backend.session_manager import SessionManager
from config.constants import (
    EN,
    ENGLISH,
    ENGLISH_EMBEDDING_MODEL,
    NONE,
    VI,
    VIETNAMESE,
    VIETNAMESE_EMBEDDING_MODEL,
)


class LanguageSetupUI:
    """UI component for language and embedding model configuration."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self, header_number: int) -> None:
        st.header(f"{header_number}. Setup Language")

        language_choice = st.radio(
            "Select language:",
            [NONE, ENGLISH, VIETNAMESE],
            index=0,
            help="Choose the language for embedding model",
        )

        self._handle_language_selection(language_choice)

    def _handle_language_selection(self, language_choice: str) -> None:
        """
        Handle language selection and load appropriate embedding model.

        Args:
            language_choice: Selected language
        """
        if language_choice == ENGLISH:
            self._setup_english_model()
        elif language_choice == VIETNAMESE:
            self._setup_vietnamese_model()

    def _setup_english_model(self) -> None:
        """Setup English embedding model."""
        current_language = self.session_manager.get("language")
        current_model_name = self.session_manager.get("embedding_model_name")

        if current_language != EN or current_model_name != ENGLISH_EMBEDDING_MODEL:
            with st.spinner("Loading English embedding model..."):
                try:
                    embedding_model = SentenceTransformer(ENGLISH_EMBEDDING_MODEL)
                    self.session_manager.update(
                        {
                            "language": EN,
                            "embedding_model": embedding_model,
                            "embedding_model_name": ENGLISH_EMBEDDING_MODEL,
                        }
                    )
                    st.success(
                        f"✅ Using English embedding model: {ENGLISH_EMBEDDING_MODEL}"
                    )
                except Exception as e:
                    st.error(f"Error loading English model: {str(e)}")

    def _setup_vietnamese_model(self) -> None:
        """Setup Vietnamese embedding model."""
        current_language = self.session_manager.get("language")
        current_model_name = self.session_manager.get("embedding_model_name")

        if current_language != VI or current_model_name != VIETNAMESE_EMBEDDING_MODEL:
            with st.spinner("Loading Vietnamese embedding model..."):
                try:
                    embedding_model = SentenceTransformer(VIETNAMESE_EMBEDDING_MODEL)
                    self.session_manager.update(
                        {
                            "language": VI,
                            "embedding_model": embedding_model,
                            "embedding_model_name": VIETNAMESE_EMBEDDING_MODEL,
                        }
                    )
                    st.success(
                        f"✅ Using Vietnamese embedding model: {VIETNAMESE_EMBEDDING_MODEL}"
                    )
                except Exception as e:
                    st.error(f"Error loading Vietnamese model: {str(e)}")

    def is_configured(self) -> bool:
        """
        Check if language is properly configured.

        Returns:
            True if configured, False otherwise
        """
        return self.session_manager.is_language_configured()
