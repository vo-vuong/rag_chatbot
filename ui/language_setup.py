import traceback

import streamlit as st

from backend.embeddings.embedding_factory import EmbeddingFactory
from backend.session_manager import SessionManager
from config.constants import (
    EN,
    ENGLISH,
    NONE,
    OPENAI_API_KEY,
    OPENAI_DEFAULT_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_PROVIDER,
    VI,
    VIETNAMESE,
)


class LanguageSetupUI:
    """UI component for language and embedding configuration."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self, header_number: int) -> None:
        st.header(f"{header_number}. Setup Language & Embeddings")

        # Language selection
        st.subheader("Select Document Language")
        language_choice = st.radio(
            "Choose the primary language of your documents:",
            [NONE, ENGLISH, VIETNAMESE],
            index=0,
            help="This sets metadata for your documents. ",
        )

        if language_choice != NONE:
            lang_code = EN if language_choice == ENGLISH else VI
            self.session_manager.set("language", lang_code)

            st.success(f"✅ Language set to: **{language_choice}**")

            # Show embedding configuration
            self._render_embedding_config()

    def _render_embedding_config(self) -> None:
        """Render embedding configuration section."""
        st.subheader("Embedding Configuration")

        # Check if embedding is already configured
        if self.session_manager.is_embedding_configured():
            self._show_configured_embedding()
        else:
            self._configure_embedding()

    def _show_configured_embedding(self) -> None:
        """Show currently configured embedding."""
        provider = self.session_manager.get('embedding_provider')
        model = self.session_manager.get('embedding_model_name')
        dimension = self.session_manager.get('embedding_dimension')

        st.info(
            f"✅ **Embedding Configured**\n\n"
            f"- Provider: {provider.upper()}\n"
            f"- Model: {model}\n"
            f"- Dimension: {dimension}"
        )

        if st.button("🔄 Reconfigure Embedding"):
            self.session_manager.update(
                {
                    'embedding_strategy': None,
                    'embedding_provider': None,
                    'embedding_model_name': None,
                    'embedding_dimension': None,
                }
            )
            st.rerun()

    def _configure_embedding(self) -> None:
        """Configure embedding strategy."""
        # API Key input
        default_key = OPENAI_API_KEY
        api_key = st.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=default_key,
            help="Get your API key from https://platform.openai.com/api-keys",
        )

        if not api_key and not default_key:
            st.warning("⚠️ Please enter your OpenAI API key to continue")
            return

        selected_model = OPENAI_DEFAULT_EMBEDDING_MODEL
        # TODO: Apply strategy init embedding model
        self._initialize_embedding(api_key, selected_model)
        return

    def _initialize_embedding(self, api_key: str, model: str) -> None:
        """
        Initialize embedding strategy.

        Args:
            api_key: OpenAI API key
            model: Model name
        """
        try:
            with st.spinner(f"Initializing {model}..."):
                # Create embedding strategy
                embedding_strategy = EmbeddingFactory.create_online_embedding(
                    provider=OPENAI_EMBEDDING_PROVIDER,
                    api_key=api_key,
                    model=model,
                )

                # Test if available
                if not embedding_strategy.is_available():
                    st.error(
                        "❌ Failed to connect to OpenAI API. "
                        "Please check your API key."
                    )
                    return

                # Get dimension
                dimension = embedding_strategy.get_dimension()

                # Save to session
                self.session_manager.update(
                    {
                        'embedding_strategy': embedding_strategy,
                        'embedding_provider': OPENAI_EMBEDDING_PROVIDER,
                        'embedding_model_name': model,
                        'embedding_api_key': api_key,
                        'embedding_dimension': dimension,
                    }
                )

                st.success(f"✅ Successfully initialized **{model}**!\n\n")

        except Exception as e:
            st.error(f"❌ Error initializing embedding: {str(e)}")
            st.code(traceback.format_exc())

    def is_configured(self) -> bool:
        """
        Check if language and embedding are configured.

        Returns:
            True if both are configured
        """
        return (
            self.session_manager.is_language_configured()
            and self.session_manager.is_embedding_configured()
        )
