"""
Sidebar Navigation and Settings.

Provides navigation between pages and system configuration.
"""

import streamlit as st

from backend.embeddings.embedding_factory import EmbeddingFactory
from backend.llms.llm_factory import LLMFactory
from backend.session_manager import SessionManager
from backend.vector_db.qdrant_manager import QdrantManager
from config.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    OPENAI,
    OPENAI_API_KEY,
    OPENAI_DEFAULT_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_PROVIDER,
    OPENAI_LLM_MODELS,
    PAGE_CHAT,
    PAGE_UPLOAD,
)


class SidebarNavigation:
    """Sidebar navigation and settings component."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self) -> None:
        """Render complete sidebar."""
        st.sidebar.title("🤖 RAG Chatbot")

        # Navigation
        self._render_navigation()

        st.sidebar.markdown("---")

        # Settings
        self._render_settings()

        st.sidebar.markdown("---")

        # Statistics
        self._render_statistics()

    def _render_navigation(self) -> None:
        """Render page navigation."""
        st.sidebar.subheader("📍 Navigation")

        current_page = self.session_manager.get("current_page", PAGE_CHAT)

        # Page buttons
        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button(
                "💬 Chat",
                use_container_width=True,
                type="primary" if current_page == PAGE_CHAT else "secondary",
            ):
                self.session_manager.set("current_page", PAGE_CHAT)
                st.rerun()

        with col2:
            if st.button(
                "📁 Upload",
                use_container_width=True,
                type="primary" if current_page == PAGE_UPLOAD else "secondary",
            ):
                self.session_manager.set("current_page", PAGE_UPLOAD)
                st.rerun()

    def _render_settings(self) -> None:
        """Render settings section."""
        st.sidebar.subheader("⚙️ Settings")

        # API Key
        self._render_api_key_input()

        # LLM Model selection
        self._render_llm_model_selection()

        # Retrieval settings
        self._render_retrieval_settings()

        # Advanced settings (collapsible)
        self._render_advanced_settings()

    def _render_api_key_input(self) -> None:
        """Render API key input."""
        api_key = st.sidebar.text_input(
            "🔑 OpenAI API Key",
            type="password",
            value=OPENAI_API_KEY,
            help="Your API key for OpenAI services",
        )

        # Auto-initialize if API key is provided
        if api_key and api_key != self.session_manager.get("llm_api_key"):
            self._initialize_system(api_key)

    def _initialize_system(self, api_key: str) -> None:
        """Auto-initialize embedding and LLM with API key."""
        try:
            with st.spinner("Initializing system..."):
                # 1. Initialize embeddings
                if not self.session_manager.is_embedding_configured():
                    embedding_strategy = EmbeddingFactory.create_online_embedding(
                        provider=OPENAI_EMBEDDING_PROVIDER,
                        api_key=api_key,
                        model=OPENAI_DEFAULT_EMBEDDING_MODEL,
                    )

                    self.session_manager.update(
                        {
                            'embedding_strategy': embedding_strategy,
                            'embedding_provider': OPENAI_EMBEDDING_PROVIDER,
                            'embedding_model_name': OPENAI_DEFAULT_EMBEDDING_MODEL,
                            'embedding_api_key': api_key,
                            'embedding_dimension': embedding_strategy.get_dimension(),
                        }
                    )

                # 2. Initialize LLM
                llm_model_name = self.session_manager.get(
                    "llm_model_name", DEFAULT_LLM_MODEL
                )
                system_prompt = self.session_manager.get(
                    "system_prompt", DEFAULT_SYSTEM_PROMPT
                )

                llm_model = LLMFactory.create_online_llm(
                    provider_name=OPENAI,
                    api_key=api_key,
                    model_version=llm_model_name,
                    system_prompt=system_prompt,
                )

                self.session_manager.update(
                    {
                        'llm_model': llm_model,
                        'llm_model_name': llm_model_name,
                        'llm_api_key': api_key,
                    }
                )

                # 3. Initialize Qdrant
                if not self.session_manager.is_qdrant_connected():
                    try:
                        qdrant_manager = QdrantManager()
                        if qdrant_manager.is_healthy():
                            self.session_manager.set("qdrant_manager", qdrant_manager)
                    except Exception:
                        pass  # Qdrant optional

                st.sidebar.success("✅ System initialized!")

        except Exception as e:
            st.sidebar.error(f"❌ Initialization error: {str(e)}")

    def _render_llm_model_selection(self) -> None:
        """Render LLM model selection."""
        current_model = self.session_manager.get("llm_model_name", DEFAULT_LLM_MODEL)

        # Find display name
        display_names = list(OPENAI_LLM_MODELS.keys())
        model_values = list(OPENAI_LLM_MODELS.values())

        try:
            current_index = model_values.index(current_model)
        except ValueError:
            current_index = 0

        selected_display = st.sidebar.selectbox(
            "🤖 LLM Model",
            display_names,
            index=current_index,
            help="Select OpenAI model for chat",
        )

        selected_model = OPENAI_LLM_MODELS[selected_display]

        # Update if changed
        if selected_model != current_model:
            self.session_manager.set("llm_model_name", selected_model)

            # Reinitialize LLM if already configured
            if self.session_manager.is_llm_configured():
                api_key = self.session_manager.get("llm_api_key")
                system_prompt = self.session_manager.get(
                    "system_prompt", DEFAULT_SYSTEM_PROMPT
                )

                llm_model = LLMFactory.create_online_llm(
                    provider_name=OPENAI,
                    api_key=api_key,
                    model_version=selected_model,
                    system_prompt=system_prompt,
                )
                self.session_manager.set("llm_model", llm_model)

    def _render_retrieval_settings(self) -> None:
        """Render retrieval settings."""
        num_docs = st.sidebar.slider(
            "📊 Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=self.session_manager.get("number_docs_retrieval", 3),
            help="Number of documents to retrieve for each query",
        )
        self.session_manager.set("number_docs_retrieval", num_docs)

        score_threshold = st.sidebar.slider(
            "🎯 Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=self.session_manager.get("score_threshold", 0.5),
            step=0.05,
            help="Minimum similarity score (0.0 = all, 1.0 = perfect match)",
        )
        self.session_manager.set("score_threshold", score_threshold)

    def _render_advanced_settings(self) -> None:
        """Render advanced settings (collapsible)."""
        with st.sidebar.expander("🔧 Advanced Settings"):
            # Temperature
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=self.session_manager.get("temperature", 0.7),
                step=0.1,
                help="Higher = more creative, Lower = more focused",
            )
            self.session_manager.set("temperature", temperature)

            # Update LLM temperature if configured
            llm_model = self.session_manager.get("llm_model")
            if llm_model:
                llm_model.set_temperature(temperature)

            # System Prompt
            st.markdown("**System Prompt:**")
            current_prompt = self.session_manager.get(
                "system_prompt", DEFAULT_SYSTEM_PROMPT
            )

            system_prompt = st.text_area(
                "Customize assistant behavior",
                value=current_prompt,
                height=200,
                help="Instructions for how the assistant should behave",
            )

            if system_prompt != current_prompt:
                self.session_manager.set("system_prompt", system_prompt)

                # Update LLM system prompt if configured
                if llm_model:
                    llm_model.set_system_prompt(system_prompt)

            if st.button("🔄 Reset to Default Prompt"):
                self.session_manager.set("system_prompt", DEFAULT_SYSTEM_PROMPT)
                if llm_model:
                    llm_model.set_system_prompt(DEFAULT_SYSTEM_PROMPT)
                st.rerun()

    def _render_statistics(self) -> None:
        """Render system statistics."""
        st.sidebar.subheader("📊 System Status")

        status = self.session_manager.get_status_summary()

        for component, is_ready in status.items():
            icon = "✅" if is_ready else "❌"
            st.sidebar.markdown(f"{icon} {component}")

        # Document count
        if self.session_manager.is_qdrant_connected():
            qdrant_manager = self.session_manager.get("qdrant_manager")
            try:
                stats = qdrant_manager.get_statistics()
                doc_count = stats.get("total_documents", 0)
                st.sidebar.metric("📄 Total Documents", doc_count)
            except Exception:
                pass
