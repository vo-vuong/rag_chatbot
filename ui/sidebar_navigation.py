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
    OPENAI,
    OPENAI_API_KEY,
    OPENAI_DEFAULT_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_PROVIDER,
    OPENAI_LLM_MODELS,
    PAGE_CHAT,
    PAGE_DATA_MANAGEMENT,
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

        # Page buttons - Chat, Upload, Data Management
        if st.sidebar.button(
            "💬 Chat",
            use_container_width=True,
            type="primary" if current_page == PAGE_CHAT else "secondary",
        ):
            self.session_manager.set("current_page", PAGE_CHAT)
            st.rerun()

        if st.sidebar.button(
            "📁 Upload",
            use_container_width=True,
            type="primary" if current_page == PAGE_UPLOAD else "secondary",
        ):
            self.session_manager.set("current_page", PAGE_UPLOAD)
            st.rerun()

        if st.sidebar.button(
            "🗂️ Data Management",
            use_container_width=True,
            type="primary" if current_page == PAGE_DATA_MANAGEMENT else "secondary",
        ):
            self.session_manager.set("current_page", PAGE_DATA_MANAGEMENT)
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

                # Get system prompt from active template
                system_prompt = self.session_manager.get_active_system_prompt()

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

                # Get system prompt from active template
                system_prompt = self.session_manager.get_active_system_prompt()

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

        # Image search settings (if image collection exists)
        if self.session_manager.has_image_collection():
            st.sidebar.markdown("---")
            st.sidebar.markdown("**🖼️ Image Search**")

            # Enable/disable image search
            enable_image_search = st.sidebar.checkbox(
                "Enable Image Search",
                value=self.session_manager.get("enable_image_search", True),
                help="Include images in RAG responses"
            )
            self.session_manager.set("enable_image_search", enable_image_search)

            if enable_image_search:
                # Image score threshold
                image_threshold = st.sidebar.slider(
                    "Image Score Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.session_manager.get_image_score_threshold(),
                    step=0.05,
                    help="Minimum similarity score for images (higher = stricter)"
                )
                if image_threshold != self.session_manager.get_image_score_threshold():
                    self.session_manager.set_image_score_threshold(image_threshold)

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

            # System Prompt Template Selector
            st.markdown("**System Prompt Template:**")

            # Get available system templates
            prompt_manager = self.session_manager.get('prompt_manager')
            if prompt_manager:
                system_templates = prompt_manager.list_templates(category='system')
                template_names = [t.name for t in system_templates]
                template_display = {
                    t.name: f"{t.name} - {t.description}" for t in system_templates
                }

                current_template = self.session_manager.get(
                    'active_system_template', 'system_helpful_assistant'
                )

                # Template selector
                selected_template = st.selectbox(
                    "Select template",
                    options=template_names,
                    index=(
                        template_names.index(current_template)
                        if current_template in template_names
                        else 0
                    ),
                    format_func=lambda x: template_display[x],
                    help="Choose a system prompt template",
                )

                # Update if changed
                if selected_template != current_template:
                    self.session_manager.set(
                        'active_system_template', selected_template
                    )

                    # Update LLM system prompt if configured
                    if llm_model:
                        new_prompt = self.session_manager.get_active_system_prompt()
                        llm_model.set_system_prompt(new_prompt)
                        st.success(f"✅ Switched to: {selected_template}")
                        st.rerun()

                # Show preview with checkbox toggle
                show_preview = st.checkbox("📝 Show Template Preview", value=False)
                if show_preview:
                    template = prompt_manager.get_template(selected_template)
                    if template:
                        st.caption("Template content:")
                        st.code(template.template, language="text")

            # RAG Template Selector
            st.markdown("---")
            st.markdown("**RAG Prompt Template:**")

            if prompt_manager:
                rag_templates = prompt_manager.list_templates(category='rag')
                rag_template_names = [t.name for t in rag_templates]
                rag_template_display = {
                    t.name: f"{t.name.replace('rag_qa_', '').title()} - {t.description}"
                    for t in rag_templates
                }

                current_rag_template = self.session_manager.get(
                    'active_rag_template', 'rag_qa_with_history'
                )

                # RAG template selector
                selected_rag_template = st.selectbox(
                    "Select RAG template",
                    options=rag_template_names,
                    index=(
                        rag_template_names.index(current_rag_template)
                        if current_rag_template in rag_template_names
                        else 0
                    ),
                    format_func=lambda x: rag_template_display.get(x, x),
                    help="Choose how the system formats RAG questions",
                    key="rag_template_selector",
                )

                # Update if changed
                if selected_rag_template != current_rag_template:
                    self.session_manager.set(
                        'active_rag_template', selected_rag_template
                    )
                    st.success(f"✅ RAG template: {selected_rag_template}")

                # Show preview
                show_rag_preview = st.checkbox(
                    "📝 Show RAG Template Preview",
                    value=False,
                    key="rag_preview_checkbox",
                )
                if show_rag_preview:
                    rag_template = prompt_manager.get_template(selected_rag_template)
                    if rag_template:
                        st.caption("RAG template content:")
                        st.code(rag_template.template, language="text")
                        st.caption(f"Variables: {', '.join(rag_template.variables)}")

            # Chat Template Selector
            st.markdown("---")
            st.markdown("**Chat Prompt Template:**")

            if prompt_manager:
                chat_templates = prompt_manager.list_templates(category='chat')
                chat_template_names = [t.name for t in chat_templates]
                chat_template_display = {
                    t.name: f"{t.name.replace('chat_', '').title()} - {t.description}"
                    for t in chat_templates
                }

                current_chat_template = self.session_manager.get(
                    'active_chat_template', 'chat_conversational'
                )

                # Chat template selector
                selected_chat_template = st.selectbox(
                    "Select chat template",
                    options=chat_template_names,
                    index=(
                        chat_template_names.index(current_chat_template)
                        if current_chat_template in chat_template_names
                        else 0
                    ),
                    format_func=lambda x: chat_template_display.get(x, x),
                    help="Choose how the system handles chat-only conversations",
                    key="chat_template_selector",
                )

                # Update if changed
                if selected_chat_template != current_chat_template:
                    self.session_manager.set(
                        'active_chat_template', selected_chat_template
                    )
                    st.success(f"✅ Chat template: {selected_chat_template}")

                # Show preview
                show_chat_preview = st.checkbox(
                    "📝 Show Chat Template Preview",
                    value=False,
                    key="chat_preview_checkbox",
                )
                if show_chat_preview:
                    chat_template = prompt_manager.get_template(selected_chat_template)
                    if chat_template:
                        st.caption("Chat template content:")
                        st.code(chat_template.template, language="text")
                        st.caption(f"Variables: {', '.join(chat_template.variables)}")
            else:
                st.info("Prompt templates not loaded. Using default prompt.")

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
