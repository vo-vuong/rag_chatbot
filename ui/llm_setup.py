import os

import streamlit as st

from backend.llms.llm_factory import LLMFactory
from backend.llms.ollama_manager import GGUF_MODEL_OPTIONS, OLLAMA_MODEL_OPTIONS
from backend.session_manager import SessionManager
from config.constants import GEMINI, LOCAL_LLM, ONLINE_LLM, OPENAI
from ui.components import notify


class LLMSetupUI:
    """UI component for LLM configuration."""

    def __init__(self, session_manager: SessionManager):
        """
        Initialize LLM setup UI.

        Args:
            session_manager: Session manager instance
        """
        self.session_manager = session_manager
        self.llm_factory = LLMFactory()

    def render(self, header_number: int) -> None:
        """
        Render LLM setup section.

        Args:
            header_number: Section number for header
        """
        st.header(f"{header_number}. Setup LLMs")

        # LLM source selection
        llm_choice = st.selectbox(
            "Choose Model Source:",
            ["Online", "Local (Ollama)"],
            index=1,  # Default to Local
            key="llm_choice",
            help="Select whether to use online APIs or local models",
        )

        if llm_choice == "Online":
            self._render_online_llm_setup()
        else:
            self._render_local_llm_setup()

    def _render_online_llm_setup(self) -> None:
        """Render online LLM configuration."""
        self.session_manager.set("llm_type", ONLINE_LLM)

        # Provider selection
        provider = st.selectbox(
            "Select LLM Provider:",
            [OPENAI, GEMINI],
            index=0,
            help="Choose your LLM provider",
        )
        self.session_manager.set("llm_name", provider)

        # API key input
        self._render_api_key_input(provider)

        # Model version selection
        self._render_model_version_selection(provider)

        # Initialize LLM if configured
        if self.session_manager.get("llm_api_key"):
            self._initialize_online_llm(provider)

    def _render_api_key_input(self, provider: str) -> None:
        """
        Render API key input field.

        Args:
            provider: LLM provider name
        """
        # Show API key documentation link
        if provider == OPENAI:
            default_key = os.getenv("OPENAI_API_KEY", "")
        else:  # GEMINI
            default_key = os.getenv("GEMINI_API_KEY", "")

        api_key = st.text_input(
            "Enter your API Key:",
            type="password",
            value=default_key,
            key="llm_api_key",
            help="Your API key will be securely stored in session",
        )

        if api_key:
            st.success("✅ API Key saved successfully!")

    def _render_model_version_selection(self, provider: str) -> None:
        """
        Render model version selection.

        Args:
            provider: LLM provider name
        """
        if provider == OPENAI:
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        else:  # GEMINI
            model_options = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]

        model_version = st.selectbox(
            "Select Model Version:",
            model_options,
            index=0,
            help="Choose the model version to use",
        )
        self.session_manager.set("model_version", model_version)

    def _initialize_online_llm(self, provider: str) -> None:
        """
        Initialize online LLM with current configuration.

        Args:
            provider: LLM provider name
        """
        api_key = self.session_manager.get("llm_api_key")
        model_version = self.session_manager.get("model_version", "gpt-4o-mini")

        try:
            llm_model = self.llm_factory.create_online_llm(
                provider_name=provider, api_key=api_key, model_version=model_version
            )
            self.session_manager.set("online_llms", llm_model)
        except Exception as e:
            notify(f"Error initializing LLM: {str(e)}", "error")

    def _render_local_llm_setup(self) -> None:
        """Render local LLM (Ollama) configuration."""
        self.session_manager.set("llm_type", LOCAL_LLM)

        # Container initialization
        if st.button("Initialize Ollama Container", type="primary"):
            with st.spinner("Setting up Ollama container..."):
                # TODO: Implement container initialization
                notify("Ollama container initialization to be implemented", "info")

        # Model format selection
        model_format = st.radio(
            "Select Model Format:",
            ["Normal", "High Performance"],
            captions=[
                "HuggingFace normal format",
                "HuggingFace GGUF format (optimized)",
            ],
            index=0,
            help="GGUF format is optimized for better performance",
        )

        # Model selection
        if model_format == "Normal":
            self._render_normal_model_selection()
        else:
            self._render_gguf_model_selection()

        # Run model button
        if st.button("Run Selected Model", type="primary"):
            self._initialize_local_llm()

    def _render_normal_model_selection(self) -> None:
        """Render normal model selection."""
        selected_model = st.selectbox(
            "Select a Model:",
            list(OLLAMA_MODEL_OPTIONS.keys()),
            help="Choose from available Ollama models",
        )
        model_name = OLLAMA_MODEL_OPTIONS[selected_model]
        self.session_manager.update(
            {"llm_name": model_name, "selected_model_display": selected_model}
        )

    def _render_gguf_model_selection(self) -> None:
        """Render GGUF model selection."""
        selected_model = st.selectbox(
            "Select a GGUF Model:",
            list(GGUF_MODEL_OPTIONS.keys()),
            help="Choose from available GGUF models",
        )
        model_name = GGUF_MODEL_OPTIONS[selected_model]

        st.markdown(
            "Or enter custom model name. Format: `hf.co/{username}/{repository}`. "
            "See [HuggingFace GGUF models](https://huggingface.co/models?library=gguf&sort=trending)"
        )

        custom_model = st.text_input(
            "Custom GGUF Model Name:",
            placeholder="hf.co/username/model-name",
            help="Enter custom HuggingFace GGUF model path",
        )

        final_model = custom_model if custom_model else model_name
        self.session_manager.update(
            {"llm_name": final_model, "selected_model_display": selected_model}
        )

    def _initialize_local_llm(self) -> None:
        """Initialize local LLM model."""
        model_name = self.session_manager.get("llm_name")

        if not model_name:
            notify("Please select a model first", "warning")
            return

        with st.spinner(f"Initializing {model_name}..."):
            try:
                # TODO: Implement actual local LLM initialization
                llm_model = self.llm_factory.create_local_llm(model_name=model_name)
                self.session_manager.set("local_llms", llm_model)
                st.success(f"✅ Model {model_name} initialized successfully!")
            except Exception as e:
                notify(f"Error initializing model: {str(e)}", "error")

    def is_configured(self) -> bool:
        """
        Check if LLM is properly configured.

        Returns:
            True if configured, False otherwise
        """
        return self.session_manager.is_llm_configured()
