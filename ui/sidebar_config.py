import streamlit as st

from backend.session_manager import SessionManager


class SidebarConfig:
    """UI component for sidebar configuration display."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self) -> None:
        config = self.session_manager.get_config_summary()
        st.sidebar.subheader("Current Settings")

        # Format and display
        for idx, (key, value) in enumerate(config.items(), 1):
            formatted_key = self._format_key(key)
            st.sidebar.markdown(f"{idx}. **{formatted_key}:** {value}")

        st.sidebar.markdown("---")

        # Additional settings
        self._render_chunk_settings()

    def _format_key(self, key: str) -> str:
        """
        Format configuration key for display.

        Args:
            key: Configuration key

        Returns:
            Formatted key
        """
        return key.replace('_', ' ').title()

    def _render_chunk_settings(self) -> None:
        """Render chunk size and overlap settings in sidebar."""
        st.sidebar.subheader("Chunking Parameters")

        chunk_size = st.sidebar.number_input(
            "Chunk Size",
            min_value=10,
            max_value=1000,
            value=self.session_manager.get("chunk_size", 200),
            step=10,
            help="Size of each text chunk in tokens",
            key="sidebar_chunk_size",
        )

        chunk_overlap = st.sidebar.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=self.session_manager.get("chunk_overlap", 20),
            step=10,
            help="Number of overlapping tokens between chunks",
            key="sidebar_chunk_overlap",
        )

        # Update session state
        self.session_manager.update(
            {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        )

        st.sidebar.markdown("---")

        # Retrieval settings
        self._render_retrieval_settings()

    def _render_retrieval_settings(self) -> None:
        """Render retrieval settings in sidebar."""
        st.sidebar.subheader("Retrieval Settings")

        num_docs = st.sidebar.number_input(
            "Number of Documents",
            min_value=1,
            max_value=50,
            value=self.session_manager.get("number_docs_retrieval", 3),
            step=1,
            help="Number of documents to retrieve for each query",
            key="sidebar_num_retrieval",
        )

        self.session_manager.set("number_docs_retrieval", num_docs)
