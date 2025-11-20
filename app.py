import streamlit as st
from dotenv import load_dotenv

from backend.session_manager import SessionManager
from config.constants import PAGE_CHAT, PAGE_UPLOAD, PAGE_DATA_MANAGEMENT
from ui.chat_main import ChatMainUI
from ui.data_upload import DataUploadUI
from ui.data_management import DataManagementUI
from ui.sidebar_navigation import SidebarNavigation

load_dotenv()


class RAGChatbotApp:
    """Main application class with chat-first architecture."""

    def __init__(self):
        self.session_manager = SessionManager()
        self._initialize_prompt_manager()
        self._initialize_ui_components()

    def _initialize_prompt_manager(self):
        """Initialize prompt management system."""
        self.session_manager.initialize_prompt_manager()

    def _initialize_ui_components(self):
        """Initialize all UI components."""
        self.sidebar_nav = SidebarNavigation(self.session_manager)
        self.chat_ui = ChatMainUI(self.session_manager)
        self.upload_ui = DataUploadUI(self.session_manager)
        self.data_management_ui = DataManagementUI(self.session_manager)

    def run(self):
        """Run the application."""
        # Page config
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Render sidebar (always visible)
        self.sidebar_nav.render()

        # Get current page
        current_page = self.session_manager.get("current_page", PAGE_CHAT)

        # Render appropriate page
        if current_page == PAGE_UPLOAD:
            self._render_upload_page()
        elif current_page == PAGE_DATA_MANAGEMENT:
            self._render_data_management_page()
        else:
            self._render_chat_page()

    def _render_chat_page(self) -> None:
        """Render main chat page."""
        # Header
        st.title("💬 Chat with Your Documents")
        st.caption(
            "Ask questions and get AI-powered answers. "
            "Upload documents via sidebar for enhanced responses."
        )
        st.markdown("---")

        # Render chat interface
        try:
            self.chat_ui.render()
        except Exception as e:
            st.error(f"Error in chat interface: {str(e)}")

    def _render_upload_page(self) -> None:
        """Render upload page."""
        # Header
        st.title("📁 Upload Documents")
        st.caption("Upload and process documents for the RAG system.")

        # Reset button
        col2 = st.columns([3, 1])
        with col2[1]:
            if st.button("🗑️ Reset All", help="Clear all data"):
                if st.session_state.get("confirm_reset"):
                    self.session_manager.clear()
                    st.success("✅ Reset complete!")
                    st.rerun()
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ Click again to confirm")

        st.markdown("---")

        # Render upload interface
        try:
            self.upload_ui.render(header_number=1)
        except Exception as e:
            st.error(f"Error in upload interface: {str(e)}")

        # Show data status if saved
        if self.session_manager.get("data_saved_success"):
            st.markdown("---")
            self._render_upload_success()

    def _render_data_management_page(self) -> None:
        """Render data management page."""
        # Header
        st.title("🗂️ Data Management")
        st.caption("Manage your documents, collections, and data operations.")
        st.markdown("---")

        # Render data management interface
        try:
            self.data_management_ui.render(header_number=1)
        except Exception as e:
            st.error(f"Error in data management interface: {str(e)}")

    def _render_upload_success(self) -> None:
        """Render upload success status."""
        st.success("✅ Data uploaded successfully!")

        chunks_df = self.session_manager.get("chunks_df")
        if chunks_df is not None and not chunks_df.empty:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Chunks", len(chunks_df))

            with col2:
                st.metric("Columns", len(chunks_df.columns))

            with col3:
                avg_length = (
                    chunks_df['chunk'].str.len().mean()
                    if 'chunk' in chunks_df.columns
                    else 0
                )
                st.metric("Avg Length", f"{int(avg_length)} chars")

        # Button to go back to chat
        if st.button("💬 Go to Chat", type="primary"):
            self.session_manager.set("current_page", PAGE_CHAT)
            st.rerun()


def main():
    """Main entry point."""
    app = RAGChatbotApp()
    app.run()


if __name__ == "__main__":
    main()
