import streamlit as st
from dotenv import load_dotenv

from backend.session_manager import SessionManager
from ui.chatbot_interface import ChatbotUI
from ui.data_upload import DataUploadUI
from ui.language_setup import LanguageSetupUI
from ui.llm_setup import LLMSetupUI
from ui.sidebar_config import SidebarConfig

load_dotenv()


class RAGChatbotApp:
    def __init__(self):
        self.session_manager = SessionManager()
        self._initialize_ui_components()

    def _initialize_ui_components(self):
        """Initialize all UI components."""
        self.language_ui = LanguageSetupUI(self.session_manager)
        self.data_ui = DataUploadUI(self.session_manager)
        self.llm_ui = LLMSetupUI(self.session_manager)
        self.chatbot_ui = ChatbotUI(self.session_manager)
        self.sidebar = SidebarConfig(self.session_manager)

    def run(self):
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Main header
        st.title("RAG Chatbot")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("Design your own chatbot using the RAG system.")

        with col2:
            if st.button("🗑️ Reset All", help="Clear all data and start over"):
                if st.session_state.get("confirm_reset"):
                    self.session_manager.clear()
                    st.success("✅ Reset complete!")
                    st.rerun()
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ Click again to confirm reset")

        st.markdown("---")

        # Section counter
        section_num = 1

        # 1. Language Setup
        try:
            self.language_ui.render(section_num)
            st.markdown("---")
            section_num += 1
        except Exception as e:
            st.error(f"Error in Language Setup: {str(e)}")

        # 2. Data Source Setup
        try:
            self.data_ui.render(section_num)
            st.markdown("---")
            section_num += 1
        except Exception as e:
            st.error(f"Error in Data Upload: {str(e)}")

        # 3. Data Status
        self._render_data_status(section_num)
        st.markdown("---")
        section_num += 1

        # 4. Column Selection (if data is saved)
        if self.session_manager.get("data_saved_success"):
            self._render_column_selection(section_num)
            st.markdown("---")
            section_num += 1

        # 5. LLM Setup
        try:
            self.llm_ui.render(section_num)
            st.markdown("---")
            section_num += 1
        except Exception as e:
            st.error(f"Error in LLM Setup: {str(e)}")

        # 6. Interactive Chatbot
        try:
            self.chatbot_ui.render(section_num)
        except Exception as e:
            st.error(f"Error in Chatbot: {str(e)}")

        # Render sidebar configuration
        try:
            self.sidebar.render()
        except Exception as e:
            st.sidebar.error(f"Error in sidebar: {str(e)}")

    def _render_data_status(self, header_num: int):
        """
        Render data status section.

        Args:
            header_num: Section number
        """
        header_text = f"{header_num}. Data Status"

        if self.session_manager.get("data_saved_success"):
            header_text += " ✅"
            st.header(header_text)

            col1, col2 = st.columns(2)
            with col1:
                st.success("✅ Data saved successfully!")
            with col2:
                collection_name = self.session_manager.get("collection_name")
                if collection_name:
                    st.info(f"📦 Collection: `{collection_name}`")

            # Show statistics
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
                    st.metric("Avg Chunk Length", f"{int(avg_length)} chars")
        else:
            st.header(header_text)
            st.warning("⚠️ No data loaded yet. Please upload and save data first.")

    def _render_column_selection(self, header_num: int):
        """
        Render column selection for chatbot answers.

        Args:
            header_num: Section number
        """
        st.header(f"{header_num}. Select Answer Columns")

        chunks_df = self.session_manager.get("chunks_df")

        if chunks_df is not None and not chunks_df.empty:
            st.info(
                "💡 Select which columns contain information for the chatbot to use"
            )

            selected_columns = st.multiselect(
                "Select columns:",
                chunks_df.columns.tolist(),
                default=self.session_manager.get("columns_to_answer", []),
                help="Choose one or more columns that contain relevant information",
            )

            self.session_manager.set("columns_to_answer", selected_columns)

            if selected_columns:
                st.success(
                    f"✅ Selected {len(selected_columns)} column(s): {', '.join(selected_columns)}"
                )
            else:
                st.warning("⚠️ Please select at least one column")
        else:
            st.warning("No data available. Please upload and save data first.")


def main():
    """Main entry point for the application."""
    app = RAGChatbotApp()
    app.run()


if __name__ == "__main__":
    main()
