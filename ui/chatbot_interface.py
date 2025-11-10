import streamlit as st

from backend.session_manager import SessionManager


class ChatbotUI:
    """UI component for interactive chatbot."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self, header_number: int) -> None:
        st.header(f"{header_number}. Interactive Chatbot")

        # Check prerequisites
        ready, message = self._check_prerequisites()

        if not ready:
            st.warning(f"⚠️ {message}")
            st.info("Complete the setup steps above to start chatting!")

            # Show checklist
            with st.expander("📋 Setup Checklist", expanded=True):
                status = self.session_manager.get_status_summary()
                for component, is_ready in status.items():
                    icon = "✅" if is_ready else "❌"
                    st.markdown(f"{icon} {component}")
            return

        # Chat interface
        st.success("✅ All set! You can now chat with your documents.")

        # Display chat history
        self._display_chat_history()

        # Chat input
        self._handle_chat_input()

        # Clear chat button in sidebar
        if st.sidebar.button("🗑️ Clear Chat History"):
            self.session_manager.clear_chat_history()
            st.rerun()

    def _check_prerequisites(self) -> tuple[bool, str]:
        """
        Check if all prerequisites are met.

        Returns:
            Tuple of (is_ready, message)
        """
        if not self.session_manager.is_language_configured():
            return False, "Please select a language and load embedding model"

        if not self.session_manager.is_llm_configured():
            return False, "Please configure LLM (Online or Local)"

        # TODO: Implement Qdant database
        # if not self.session_manager.is_data_loaded():
        #     return False, "Please upload and save data"

        columns = self.session_manager.get("columns_to_answer", [])
        if not columns:
            return False, "Please select columns for the chatbot to answer from"

        return True, "Ready"

    def _display_chat_history(self) -> None:
        """Display chat history."""
        chat_history = self.session_manager.get("chat_history", [])

        if not chat_history:
            st.info("💬 No messages yet. Start by asking a question below!")
            return

        for message in chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _handle_chat_input(self) -> None:
        """Handle user chat input."""
        # Chat input
        prompt = st.chat_input("Ask me anything about your documents...")

        if prompt:
            # Add user message to history
            self._add_message_to_history("user", prompt)

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self._generate_response(prompt)
                    st.markdown(response)
                    self._add_message_to_history("assistant", response)

    def _add_message_to_history(self, role: str, content: str) -> None:
        """
        Add message to chat history.

        Args:
            role: Message role (user or assistant)
            content: Message content
        """
        chat_history = self.session_manager.get("chat_history", [])
        chat_history.append({"role": role, "content": content})
        self.session_manager.set("chat_history", chat_history)

    def _generate_response(self, query: str) -> str:
        """
        Generate response for user query.

        Args:
            query: User query

        Returns:
            Generated response
        """
        try:
            # Get chunks dataframe
            chunks_df = self.session_manager.get("chunks_df")
            columns_to_answer = self.session_manager.get("columns_to_answer", [])

            if chunks_df is None or chunks_df.empty:
                return "❌ No data available. Please upload and process data first."

            # Simple keyword-based search (placeholder for vector search)
            query_lower = query.lower()
            relevant_chunks = []

            for _, row in chunks_df.iterrows():
                chunk_text = str(row.get('chunk', ''))
                if any(word in chunk_text.lower() for word in query_lower.split()):
                    relevant_chunks.append(chunk_text)

                if len(relevant_chunks) >= 3:  # Limit to top 3
                    break

            if not relevant_chunks:
                return (
                    "❌ I couldn't find relevant information to answer your"
                    " question. Try rephrasing your query."
                )

            # Display retrieved chunks in sidebar
            with st.sidebar.expander("📄 Retrieved Documents", expanded=True):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    st.markdown("---")

            # Create enhanced prompt
            context = "\n\n".join(relevant_chunks)
            enhanced_prompt = f"""Based on the following information:
                {context}
                Please answer this question: {query}
                If the information provided doesn't fully answer the question, say so."""
            # Show prompt in sidebar
            with st.sidebar.expander("🔍 Full Prompt to LLM"):
                st.code(enhanced_prompt, language="text")

            # Check if LLM is configured
            llm_model = self._get_current_llm()

            if llm_model is None:
                return f"""📝 **Based on the retrieved information:**
                    {context[:500]}...
                    💡 **Note:** LLM is not configured yet. The response above is a simple retrieval result. 
                    Configure an LLM in the "Setup LLMs" section for AI-generated answers."""

            # Try to generate response with LLM
            try:
                response = llm_model.generate_content(enhanced_prompt)
                return response
            except Exception as e:
                return f"""📝 **Retrieved Information:**
                    {context[:500]}...
                    ⚠️ **Note:** Could not generate AI response: {str(e)}
                    The information above was retrieved from your documents."""

        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def _get_current_llm(self):
        """
        Get current LLM model.

        Returns:
            LLM model instance or None
        """
        llm_type = self.session_manager.get("llm_type")

        if llm_type == "online_llm":
            return self.session_manager.get("online_llms")
        elif llm_type == "local_llm":
            return self.session_manager.get("local_llms")

        return None
