import streamlit as st

from backend.session_manager import SessionManager


class ChatbotUI:
    """UI component for interactive chatbot with vector search."""

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
        """Check if all prerequisites are met."""
        if not self.session_manager.is_language_configured():
            return False, "Please select a language and configure embeddings"

        if not self.session_manager.is_embedding_configured():
            return False, "Please configure embedding strategy"

        if not self.session_manager.is_llm_configured():
            return False, "Please configure LLM (Online or Local)"

        if not self.session_manager.is_data_loaded():
            return False, "Please upload and save data to vector database"

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
        """Add message to chat history."""
        chat_history = self.session_manager.get("chat_history", [])
        chat_history.append({"role": role, "content": content})
        self.session_manager.set("chat_history", chat_history)

    def _generate_response(self, query: str) -> str:
        """Generate response using vector search + LLM."""
        try:
            # Get components
            qdrant_manager = self.session_manager.get("qdrant_manager")
            embedding_strategy = self.session_manager.get("embedding_strategy")
            language = self.session_manager.get("language")
            num_docs = self.session_manager.get("number_docs_retrieval", 3)

            if not qdrant_manager or not embedding_strategy:
                return "❌ System not properly configured. Please check setup."

            # Step 1: Generate query embedding
            try:
                query_embedding = embedding_strategy.embed_query(query)
            except Exception as e:
                return f"❌ Failed to generate query embedding: {str(e)}"

            # Step 2: Search in Qdrant
            try:
                search_results = qdrant_manager.search(
                    query_vector=query_embedding,
                    language=language,
                    top_k=num_docs,
                    score_threshold=0.5,  # Minimum similarity score
                )
            except Exception as e:
                return f"❌ Search failed: {str(e)}"

            # Step 3: Check if results found
            if not search_results:
                return (
                    "🔍 I couldn't find relevant information to answer your question. "
                    "Try rephrasing or asking about different topics in your documents."
                )

            # Step 4: Extract chunks and display in sidebar
            retrieved_chunks = []
            for result in search_results:
                chunk = result["payload"].get("chunk", "")
                score = result["score"]
                retrieved_chunks.append((chunk, score))

            # Display retrieved documents in sidebar
            with st.sidebar.expander("📄 Retrieved Documents", expanded=True):
                for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Document {i}** (Score: {score:.3f})")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.markdown("---")

            # Step 5: Create context for LLM
            context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])

            # Step 6: Create enhanced prompt
            enhanced_prompt = f"""Based on the following context from the documents:
                {context}
                Please answer this question: {query}
                If the context doesn't contain enough information to fully answer the question, 
                say so and provide what information is available."""

            # Show full prompt in sidebar
            with st.sidebar.expander("📝 Full Prompt to LLM"):
                st.code(enhanced_prompt, language="text")

            # Step 7: Generate response with LLM
            llm_model = self._get_current_llm()
            if llm_model is None:
                # Return retrieval results if no LLM
                return f"""📚 **Retrieved Information** (No LLM configured):
                    {context[:500]}...
                    **Note:** Configure an LLM in the "Setup LLMs" 
                    section for AI-generated answers."""

            # Try to generate with LLM
            try:
                response = llm_model.generate_content(enhanced_prompt)

                # Add source attribution
                response += "\n\n---\n📊 *Response generated from {} retrieved document(s)*".format(
                    len(retrieved_chunks)
                )

                return response
            except Exception as e:
                # Fallback to retrieval results
                return f"""📚 **Retrieved Information:**
                    {context[:500]}...
                    ⚠️ **Note:** Could not generate AI response: {str(e)}
                    The information above was retrieved from your documents."""

        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def _get_current_llm(self):
        """Get current LLM model."""
        llm_type = self.session_manager.get("llm_type")

        if llm_type == "online_llm":
            return self.session_manager.get("online_llms")
        elif llm_type == "local_llm":
            return self.session_manager.get("local_llms")

        return None
