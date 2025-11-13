"""
Main Chat Interface - Default page for RAG Chatbot.
"""

import streamlit as st

from backend.session_manager import SessionManager


class ChatMainUI:
    """Main chat interface component."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self) -> None:
        """Render main chat interface."""
        # Check if system is ready
        if not self._check_system_status():
            return

        # Display chat messages
        self._display_chat_history()

        # Chat input
        self._handle_chat_input()

    def _check_system_status(self) -> bool:
        """Check system status and show appropriate messages."""
        # Check if LLM is configured
        if not self.session_manager.is_llm_configured():
            st.info(
                "👋 **Welcome to RAG Chatbot!**\n\n"
                "To get started:\n"
                "1. Enter your OpenAI API key in the sidebar\n"
                "2. The system will auto-configure\n"
                "3. Start chatting!\n\n"
                "💡 You can upload documents via the sidebar for enhanced answers."
            )
            return False

        # Check if embeddings configured
        if not self.session_manager.is_embedding_configured():
            st.warning("⚠️ Embeddings not configured. Please check sidebar settings.")
            return False

        # System ready
        return True

    def _display_chat_history(self) -> None:
        """Display chat message history."""
        chat_history = self.session_manager.get("chat_history", [])

        if not chat_history:
            # Show welcome message with status
            has_docs = self.session_manager.has_documents()

            if has_docs:
                st.success(
                    "✅ **System Ready with RAG Mode**\n\n"
                    "I have access to your uploaded documents. "
                    "Ask me anything!"
                )
            else:
                st.info(
                    "💬 **System Ready - LLM Mode**\n\n"
                    "I'm ready to chat! However, I don't have access to any documents yet.\n\n"
                    "📁 Upload documents via sidebar for enhanced answers with RAG."
                )
            return

        # Display chat messages
        for message in chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _handle_chat_input(self) -> None:
        """Handle user chat input."""
        prompt = st.chat_input("Ask me anything...")

        if prompt:
            # Add user message
            self._add_message("user", prompt)

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self._generate_response(prompt)
                    st.markdown(response)
                    self._add_message("assistant", response)

    def _add_message(self, role: str, content: str) -> None:
        """Add message to chat history."""
        chat_history = self.session_manager.get("chat_history", [])
        chat_history.append({"role": role, "content": content})
        self.session_manager.set("chat_history", chat_history)

    def _generate_response(self, query: str) -> str:
        """Generate response with or without RAG."""
        try:
            llm_model = self.session_manager.get("llm_model")
            qdrant_manager = self.session_manager.get("qdrant_manager")
            embedding_strategy = self.session_manager.get("embedding_strategy")

            # Check if we can do RAG
            can_rag = (
                qdrant_manager is not None
                and embedding_strategy is not None
                and self.session_manager.has_documents()
            )

            if can_rag:
                return self._generate_rag_response(
                    query, llm_model, qdrant_manager, embedding_strategy
                )
            else:
                return self._generate_llm_only_response(query, llm_model)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _generate_rag_response(
        self, query: str, llm_model, qdrant_manager, embedding_strategy
    ) -> str:
        """Generate response using RAG (Retrieval + LLM)."""
        try:
            # Step 1: Generate query embedding
            query_embedding = embedding_strategy.embed_query(query)

            # Step 2: Search Qdrant
            num_docs = self.session_manager.get("number_docs_retrieval", 3)
            score_threshold = self.session_manager.get("score_threshold", 0.5)

            search_results = qdrant_manager.search(
                query_vector=query_embedding,
                top_k=num_docs,
                score_threshold=score_threshold,
            )

            # Step 3: Check results
            if not search_results:
                # No relevant documents, fall back to LLM only
                return self._generate_llm_only_response(query, llm_model) + (
                    "\n\n*Note: No relevant documents found. "
                    "Answer based on general knowledge.*"
                )

            # Step 4: Extract context
            retrieved_chunks = []
            for result in search_results:
                chunk = result["payload"].get("chunk", "")
                score = result["score"]
                retrieved_chunks.append((chunk, score))

            # Display retrieved docs in sidebar
            with st.sidebar.expander("📄 Retrieved Documents", expanded=False):
                for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Doc {i}** (Score: {score:.3f})")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    st.markdown("---")

            # Step 5: Build context
            context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])

            # Step 6: Generate response with context
            response = llm_model.generate_content(prompt=query, context=context)

            # Add attribution
            response += (
                f"\n\n---\n*📚 Answer based on {len(retrieved_chunks)} "
                f"retrieved document(s)*"
            )

            return response

        except Exception as e:
            return f"❌ RAG Error: {str(e)}"

    def _generate_llm_only_response(self, query: str, llm_model) -> str:
        """Generate response using LLM only (no RAG)."""
        try:
            # Get chat history for context
            chat_history = self.session_manager.get("chat_history", [])

            # Generate response
            response = llm_model.generate_content(
                prompt=query, chat_history=chat_history
            )

            return response

        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

    def render_sidebar_stats(self) -> None:
        """Render chat statistics in sidebar."""
        chat_history = self.session_manager.get("chat_history", [])

        if chat_history:
            st.sidebar.markdown("### 💬 Chat Stats")
            st.sidebar.metric("Messages", len(chat_history))

            if st.sidebar.button("🗑️ Clear Chat"):
                self.session_manager.clear_chat_history()
                st.rerun()
