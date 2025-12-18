"""
Main Chat Interface - Default page for RAG Chatbot.
"""

import logging
import os
from typing import List, Dict, Any

import streamlit as st

from backend.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import PromptBuilder for prompt construction
try:
    from backend.prompts.prompt_builder import PromptBuilder

    PROMPT_BUILDER_AVAILABLE = True
except ImportError:
    PROMPT_BUILDER_AVAILABLE = False


class ChatMainUI:
    """Main chat interface component."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self) -> None:
        """Render main chat interface."""
        # Check if system is ready
        if not self._check_system_status():
            return

        # Render mode selector
        self._render_mode_selector()

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

    def _render_mode_selector(self) -> None:
        """Render chat mode selector."""
        # Check if RAG is available
        qdrant_manager = self.session_manager.get("qdrant_manager")
        embedding_strategy = self.session_manager.get("embedding_strategy")
        has_documents = self.session_manager.has_documents()

        can_use_rag = (
            qdrant_manager is not None
            and embedding_strategy is not None
            and has_documents
        )

        # Create mode options
        if can_use_rag:
            mode_options = {
                "🤖 RAG Mode (with Documents)": "rag",
                "💬 Chat Mode (LLM Only)": "llm_only",
            }
            default_mode = "rag"
        else:
            # Only LLM mode available
            mode_options = {"💬 Chat Mode (LLM Only)": "llm_only"}
            default_mode = "llm_only"

        # Get current mode from session
        current_mode = self.session_manager.get("chat_mode", default_mode)

        # Find current index
        mode_values = list(mode_options.values())
        try:
            current_index = mode_values.index(current_mode)
        except ValueError:
            current_index = 0

        # Render selector
        col1, col2 = st.columns([3, 1])

        with col1:
            selected_mode_label = st.selectbox(
                "🎯 Chat Mode",
                options=list(mode_options.keys()),
                index=current_index,
                help=(
                    "**RAG Mode**: Uses retrieved documents to answer questions\n\n"
                    "**Chat Mode**: Direct conversation with LLM without document retrieval"
                ),
                key="mode_selector",
            )

            # Update session state if changed
            selected_mode = mode_options[selected_mode_label]
            if selected_mode != current_mode:
                self.session_manager.set("chat_mode", selected_mode)
                st.rerun()

        with col2:
            # Show status indicator
            if selected_mode == "rag":
                st.metric("📚 Status", "RAG Active", help="Using document retrieval")
            else:
                st.metric("💭 Status", "Chat Only", help="LLM without retrieval")

        # Show info message about current mode
        if selected_mode == "rag":
            st.info(
                "📖 **RAG Mode Active** - I'll search your documents to provide accurate answers."
            )
        else:
            st.info(
                "💬 **Chat Mode Active** - I'll answer using my general knowledge "
                "without searching documents."
            )

    def _display_chat_history(self) -> None:
        """Display chat message history."""
        chat_history = self.session_manager.get("chat_history", [])

        if not chat_history:
            # Show welcome message with status
            has_docs = self.session_manager.has_documents()

            if not has_docs:
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

                    # Display response with images
                    image_paths = st.session_state.get("last_response_images", [])
                    if image_paths and self.session_manager.get("chat_mode") == "rag":
                        # Use the enhanced display method for RAG responses with images
                        self.display_response_with_images(response, image_paths)
                    else:
                        # Standard text display for LLM-only or no images
                        st.markdown("### 🤖 Assistant Response")
                        st.markdown(response)

                    self._add_message("assistant", response)

                    # Clear stored images after display
                    if "last_response_images" in st.session_state:
                        del st.session_state.last_response_images

    def _add_message(self, role: str, content: str) -> None:
        """Add message to chat history."""
        chat_history = self.session_manager.get("chat_history", [])
        chat_history.append({"role": role, "content": content})
        self.session_manager.set("chat_history", chat_history)

    def _generate_response(self, query: str) -> str:
        """Generate response based on selected mode."""
        try:
            llm_model = self.session_manager.get("llm_model")
            qdrant_manager = self.session_manager.get("qdrant_manager")
            embedding_strategy = self.session_manager.get("embedding_strategy")

            # Get selected mode from session
            selected_mode = self.session_manager.get("chat_mode", "rag")

            # Check if RAG mode is selected
            if selected_mode == "rag":
                return self._generate_rag_response(
                    query, llm_model, qdrant_manager, embedding_strategy
                )
            else:
                # LLM-only mode
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

            # Step 4: Extract context and images
            retrieved_chunks = []
            all_image_paths = []

            for result in search_results:
                chunk = result["payload"].get("chunk", "")
                score = result["score"]
                retrieved_chunks.append((chunk, score))

                # Extract image paths from metadata
                image_paths = result["payload"].get("image_paths", [])
                if image_paths:
                    all_image_paths.extend(image_paths)

            # Remove duplicate image paths while preserving order
            unique_image_paths = list(dict.fromkeys(all_image_paths))

            # Display retrieved docs in sidebar
            with st.sidebar.expander("📄 Retrieved Documents", expanded=False):
                for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Doc {i}** (Score: {score:.3f})")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    st.markdown("---")

            # Step 5: Build context using PromptBuilder
            if PROMPT_BUILDER_AVAILABLE:
                # Format context with scores
                chunks_only = [chunk for chunk, _ in retrieved_chunks]
                scores_only = [score for _, score in retrieved_chunks]
                context = PromptBuilder.format_context(
                    chunks_only, scores_only, include_scores=False
                )
            else:
                # Legacy format
                context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])

            # Step 6: Build RAG prompt using active template
            # Get active RAG template from SessionManager
            rag_template = self.session_manager.get_active_rag_template()

            if PROMPT_BUILDER_AVAILABLE and rag_template:
                # Use PromptBuilder with active template
                try:
                    # Get chat history for templates that support it (e.g., rag_qa_with_history)
                    chat_history = self.session_manager.get("chat_history", [])

                    full_prompt = PromptBuilder.build_rag_prompt(
                        query=query,
                        context=context,
                        template=rag_template,
                        chat_history=chat_history,  # Pass history for templates that need it
                    )
                    # Generate response with the built prompt
                    response = llm_model.generate_content(prompt=full_prompt)
                except Exception as e:
                    logger.error(f"Error building RAG prompt: {e}")
                    # Fallback to direct context passing
                    response = ""
            else:
                # Direct context passing (LLM will build prompt internally)
                response = llm_model.generate_content(prompt=query, context=context)

            # Add attribution
            response += (
                f"\n\n---\n*📚 Answer based on {len(retrieved_chunks)} "
                f"retrieved document(s)*"
            )

            # Step 7: Display response with images
            # This will display the images directly in the chat UI
            if unique_image_paths:
                # Store images in session state for display in chat message
                st.session_state.last_response_images = unique_image_paths
            else:
                st.session_state.last_response_images = []

            return response

        except Exception as e:
            return f"❌ RAG Error: {str(e)}"

    def _generate_llm_only_response(self, query: str, llm_model) -> str:
        """Generate response using LLM only (no RAG)."""
        try:
            # Get chat history for context
            chat_history = self.session_manager.get("chat_history", [])

            # Get active chat template from SessionManager
            chat_template = self.session_manager.get_active_chat_template()

            if PROMPT_BUILDER_AVAILABLE and chat_template:
                # Use PromptBuilder with active chat template
                try:
                    full_prompt = PromptBuilder.build_chat_prompt(
                        query=query, chat_history=chat_history, template=chat_template
                    )
                    # Generate response with the built prompt
                    response = llm_model.generate_content(prompt=full_prompt)
                except Exception as e:
                    logger.error(f"Error building chat prompt: {e}")
                    # Fallback to direct chat history passing
                    response = llm_model.generate_content(
                        prompt=query, chat_history=chat_history
                    )
            else:
                # Direct chat history passing (LLM will handle internally)
                response = llm_model.generate_content(
                    prompt=query, chat_history=chat_history
                )

            return response

        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

    def display_response_with_images(self, response_text: str, image_paths: List[str]) -> None:
        """Display text response with associated images below."""
        # Display text response (existing functionality)
        st.markdown("### 🤖 Assistant Response")
        st.markdown(response_text)

        # Display images if available
        if image_paths:
            st.markdown("### 📎 Related Images")
            self._display_images(image_paths)

    def _display_images(self, image_paths: List[str]) -> None:
        """Display images with error handling."""
        for i, img_path in enumerate(image_paths):
            try:
                if self._validate_image_file(img_path):
                    # Display image with caption
                    st.image(
                        img_path,
                        caption=f"Extracted from document (Image {i+1})",
                        use_column_width=True,
                        output_format="auto"
                    )
                else:
                    # Show error for missing image
                    st.error(f"🖼️ Image {i+1}: File not found or corrupted")

            except Exception as e:
                st.error(f"🖼️ Image {i+1}: Error loading image - {str(e)}")

    def _validate_image_file(self, image_path: str) -> bool:
        """Validate if image file is accessible and valid."""
        try:
            return (
                os.path.exists(image_path) and
                os.path.isfile(image_path) and
                os.path.getsize(image_path) > 0 and
                image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
            )
        except Exception:
            return False

    def _get_relevant_chunks_with_images(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve chunks with image metadata."""
        # Get session managers
        qdrant_manager = self.session_manager.get("qdrant_manager")
        embedding_strategy = self.session_manager.get("embedding_strategy")

        # Generate query embedding
        query_embedding = embedding_strategy.embed_query(query)

        # Search Qdrant
        num_docs = self.session_manager.get("number_docs_retrieval", 3)
        score_threshold = self.session_manager.get("score_threshold", 0.5)

        search_results = qdrant_manager.search(
            query_vector=query_embedding,
            top_k=num_docs,
            score_threshold=score_threshold,
        )

        # Enhance with image information
        enhanced_chunks = []
        for chunk_data in search_results:
            chunk_dict = {
                'text': chunk_data.get("payload", {}).get("chunk", ""),
                'metadata': chunk_data.get("payload", {}),
                'image_paths': chunk_data.get("payload", {}).get("image_paths", [])
            }
            enhanced_chunks.append(chunk_dict)

        return enhanced_chunks

    def render_sidebar_stats(self) -> None:
        """Render chat statistics in sidebar."""
        chat_history = self.session_manager.get("chat_history", [])

        if chat_history:
            st.sidebar.markdown("### 💬 Chat Stats")
            st.sidebar.metric("Messages", len(chat_history))

            if st.sidebar.button("🗑️ Clear Chat"):
                self.session_manager.clear_chat_history()
                st.rerun()
