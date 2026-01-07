"""
Main Chat Interface - Default page for RAG Chatbot.
"""

import logging
import os
from typing import List, Dict, Any

import streamlit as st

from backend.session_manager import SessionManager
from backend.routing import QueryRouter

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

    def _get_query_router(self) -> QueryRouter:
        """Get or create QueryRouter instance."""
        if not hasattr(self, '_query_router'):
            api_key = self.session_manager.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in session or environment")
            self._query_router = QueryRouter(openai_api_key=api_key)
        return self._query_router

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

                # Display images if available in message
                if message.get("image_paths"):
                    image_paths = message.get("image_paths", [])
                    image_captions = message.get("image_captions", [])
                    self._display_response_images(image_paths, image_captions)

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

                    # Get image data from session state
                    image_paths = st.session_state.get("last_response_images", [])
                    image_captions = st.session_state.get("last_response_image_captions", [])
                    chat_mode = self.session_manager.get("chat_mode", "rag")  # Default to rag

                    logger.info(
                        f"After response: image_paths={image_paths}, "
                        f"chat_mode={chat_mode}"
                    )

                    # Display response
                    st.markdown(response)

                    # Display images if available
                    if image_paths and chat_mode == "rag":
                        logger.info("Calling _display_response_images")
                        self._display_response_images(image_paths, image_captions)
                    else:
                        logger.info(
                            f"Skipping image display: paths={bool(image_paths)}, "
                            f"mode={chat_mode}"
                        )

                    # Save message with images to chat history
                    self._add_message("assistant", response, image_paths, image_captions)

                    # Clear stored images after display
                    if "last_response_images" in st.session_state:
                        del st.session_state.last_response_images
                    if "last_response_image_captions" in st.session_state:
                        del st.session_state.last_response_image_captions

    def _add_message(self, role: str, content: str, image_paths: List[str] = None, image_captions: List[str] = None) -> None:
        """Add message to chat history with optional images."""
        chat_history = self.session_manager.get("chat_history", [])
        message = {"role": role, "content": content}

        # Add image data if available
        if image_paths:
            message["image_paths"] = image_paths
            message["image_captions"] = image_captions or []

        chat_history.append(message)
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
        """Generate response using routed single-collection search."""
        try:
            # Step 1: Classify query using QueryRouter
            router = self._get_query_router()
            classification = router.classify(query)

            logger.info(f"Query routed to: {classification.route} (reason: {classification.reasoning})")

            # Step 2: Route to appropriate handler
            if classification.route == "text_only":
                logger.info("Call text_only response")
                return self._generate_text_response(
                    query, llm_model, qdrant_manager, embedding_strategy
                )
            else:  # image_only
                return self._generate_image_response(
                    query, llm_model, embedding_strategy
                )

        except Exception as e:
            logger.error(f"Router error, falling back to text: {e}")
            # Fallback to text_only on router error
            return self._generate_text_response(
                query, llm_model, qdrant_manager, embedding_strategy
            )

    def _generate_text_response(
        self, query: str, llm_model, qdrant_manager, embedding_strategy
    ) -> str:
        """Generate response from text collection only."""
        try:
            logger.info(f"User query: {query}")
            # Embed query
            query_embedding = embedding_strategy.embed_query(query)

            # Search text collection
            num_docs = self.session_manager.get("number_docs_retrieval", 3)
            score_threshold = self.session_manager.get("score_threshold", 0.5)

            text_results = qdrant_manager.search(
                query_vector=query_embedding,
                top_k=num_docs,
                score_threshold=score_threshold,
            )

            if not text_results:
                return self._generate_llm_only_response(query, llm_model) + (
                    "\n\n*Note: No relevant documents found.*"
                )

            # Extract chunks
            retrieved_chunks = [
                (result["payload"].get("chunk", ""), result["score"])
                for result in text_results
            ]

            # Display in sidebar
            self._display_sidebar_text_results(retrieved_chunks)

            # Build context
            if PROMPT_BUILDER_AVAILABLE:
                chunks_only = [chunk for chunk, _ in retrieved_chunks]
                scores_only = [score for _, score in retrieved_chunks]
                context = PromptBuilder.format_context(
                    chunks_only, scores_only, include_scores=False
                )
            else:
                context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])

            # Generate response
            rag_template = self.session_manager.get_active_rag_template()
            chat_history = self.session_manager.get("chat_history", [])

            if PROMPT_BUILDER_AVAILABLE:
                # Convert chat_history list to formatted string for template
                chat_history_str = (
                    PromptBuilder.format_chat_history(chat_history)
                    if chat_history
                    else "No previous conversation."
                )

                full_prompt = PromptBuilder.build_rag_prompt(
                    query=query,
                    context=context,
                    template=rag_template,
                    chat_history=chat_history_str,
                )
                response = llm_model.generate_content(prompt=full_prompt)
            else:
                response = llm_model.generate_content(prompt=query, context=context)

            # Attribution
            response += f"\n\n---\n*📚 Answer based on {len(retrieved_chunks)} text document(s)*"

            # Clear image state (text-only response)
            st.session_state.last_response_images = []
            st.session_state.last_response_image_captions = []

            return response

        except Exception as e:
            return f"❌ Text RAG Error: {str(e)}"

    def _generate_image_response(
        self, query: str, llm_model, embedding_strategy
    ) -> str:
        """Generate response from image collection only."""
        try:
            # Get image manager
            image_manager = self.session_manager.get_image_qdrant_manager()

            if not image_manager:
                return (
                    "❌ No image collection available. "
                    "Please upload documents with images first."
                )

            # Embed query
            query_embedding = embedding_strategy.embed_query(query)

            # Search image collection
            image_score_threshold = self.session_manager.get_image_score_threshold()
            image_top_k = self.session_manager.get("image_top_k", 3)

            image_results = image_manager.search(
                query_vector=query_embedding,
                top_k=image_top_k,
                score_threshold=image_score_threshold,
            )

            if not image_results:
                return (
                    "❌ No relevant images found matching your query. "
                    "Try rephrasing or ask about text content instead."
                )

            # Get top image
            top_image = image_results[0]
            caption = top_image["payload"].get("chunk", "")
            image_path = top_image["payload"].get("image_path", "")
            image_score = top_image["score"]
            page_number = top_image["payload"].get("page_number")
            source_doc = top_image["payload"].get("source_document", "")

            # Display in sidebar
            self._display_sidebar_image_results(image_results)

            # Build image context
            if PROMPT_BUILDER_AVAILABLE:
                image_context = PromptBuilder.format_image_context(
                    image_caption=caption,
                    page_number=page_number,
                    source_document=source_doc,
                    score=image_score,
                )
            else:
                image_context = f"Image Description: {caption}"

            # Generate response about the image
            chat_history = self.session_manager.get("chat_history", [])

            if PROMPT_BUILDER_AVAILABLE:
                full_prompt = PromptBuilder.build_rag_prompt_with_images(
                    query=query,
                    text_context="",  # No text context
                    image_context=image_context,
                    chat_history=chat_history if chat_history else None,
                )
                response = llm_model.generate_content(prompt=full_prompt)
            else:
                response = llm_model.generate_content(
                    prompt=query,
                    context=image_context
                )

            # Attribution
            response += "\n\n---\n*🖼️ Answer based on 1 image*"

            # Store image for display
            if image_path and self._validate_image_file(image_path):
                st.session_state.last_response_images = [image_path]
                st.session_state.last_response_image_captions = [caption]
                logger.info(f"Stored image for display: {image_path}")
            else:
                st.session_state.last_response_images = []
                st.session_state.last_response_image_captions = []

            return response

        except Exception as e:
            return f"❌ Image RAG Error: {str(e)}"

    def _display_sidebar_text_results(self, retrieved_chunks: List[tuple]) -> None:
        """Display retrieved text chunks in sidebar."""
        with st.sidebar.expander("📄 Retrieved Documents", expanded=False):
            for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                st.markdown(f"**Doc {i}** (Score: {score:.3f})")
                st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                st.markdown("---")

    def _display_sidebar_image_results(self, image_results: List[Dict]) -> None:
        """Display retrieved images in sidebar."""
        with st.sidebar.expander("🖼️ Retrieved Images", expanded=False):
            for i, img_result in enumerate(image_results, 1):
                payload = img_result["payload"]
                img_score = img_result["score"]
                img_caption = payload.get("chunk", "No caption")
                img_path = payload.get("image_path", "")
                page_num = payload.get("page_number", "?")

                st.markdown(f"**Image {i}** (Score: {img_score:.3f})")
                st.markdown(f"📍 Page: {page_num}")

                if img_path and self._validate_image_file(img_path):
                    st.image(img_path, width=200)
                else:
                    st.warning("⚠️ Image file not found")

                caption_display = (
                    img_caption[:150] + "..."
                    if len(img_caption) > 150
                    else img_caption
                )
                st.caption(f"📝 {caption_display}")
                st.markdown("---")

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

    def _display_response_images(
        self,
        image_paths: List[str],
        captions: List[str]
    ) -> None:
        """
        Display images with captions below response.

        Args:
            image_paths: List of image file paths
            captions: List of image captions (GPT-4o Mini generated)
        """
        import html

        logger.info(f"Displaying {len(image_paths)} image(s): {image_paths}")

        st.markdown("### 📸 Relevant Image")

        for i, (img_path, caption) in enumerate(zip(image_paths, captions)):
            try:
                logger.info(f"Validating image {i+1}: {img_path}")
                if self._validate_image_file(img_path):
                    # Security: Sanitize caption to prevent XSS
                    safe_caption = html.escape(caption)

                    # Display image with caption
                    st.image(
                        img_path,
                        caption=f"📌 {safe_caption}",
                        width=600,
                        output_format="auto"
                    )
                    logger.info(f"Image displayed successfully: {img_path}")

                    # Show image metadata in expander
                    with st.expander("ℹ️ Image Details", expanded=False):
                        st.text(f"Path: {img_path}")
                        st.text(f"Caption: {safe_caption}")

                else:
                    from pathlib import Path
                    st.error(f"🖼️ Image not found or invalid: {Path(img_path).name}")
                    logger.warning(f"Image validation failed: {img_path}")

            except Exception as e:
                st.error(f"🖼️ Error displaying image: {str(e)}")
                logger.error(f"Failed to display image {img_path}: {e}")

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
                        width=600,
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
            from pathlib import Path

            # Convert to Path object
            img_path = Path(image_path)
            img_path_resolved = img_path.resolve()

            # Security: Ensure path is within extracted_images directory
            extracted_images_dir = Path("extracted_images").resolve()
            try:
                # Check if path is within allowed directory (Python 3.9+)
                if not img_path_resolved.is_relative_to(extracted_images_dir):
                    logger.warning(f"Security: Image path outside allowed directory: {image_path}")
                    return False
            except (ValueError, AttributeError):
                # Fallback for older Python or invalid paths
                logger.warning(f"Security: Invalid image path: {image_path}")
                return False

            # Validate file existence and properties
            if not img_path_resolved.exists():
                logger.warning(f"Image file not found: {img_path_resolved}")
                return False
            if not img_path_resolved.is_file():
                logger.warning(f"Image path is not a file: {img_path_resolved}")
                return False
            if img_path_resolved.stat().st_size == 0:
                logger.warning(f"Image file is empty: {img_path_resolved}")
                return False
            if img_path_resolved.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'):
                logger.warning(f"Invalid image extension: {img_path_resolved.suffix}")
                return False

            logger.debug(f"Image validation passed: {img_path_resolved}")
            return True

        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False

    def render_sidebar_stats(self) -> None:
        """Render chat statistics in sidebar."""
        chat_history = self.session_manager.get("chat_history", [])

        if chat_history:
            st.sidebar.markdown("### 💬 Chat Stats")
            st.sidebar.metric("Messages", len(chat_history))

            if st.sidebar.button("🗑️ Clear Chat"):
                self.session_manager.clear_chat_history()
                st.rerun()
