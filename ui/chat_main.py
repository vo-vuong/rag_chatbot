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
        """Generate response using dual search (text + images)."""
        try:
            # Step 1: Generate query embedding (once for both searches)
            query_embedding = embedding_strategy.embed_query(query)

            # Step 2: Search text collection (existing)
            num_docs = self.session_manager.get("number_docs_retrieval", 3)
            text_score_threshold = self.session_manager.get("score_threshold", 0.5)

            text_results = qdrant_manager.search(
                query_vector=query_embedding,
                top_k=num_docs,
                score_threshold=text_score_threshold,
            )

            # Step 3: Search image collection (NEW)
            image_manager = self.session_manager.get_image_qdrant_manager()
            image_results = []

            if image_manager and self.session_manager.get("enable_image_search", True):
                image_score_threshold = self.session_manager.get_image_score_threshold()
                image_top_k = self.session_manager.get("image_top_k", 1)

                try:
                    image_results = image_manager.search(
                        query_vector=query_embedding,
                        top_k=image_top_k,
                        score_threshold=image_score_threshold,
                    )

                    if image_results:
                        logger.info(
                            f"Found {len(image_results)} relevant image(s) "
                            f"(top score: {image_results[0]['score']:.3f})"
                        )
                    else:
                        logger.debug("No images found above threshold")

                except Exception as e:
                    logger.warning(f"Image search failed: {e}")
                    image_results = []
            else:
                logger.debug("Image collection not available, skipping image search")

            # Step 4: Check if any results found
            if not text_results:
                # No text chunks found, fall back to LLM only
                return self._generate_llm_only_response(query, llm_model) + (
                    "\n\n*Note: No relevant documents found. "
                    "Answer based on general knowledge.*"
                )

            # Step 5: Extract text context
            retrieved_chunks = []

            for result in text_results:
                chunk = result["payload"].get("chunk", "")
                score = result["score"]
                retrieved_chunks.append((chunk, score))

            # Display retrieved docs in sidebar (existing)
            with st.sidebar.expander("📄 Retrieved Documents", expanded=False):
                for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Doc {i}** (Score: {score:.3f})")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    st.markdown("---")

            # Display retrieved images in sidebar (NEW)
            if image_results:
                with st.sidebar.expander("🖼️ Retrieved Images", expanded=False):
                    for i, img_result in enumerate(image_results, 1):
                        payload = img_result["payload"]
                        img_score = img_result["score"]
                        img_caption = payload.get("chunk", "No caption")
                        img_path = payload.get("image_path", "")
                        page_num = payload.get("page_number", "?")

                        st.markdown(f"**Image {i}** (Score: {img_score:.3f})")
                        st.markdown(f"📍 Page: {page_num}")

                        # Display thumbnail if file exists
                        if img_path and self._validate_image_file(img_path):
                            st.image(img_path, width=200)
                        else:
                            st.warning("⚠️ Image file not found")

                        # Show caption (truncated)
                        caption_display = (
                            img_caption[:150] + "..."
                            if len(img_caption) > 150
                            else img_caption
                        )
                        st.caption(f"📝 {caption_display}")
                        st.markdown("---")

            # Step 6: Build text context (existing)
            if PROMPT_BUILDER_AVAILABLE:
                chunks_only = [chunk for chunk, _ in retrieved_chunks]
                scores_only = [score for _, score in retrieved_chunks]
                text_context = PromptBuilder.format_context(
                    chunks_only, scores_only, include_scores=False
                )
            else:
                text_context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])

            # Step 7: Build image context (NEW)
            image_context = ""
            image_to_display = None
            image_caption = ""

            if image_results:
                top_image = image_results[0]
                caption = top_image["payload"].get("chunk", "")  # Caption stored as "chunk"
                image_path = top_image["payload"].get("image_path", "")
                image_score = top_image["score"]

                # Add to context for LLM
                image_context = f"\n\n**Relevant Image**: {caption}"

                # Store for display
                image_to_display = image_path
                image_caption = caption

                logger.info(
                    f"Including image in context: '{caption}' "
                    f"(score: {image_score:.3f}, path: {image_path})"
                )

            # Step 8: Build RAG prompt with both contexts
            rag_template = self.session_manager.get_active_rag_template()

            if PROMPT_BUILDER_AVAILABLE and rag_template:
                try:
                    chat_history = self.session_manager.get("chat_history", [])

                    # Build combined context
                    full_context = text_context + image_context

                    full_prompt = PromptBuilder.build_rag_prompt(
                        query=query,
                        context=full_context,
                        template=rag_template,
                        chat_history=chat_history,
                    )
                    response = llm_model.generate_content(prompt=full_prompt)
                except Exception as e:
                    logger.error(f"Error building RAG prompt: {e}")
                    response = llm_model.generate_content(
                        prompt=query,
                        context=text_context + image_context
                    )
            else:
                # Direct context passing
                response = llm_model.generate_content(
                    prompt=query,
                    context=text_context + image_context
                )

            # Step 9: Add attribution
            attribution = f"\n\n---\n*📚 Answer based on {len(retrieved_chunks)} text document(s)"
            if image_results:
                attribution += f" and 1 image*"
            else:
                attribution += "*"

            response += attribution

            # Step 10: Store image for display (NEW)
            if image_to_display:
                st.session_state.last_response_images = [image_to_display]
                st.session_state.last_response_image_captions = [image_caption]
                logger.info(f"Stored image in session state: {image_to_display}")
            else:
                st.session_state.last_response_images = []
                st.session_state.last_response_image_captions = []
                logger.debug("No image to store in session state")

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
