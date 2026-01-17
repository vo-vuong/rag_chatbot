"""
Main Chat Interface - Default page for RAG Chatbot.
"""

import logging
from typing import List

import httpx
import streamlit as st

from backend.session_manager import SessionManager
from ui.api_client import get_api_client, UIChunk, UIImage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                # Display route indicator for assistant messages
                if message["role"] == "assistant" and message.get("route"):
                    route = message["route"]
                    if route == "text_only":
                        st.caption("🔤 Query type: **Text Search**")
                    elif route == "image_only":
                        st.caption("🖼️ Query type: **Image Search**")

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

                    # Get image data from session state (now typed UIImage list)
                    images: List[UIImage] = st.session_state.get("last_response_images", [])
                    chat_mode = self.session_manager.get("chat_mode", "rag")  # Default to rag

                    # Get route info from session state
                    route = st.session_state.get("last_response_route")

                    logger.info(
                        f"After response: images={len(images)}, "
                        f"chat_mode={chat_mode}, route={route}"
                    )

                    # Display route indicator (only in RAG mode)
                    if route and chat_mode == "rag":
                        if route == "text_only":
                            st.caption("🔤 Query type: **Text Search**")
                        elif route == "image_only":
                            st.caption("🖼️ Query type: **Image Search**")

                    # Display response
                    st.markdown(response)

                    # Display images if available
                    if images and chat_mode == "rag":
                        logger.info("Calling _display_response_images")
                        self._display_response_images(images)
                    else:
                        logger.info(
                            f"Skipping image display: images={bool(images)}, "
                            f"mode={chat_mode}"
                        )

                    # Save message with images and route to chat history
                    image_paths = [img.image_path for img in images]
                    image_captions = [img.caption for img in images]
                    self._add_message("assistant", response, image_paths, image_captions, route)

                    # Clear stored images after display
                    if "last_response_images" in st.session_state:
                        del st.session_state.last_response_images
                    # Clear route after display
                    if "last_response_route" in st.session_state:
                        del st.session_state.last_response_route

    def _add_message(self, role: str, content: str, image_paths: List[str] = None, image_captions: List[str] = None, route: str = None) -> None:
        """Add message to chat history with optional images and route."""
        chat_history = self.session_manager.get("chat_history", [])
        message = {"role": role, "content": content}

        # Add image data if available
        if image_paths:
            message["image_paths"] = image_paths
            message["image_captions"] = image_captions or []

        # Add route info if available
        if route:
            message["route"] = route

        chat_history.append(message)
        self.session_manager.set("chat_history", chat_history)

    def _generate_response(self, query: str) -> str:
        """Generate response via FastAPI backend."""
        try:
            api_client = get_api_client()

            selected_mode = self.session_manager.get("chat_mode", "rag")
            num_docs = self.session_manager.get("number_docs_retrieval", 3)
            score_threshold = self.session_manager.get("score_threshold", 0.5)

            result = api_client.chat(
                query=query,
                mode=selected_mode,
                top_k=num_docs,
                score_threshold=score_threshold,
            )

            # Display retrieved chunks in sidebar
            if result.retrieved_chunks:
                self._display_sidebar_text_results(result.retrieved_chunks)

            # Store images for display (using typed UIImage list)
            if result.images:
                st.session_state.last_response_images = result.images
            else:
                st.session_state.last_response_images = []

            # Store route info for display
            st.session_state.last_response_route = result.route

            return result.response

        except httpx.ConnectError:
            logger.error("API connection failed")
            return (
                "API unavailable. Please ensure FastAPI backend is running "
                "on port 8000.\n\nRun: `uvicorn api.main:app --reload --port 8000`"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"API returned error status: {e.response.status_code}")
            return "Request failed. Please try again."
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return "An error occurred. Please try again."

    def _display_sidebar_text_results(self, retrieved_chunks: List[UIChunk]) -> None:
        """Display retrieved text chunks in sidebar."""
        with st.sidebar.expander("📄 Retrieved Documents", expanded=False):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"**Doc {i}** (Score: {chunk.score:.3f})")
                st.caption(f"📁 Source: `{chunk.source_file}`")
                if chunk.page_number:
                    st.caption(f"📄 Page: {chunk.page_number}")
                st.text(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
                st.markdown("---")

    def _display_response_images(self, images: List[UIImage]) -> None:
        """
        Display images with captions below response.

        Args:
            images: List of UIImage objects with caption, image_path, source_file
        """
        import html

        logger.info(f"Displaying {len(images)} image(s)")

        st.markdown("### 📸 Relevant Image")

        for i, img in enumerate(images):
            try:
                logger.info(f"Validating image {i+1}: {img.image_path}")
                if self._validate_image_file(img.image_path):
                    # Security: Sanitize caption to prevent XSS
                    safe_caption = html.escape(img.caption)

                    # Display source document info
                    st.caption(f"📁 Source: `{img.source_file}`")
                    if img.page_number:
                        st.caption(f"📄 Page: {img.page_number}")

                    # Display image with caption
                    st.image(
                        img.image_path,
                        caption=f"📌 {safe_caption}",
                        width=600,
                        output_format="auto"
                    )
                    logger.info(f"Image displayed successfully: {img.image_path}")

                    # Show image metadata in expander
                    with st.expander("ℹ️ Image Details", expanded=False):
                        st.text(f"Source: {img.source_file}")
                        st.text(f"Path: {img.image_path}")
                        st.text(f"Caption: {safe_caption}")
                        st.text(f"Score: {img.score:.3f}")

                else:
                    from pathlib import Path
                    st.error(f"🖼️ Image not found or invalid: {Path(img.image_path).name}")
                    logger.warning(f"Image validation failed: {img.image_path}")

            except Exception as e:
                st.error(f"🖼️ Error displaying image: {str(e)}")
                logger.error(f"Failed to display image {img.image_path}: {e}")

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
