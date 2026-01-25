"""
Main Chat Interface - Default page for RAG Chatbot.
"""

import logging
from typing import List

import httpx
import streamlit as st

from backend.session_manager import SessionManager
from config.constants import DEFAULT_NUM_RETRIEVAL, DEFAULT_SCORE_THRESHOLD
from ui.api_client import UIChunk, UIImage, get_api_client

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
            # Show welcome message
            st.info(
                "💬 **System Ready - RAG Mode**\n\n"
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
                    images: List[UIImage] = st.session_state.get(
                        "last_response_images", []
                    )

                    # Get route info from session state
                    route = st.session_state.get("last_response_route")

                    logger.info(
                        f"After response: images={len(images)}, "
                        f"route={route}"
                    )

                    # Display route indicator
                    if route:
                        if route == "text_only":
                            st.caption("🔤 Query type: **Text Search**")
                        elif route == "image_only":
                            st.caption("🖼️ Query type: **Image Search**")

                    # Display response
                    st.markdown(response)

                    # Display images if available
                    if images:
                        logger.info("Calling _display_response_images")
                        self._display_response_images(images)
                    else:
                        logger.info(
                            f"Skipping image display: images={bool(images)}"
                        )

                    # Save message with images and route to chat history
                    image_paths = [img.image_path for img in images]
                    image_captions = [img.caption for img in images]
                    self._add_message(
                        "assistant", response, image_paths, image_captions, route
                    )

                    # Clear stored images after display
                    if "last_response_images" in st.session_state:
                        del st.session_state.last_response_images
                    # Clear route after display
                    if "last_response_route" in st.session_state:
                        del st.session_state.last_response_route

    def _add_message(
        self,
        role: str,
        content: str,
        image_paths: List[str] = None,
        image_captions: List[str] = None,
        route: str = None,
    ) -> None:
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

            # Always RAG mode
            num_docs = self.session_manager.get("number_docs_retrieval", DEFAULT_NUM_RETRIEVAL)
            score_threshold = self.session_manager.get("score_threshold", DEFAULT_SCORE_THRESHOLD)

            result = api_client.chat(
                query=query,
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
                if chunk.point_id:
                    st.caption(f"🔑 Point ID: `{chunk.point_id}`")
                st.text(
                    chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                )
                st.markdown("---")

    def _display_response_images(self, images: List[UIImage], captions: List[str] = None) -> None:
        """
        Display images with captions below response.
        
        Args:
            images: List of UIImage objects or list of image paths (backward compatibility)
            captions: List of captions if images is list of paths
        """
        import html
        from pathlib import Path

        # Handle backward compatibility if images is list of strings (paths)
        if images and isinstance(images[0], str):
            logger.info("Using backward compatibility for image display")
            # Convert to UIImage-like objects for display logic
            image_objs = []
            for i, path in enumerate(images):
                caption = captions[i] if captions and i < len(captions) else "Image"
                # Create dummy UIImage for display
                image_objs.append(UIImage(
                    caption=caption,
                    image_path=path,
                    score=0.0,
                    source_file="Unknown"
                ))
            images = image_objs
        
        logger.info(f"Displaying {len(images)} image(s)")

        st.markdown("### 📸 Relevant Image")

        for i, img in enumerate(images):
            try:
                # Use img.image_path directly
                image_path = img.image_path
                
                logger.info(f"Validating image {i+1}: {image_path}")
                if self._validate_image_file(image_path):
                    # Security: Sanitize caption to prevent XSS
                    safe_caption = html.escape(img.caption)

                    # Display source document info
                    st.caption(f"📁 Source: `{img.source_file}`")
                    if hasattr(img, 'page_number') and img.page_number:
                        st.caption(f"📄 Page: {img.page_number}")

                    # Display image with caption
                    st.image(
                        image_path,
                        caption=f"📌 {safe_caption}",
                        width=600,
                        output_format="auto",
                    )
                    logger.info(f"Image displayed successfully: {image_path}")

                    # Show image metadata in expander
                    with st.expander("ℹ️ Image Details", expanded=False):
                        st.text(f"Source: {img.source_file}")
                        st.text(f"Path: {image_path}")
                        st.text(f"Caption: {safe_caption}")
                        if hasattr(img, 'score'):
                            st.text(f"Score: {img.score:.3f}")

                else:
                    st.error(
                        f"🖼️ Image not found or invalid: {Path(image_path).name}"
                    )
                    logger.warning(f"Image validation failed: {image_path}")

            except Exception as e:
                st.error(f"🖼️ Error displaying image: {str(e)}")
                logger.error(f"Failed to display image {img.image_path if hasattr(img, 'image_path') else 'unknown'}: {e}")

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
                    logger.warning(
                        f"Security: Image path outside allowed directory: {image_path}"
                    )
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
            if img_path_resolved.suffix.lower() not in (
                '.png',
                '.jpg',
                '.jpeg',
                '.gif',
                '.bmp',
                '.webp',
            ):
                logger.warning(f"Invalid image extension: {img_path_resolved.suffix}")
                return False

            logger.debug(f"Image validation passed: {img_path_resolved}")
            return True

        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False
