"""
Prompt Builder - Helper for constructing complex prompts from templates.

All prompts are loaded from config/prompts.yaml.
No legacy/hardcoded fallbacks for simplicity.
"""

import logging
from typing import Dict, List, Optional

from backend.prompts.prompt_manager import PromptManager
from backend.prompts.prompt_template import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Helper class for building complex prompts from templates.

    Provides static methods for common prompt construction patterns.
    """

    @staticmethod
    def build_rag_prompt(
        query: str, context: str, template: Optional[PromptTemplate] = None, **kwargs
    ) -> str:
        """
        Build a RAG prompt with query and context.

        Uses templates from config/prompts.yaml only.
        If no template provided, loads 'rag_qa_with_history' from YAML.

        Args:
            query: User's question
            context: Retrieved context from documents
            template: Optional PromptTemplate (loads from YAML if None)
            **kwargs: Additional variables for template

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If template not found and PromptManager unavailable
        """
        # Get template from YAML if not provided
        if template is None:
            try:
                manager = PromptManager()
                template = manager.get_template('rag_qa_with_history')

                if template is None:
                    raise ValueError(
                        "RAG template not found in prompts.yaml. "
                        "Please ensure 'rag_qa_with_history' template exists."
                    )

            except Exception as e:
                logger.error(f"Failed to load RAG template from YAML: {e}")
                raise ValueError(
                    f"Cannot build RAG prompt: {e}. "
                    "Ensure config/prompts.yaml exists and contains RAG templates."
                ) from e

        # Build variables for template
        variables = {'query': query, 'context': context, **kwargs}

        # Render template
        try:
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering RAG template: {e}")
            raise ValueError(
                f"Failed to render RAG template '{template.name}': {e}. "
                f"Required variables: {template.variables}"
            ) from e

    @staticmethod
    def format_context(
        chunks: List[str],
        scores: Optional[List[float]] = None,
        include_scores: bool = False,
        max_chunks: Optional[int] = None,
    ) -> str:
        """
        Format retrieved context chunks into a single string.

        Args:
            chunks: List of text chunks
            scores: Optional similarity scores
            include_scores: Whether to include scores in output
            max_chunks: Maximum number of chunks to include

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        # Limit chunks if specified
        if max_chunks:
            chunks = chunks[:max_chunks]
            if scores:
                scores = scores[:max_chunks]

        # Format chunks
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            if include_scores and scores and i < len(scores):
                formatted_chunks.append(
                    f"[Document {i+1}, Score: {scores[i]:.3f}]\n{chunk}"
                )
            else:
                formatted_chunks.append(f"[Document {i+1}]\n{chunk}")

        return "\n\n---\n\n".join(formatted_chunks)

    @staticmethod
    def format_image_context(
        image_caption: str,
        template: Optional[PromptTemplate] = None,
    ) -> str:
        """
        Format image context using the rag_image_context template.

        Args:
            image_caption: Caption/description of the image
            template: Optional PromptTemplate (loads 'rag_image_context' if None)

        Returns:
            Formatted image context string
        """
        if not image_caption:
            return ""

        # Get template from YAML if not provided
        if template is None:
            try:
                manager = PromptManager()
                template = manager.get_template('rag_image_context')

                if template is None:
                    # Fallback to simple format
                    logger.warning(
                        "rag_image_context template not found, using fallback"
                    )
                    return f"Relevant Image Description:\n{image_caption}"

            except Exception as e:
                logger.error(f"Failed to load image context template: {e}")
                return f"Relevant Image Description:\n{image_caption}"

        # Build variables for template
        variables = {'image_caption': image_caption}

        # Render template
        try:
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering image context template: {e}")
            return f"Relevant Image Description:\n{image_caption}"

    @staticmethod
    def build_rag_prompt_with_images(
        query: str,
        text_context: str,
        image_context: str = "",
        chat_history: Optional[List[Dict[str, str]]] = None,
        template: Optional[PromptTemplate] = None,
    ) -> str:
        """
        Build a RAG prompt with separate text and image contexts.

        Uses 'rag_qa_with_images' or 'rag_qa_with_images_no_history' templates.

        Args:
            query: User's question
            text_context: Retrieved text context from documents
            image_context: Formatted image context (from format_image_context)
            chat_history: Optional conversation history
            template: Optional PromptTemplate (auto-selects if None)

        Returns:
            Rendered prompt string
        """
        # Format chat history if provided
        history_str = ""
        if chat_history:
            history_str = PromptBuilder.format_chat_history(chat_history)

        # Get template from YAML if not provided
        if template is None:
            try:
                manager = PromptManager()

                # Choose template based on whether history is provided
                if chat_history:
                    template = manager.get_template('rag_qa_with_images')
                else:
                    template = manager.get_template('rag_qa_with_images_no_history')

                if template is None:
                    # Fallback to standard RAG template
                    logger.warning("Image RAG template not found, using standard RAG")
                    combined_context = text_context
                    if image_context:
                        combined_context += f"\n\n{image_context}"
                    return PromptBuilder.build_rag_prompt(
                        query=query,
                        context=combined_context,
                        chat_history=history_str if history_str else None,
                    )

            except Exception as e:
                logger.error(f"Failed to load image RAG template: {e}")
                combined_context = text_context
                if image_context:
                    combined_context += f"\n\n{image_context}"
                return PromptBuilder.build_rag_prompt(
                    query=query, context=combined_context
                )

        # Build variables for template
        variables = {
            'query': query,
            'text_context': (
                text_context if text_context else "No text context available."
            ),
            'image_context': image_context if image_context else "",
            'chat_history': history_str if history_str else "No previous conversation.",
        }

        # Render template
        try:
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering image RAG template: {e}")
            raise ValueError(
                f"Failed to render image RAG template '{template.name}': {e}. "
                f"Required variables: {template.variables}"
            ) from e

    @staticmethod
    def format_chat_history(
        history: List[Dict[str, str]],
        max_messages: int = 10,
        include_system: bool = False,
    ) -> str:
        """
        Format chat history into a string.

        Args:
            history: List of message dictionaries with 'role' and 'content'
            max_messages: Maximum number of messages to include
            include_system: Whether to include system messages

        Returns:
            Formatted chat history string
        """
        if not history:
            return ""

        # Filter and limit messages
        filtered_history = history
        if not include_system:
            filtered_history = [m for m in history if m.get('role') != 'system']

        # Take most recent messages
        if max_messages and len(filtered_history) > max_messages:
            filtered_history = filtered_history[-max_messages:]

        # Format messages
        formatted_messages = []
        for msg in filtered_history:
            role = msg.get('role', 'user').capitalize()
            content = msg.get('content', '')
            formatted_messages.append(f"{role}: {content}")

        return "\n\n".join(formatted_messages)
