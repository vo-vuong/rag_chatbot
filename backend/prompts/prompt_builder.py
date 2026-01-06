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
                )

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
            )

    @staticmethod
    def build_chat_prompt(
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        template: Optional[PromptTemplate] = None,
        **kwargs,
    ) -> str:
        """
        Build a chat prompt with history.

        Uses templates from config/prompts.yaml only.
        If no template provided, loads 'chat_conversational' from YAML.

        Args:
            query: User's current question
            chat_history: List of previous messages
            template: Optional PromptTemplate (loads from YAML if None)
            **kwargs: Additional variables (e.g., context for RAG chat)

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If template not found and PromptManager unavailable
        """
        # Format chat history
        history_str = (
            PromptBuilder.format_chat_history(chat_history) if chat_history else ""
        )

        # Get template from YAML if not provided
        if template is None:
            try:
                manager = PromptManager()

                # Choose template based on whether context is provided
                if 'context' in kwargs:
                    template = manager.get_template('chat_with_context')
                else:
                    template = manager.get_template('chat_conversational')

                if template is None:
                    raise ValueError(
                        "Chat template not found in prompts.yaml. "
                        "Please ensure 'chat_conversational' template exists."
                    )

            except Exception as e:
                logger.error(f"Failed to load chat template from YAML: {e}")
                raise ValueError(
                    f"Cannot build chat prompt: {e}. "
                    "Ensure config/prompts.yaml exists and contains chat templates."
                )

        # Build variables for template
        variables = {'query': query, 'chat_history': history_str, **kwargs}

        # Render template
        try:
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering chat template: {e}")
            raise ValueError(
                f"Failed to render chat template '{template.name}': {e}. "
                f"Required variables: {template.variables}"
            )

    @staticmethod
    def build_system_prompt(template: Optional[PromptTemplate] = None, **kwargs) -> str:
        """
        Build a system prompt.

        Uses templates from config/prompts.yaml only.
        If no template provided, loads 'system_helpful_assistant' from YAML.

        Args:
            template: PromptTemplate for system message (loads from YAML if None)
            **kwargs: Variables for template

        Returns:
            Rendered system prompt

        Raises:
            ValueError: If template not found and PromptManager unavailable
        """
        # Get template from YAML if not provided
        if template is None:
            try:
                manager = PromptManager()
                template = manager.get_template('system_helpful_assistant')

                if template is None:
                    raise ValueError(
                        "System template not found in prompts.yaml. "
                        "Please ensure 'system_helpful_assistant' template exists."
                    )

            except Exception as e:
                logger.error(f"Failed to load system template from YAML: {e}")
                raise ValueError(
                    f"Cannot build system prompt: {e}. "
                    "Ensure config/prompts.yaml exists and contains system templates."
                )

        # Render template
        try:
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering system template: {e}")
            raise ValueError(
                f"Failed to render system template '{template.name}': {e}. "
                f"Required variables: {template.variables}"
            )

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
        page_number: Optional[int] = None,
        source_document: Optional[str] = None,
        score: Optional[float] = None,
        template: Optional[PromptTemplate] = None,
    ) -> str:
        """
        Format image context using the rag_image_context template.

        Args:
            image_caption: Caption/description of the image
            page_number: Page number (unused in simple template)
            source_document: Source document name (unused in simple template)
            score: Optional similarity score (unused in simple template)
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
                    logger.warning("rag_image_context template not found, using fallback")
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
                return PromptBuilder.build_rag_prompt(query=query, context=combined_context)

        # Build variables for template
        variables = {
            'query': query,
            'text_context': text_context if text_context else "No text context available.",
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
            )

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

    @staticmethod
    def build_error_prompt(
        query: str,
        error_type: str = "no_results",
        template: Optional[PromptTemplate] = None,
        **kwargs,
    ) -> str:
        """
        Build an error/fallback prompt.

        Uses templates from config/prompts.yaml only.
        If no template provided, loads appropriate error template from YAML.

        Args:
            query: User's question
            error_type: Type of error (no_results, retrieval_failed, llm_failed)
            template: Optional PromptTemplate (loads from YAML if None)
            **kwargs: Additional variables (e.g., error_message, context)

        Returns:
            Rendered error message

        Raises:
            ValueError: If template not found and PromptManager unavailable
        """
        # Get template from YAML if not provided
        if template is None:
            try:
                manager = PromptManager()

                # Map error type to template name
                template_map = {
                    'no_results': 'error_no_results',
                    'retrieval_failed': 'error_retrieval_failed',
                    'llm_failed': 'error_llm_failed',
                }

                template_name = template_map.get(error_type, 'error_no_results')
                template = manager.get_template(template_name)

                if template is None:
                    raise ValueError(
                        f"Error template '{template_name}' not found in prompts.yaml. "
                        "Please ensure error templates exist."
                    )

            except Exception as e:
                logger.error(f"Failed to load error template from YAML: {e}")
                raise ValueError(
                    f"Cannot build error prompt: {e}. "
                    "Ensure config/prompts.yaml exists and contains error templates."
                )

        # Build variables for template
        variables = {'query': query, 'error_type': error_type, **kwargs}

        # Render template
        try:
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering error template: {e}")
            raise ValueError(
                f"Failed to render error template '{template.name}': {e}. "
                f"Required variables: {template.variables}"
            )

    @staticmethod
    def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
        """
        Truncate text to maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix
