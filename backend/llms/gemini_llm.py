import logging
from typing import List, Optional

import google.generativeai as genai

from backend.llms.online_llm import OnlineLLMStrategy

# Import PromptBuilder for prompt construction
try:
    from backend.prompts.prompt_builder import PromptBuilder

    PROMPT_BUILDER_AVAILABLE = True
except ImportError:
    PROMPT_BUILDER_AVAILABLE = False

logger = logging.getLogger(__name__)


class GeminiLLM(OnlineLLMStrategy):
    """Google Gemini implementation of Online LLM strategy."""

    def __init__(
        self,
        api_key: str,
        model_version: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ):
        """Initialize Gemini LLM."""
        super().__init__(
            provider_name="gemini",
            api_key=api_key,
            model_version=model_version,
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

        self._initialize_client()

    def _initialize_client(self):
        """Initialize Gemini client."""
        try:
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                self.model_version,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Gemini client: {str(e)}")

    def generate_content(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Generate response using Gemini API."""
        try:
            full_prompt = self._build_messages(prompt, context, chat_history)
            response = self._client.generate_content(full_prompt)
            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def _build_messages(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Build prompt with context and history."""
        if context and PROMPT_BUILDER_AVAILABLE:
            # Use PromptBuilder for RAG prompts
            return PromptBuilder.build_rag_prompt(query=prompt, context=context)

        # Legacy format
        parts = []

        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")

        if context:
            parts.append(f"Context:\n{context}")

        parts.append(f"Question: {prompt}")

        return "\n\n".join(parts)
