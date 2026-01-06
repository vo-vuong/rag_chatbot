import logging
from typing import Iterator, List, Optional

from openai import OpenAI, OpenAIError

from backend.llms.online_llm import OnlineLLMStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import PromptBuilder for prompt construction
try:
    from backend.prompts.prompt_builder import PromptBuilder

    PROMPT_BUILDER_AVAILABLE = True
except ImportError:
    PROMPT_BUILDER_AVAILABLE = False
    logger.warning("PromptBuilder not available, using legacy prompt construction")


class OpenAILLM(OnlineLLMStrategy):
    """OpenAI implementation of Online LLM strategy."""

    def __init__(
        self,
        api_key: str,
        model_version: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ):
        """Initialize OpenAI LLM."""
        super().__init__(
            provider_name="openai",
            api_key=api_key,
            model_version=model_version,
        )

        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            self._client = OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise ConnectionError(f"Failed to initialize OpenAI client: {str(e)}")

    def generate_content(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Generate response using OpenAI Chat Completion API."""
        try:
            messages = self._build_messages(prompt, context, chat_history)

            response = self._client.chat.completions.create(
                model=self.model_version,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def generate_content_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """Generate streaming response."""
        try:
            messages = self._build_messages(prompt, context, chat_history)

            stream = self._client.chat.completions.create(
                model=self.model_version,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def _build_messages(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Build messages array for Chat Completion API."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        # Build user message with context if available
        if context:
            if PROMPT_BUILDER_AVAILABLE:
                # Use PromptBuilder for better prompt construction
                user_message = PromptBuilder.build_rag_prompt(
                    query=prompt, context=context
                )
            else:
                # Legacy format
                user_message = f"""Based on the following context:
                    {context}
                    Question: {prompt}"""
        else:
            user_message = prompt

        messages.append({"role": "user", "content": user_message})

        return messages

    def set_temperature(self, temperature: float) -> None:
        """Update temperature setting."""
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")

        self.temperature = temperature
        logger.info(f"Temperature updated to {temperature}")

    def set_system_prompt(self, system_prompt: str) -> None:
        """Update system prompt setting."""
        self.system_prompt = system_prompt
        logger.info(f"System prompt updated to {system_prompt}")

    def set_max_tokens(self, max_tokens: int) -> None:
        """Update max tokens setting."""
        if not 0 <= max_tokens <= 4096:
            raise ValueError("Max tokens must be between 0 and 4096")

        self.max_tokens = max_tokens
        logger.info(f"Max tokens updated to {max_tokens}")
