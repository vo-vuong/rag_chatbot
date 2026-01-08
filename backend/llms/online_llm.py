from abc import abstractmethod
from typing import Iterator, List, Optional

from backend.llms.llm_strategy import LLMStrategy


class OnlineLLMStrategy(LLMStrategy):
    """Base class for cloud-based LLM providers."""

    def __init__(self, provider_name: str, api_key: str, model_version: str):
        """
        Initialize online LLM adapter.

        Args:
            provider_name: Name of provider ('openai')
            api_key: API key for authentication
            model_version: Model version to use
        """
        if not api_key:
            raise ValueError(f"API key is required for {provider_name}")

        self.provider_name = provider_name
        self.api_key = api_key
        self.model_version = model_version
        self._client = None

    @abstractmethod
    def _initialize_client(self):
        """Initialize provider-specific client. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def generate_content(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Generate response using online LLM API."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key and self._client)

    @abstractmethod
    def generate_content_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """Generate response using online LLM API."""
        raise NotImplementedError
