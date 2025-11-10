from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    """Abstract adapter for LLM providers."""

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """
        Generate content from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if LLM is available and configured.

        Returns:
            True if available, False otherwise
        """


class OnlineLLMAdapter(LLMAdapter):
    """Adapter for online LLM providers (Gemini, OpenAI, etc.)."""

    def __init__(self, provider_name: str, api_key: str, model_version: str):
        """
        Initialize online LLM adapter.

        Args:
            provider_name: Name of provider ('Gemini', 'OpenAI')
            api_key: API key for authentication
            model_version: Model version to use
        """
        self.provider_name = provider_name
        self.api_key = api_key
        self.model_version = model_version
        self._client = None

    def generate_content(self, prompt: str) -> str:
        """
        Generate content using online API.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # TODO: Implement API calls to Gemini or OpenAI
        # 1. Initialize client if not already done
        # 2. Send prompt to API
        # 3. Parse and return response
        return "Generated response from online LLM"

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)


class LocalLLMAdapter(LLMAdapter):
    """Adapter for local LLM providers (Ollama)."""

    def __init__(self, model_name: str, host: str = "localhost", port: int = 11434):
        """
        Initialize local LLM adapter.

        Args:
            model_name: Name of local model
            host: Ollama host
            port: Ollama port
        """
        self.model_name = model_name
        self.host = host
        self.port = port
        self._client = None

    def generate_content(self, prompt: str) -> str:
        """
        Generate content using local model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # TODO: Implement Ollama API call
        # 1. Connect to Ollama server
        # 2. Send prompt with model name
        # 3. Return generated response
        return "Generated response from local LLM"

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        # TODO: Implement health check
        pass


class LLMFactory:
    """
    Factory for creating LLM adapters.
    Implements Factory pattern for LLM creation.
    """

    @staticmethod
    def create_online_llm(
        provider_name: str, api_key: str, model_version: str
    ) -> OnlineLLMAdapter:
        """
        Create online LLM adapter.

        Args:
            provider_name: Provider name ('Gemini', 'OpenAI')
            api_key: API key
            model_version: Model version

        Returns:
            Online LLM adapter instance
        """
        return OnlineLLMAdapter(provider_name, api_key, model_version)

    @staticmethod
    def create_local_llm(
        model_name: str, host: str = "localhost", port: int = 11434
    ) -> LocalLLMAdapter:
        """
        Create local LLM adapter.

        Args:
            model_name: Local model name
            host: Ollama host
            port: Ollama port

        Returns:
            Local LLM adapter instance
        """
        return LocalLLMAdapter(model_name, host, port)
