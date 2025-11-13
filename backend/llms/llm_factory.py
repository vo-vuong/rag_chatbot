from typing import List

from backend.llms.gemini_llm import GeminiLLM
from backend.llms.local_llm import LocalLLMStrategy
from backend.llms.online_llm import OnlineLLMStrategy
from backend.llms.openai_llm import OpenAILLM


class LLMFactory:
    """
    Factory for creating LLM adapters.
    Supports both Online and Local LLM strategies.
    """

    # Online provider registry
    ONLINE_PROVIDERS = {
        "openai": OpenAILLM,
        "gemini": GeminiLLM,
    }

    @staticmethod
    def create_online_llm(
        provider_name: str,
        api_key: str,
        model_version: str,
        **kwargs,
    ) -> OnlineLLMStrategy:
        """
        Create online LLM adapter.

        Args:
            provider_name: Provider name ('openai', 'gemini')
            api_key: API key
            model_version: Model version
            **kwargs: Additional provider-specific arguments

        Returns:
            Online LLM adapter instance

        Raises:
            ValueError: If provider is not supported
        """
        provider_name = provider_name.lower()

        if provider_name not in LLMFactory.ONLINE_PROVIDERS:
            raise ValueError(
                f"Unsupported online provider: {provider_name}. "
                f"Available: {list(LLMFactory.ONLINE_PROVIDERS.keys())}"
            )

        provider_class = LLMFactory.ONLINE_PROVIDERS[provider_name]
        return provider_class(
            api_key=api_key,
            model_version=model_version,
            **kwargs,
        )

    @staticmethod
    def create_local_llm(
        model_name: str,
        host: str = "localhost",
        port: int = 11434,
    ) -> LocalLLMStrategy:
        """
        Create local LLM adapter (Ollama).

        Args:
            model_name: Local model name
            host: Ollama host
            port: Ollama port

        Returns:
            Local LLM adapter instance
        """
        return LocalLLMStrategy(
            model_name=model_name,
            host=host,
            port=port,
        )

    @staticmethod
    def get_available_online_providers() -> List[str]:
        """Get list of supported online providers."""
        return list(LLMFactory.ONLINE_PROVIDERS.keys())
