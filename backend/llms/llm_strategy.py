from abc import ABC, abstractmethod
from typing import Iterator, List, Optional


class LLMStrategy(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate_content(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Generate content synchronously."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM is available."""

    @abstractmethod
    def generate_content_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """Generate content with streaming (optional)."""
        raise NotImplementedError("Streaming not supported by this provider")
