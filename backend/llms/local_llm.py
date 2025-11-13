import json
from typing import Iterator, List, Optional

import requests

from backend.llms.llm_strategy import LLMStrategy


class LocalLLMStrategy(LLMStrategy):
    """Base class for local LLM providers (Ollama)."""

    def __init__(self, model_name: str, host: str, port: int):
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
        self.base_url = f"http://{host}:{port}"

    def generate_content(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Generate content using local Ollama model."""
        try:
            full_prompt = self._build_prompt(prompt, context, chat_history)

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()

            return response.json()["response"]

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")

    def generate_content_stream(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """Generate content with streaming."""
        try:
            full_prompt = self._build_prompt(prompt, context, chat_history)

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": True,
                },
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")

    def _build_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Build prompt with context and history."""
        parts = []

        # Add chat history if available
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role.capitalize()}: {content}")

        # Add context if available
        if context:
            parts.append(f"Context:\n{context}")

        # Add current prompt
        parts.append(f"Question: {prompt}")

        return "\n\n".join(parts)

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except Exception:
            return []
