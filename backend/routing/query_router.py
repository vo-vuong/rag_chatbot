"""
Query Router Module

LLM-based query classification for routing to text or image collections.
Uses gpt-4o-mini with structured output for reliable route decisions.
"""

import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class QueryClassification(BaseModel):
    """Classification result for query routing."""

    route: Literal["text_only", "image_only"] = Field(
        description="Target collection: text_only or image_only"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification"
    )


class QueryRouter:
    """
    LLM-based query router for text/image collection selection.

    Uses gpt-4o-mini with structured output to classify queries
    into exactly one route: text_only or image_only.
    """

    def __init__(self, openai_api_key: str):
        """
        Initialize router with OpenAI API key.

        Args:
            openai_api_key: Valid OpenAI API key
        """
        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key,
        )
        self._structured_llm = self._llm.with_structured_output(
            QueryClassification
        )
        self._prompt_manager = PromptManager()

    def _get_system_prompt(self) -> str:
        """Get system prompt from PromptManager."""
        template = self._prompt_manager.get_template("router_system_prompt")
        if template:
            return template.template
        raise ValueError("router_system_prompt not found in prompts.yaml")

    def classify(self, query: str) -> QueryClassification:
        """
        Classify query to determine target collection.

        Args:
            query: User's search query

        Returns:
            QueryClassification with route and reasoning
        """
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Query: {query}"}
        ]

        result = self._structured_llm.invoke(messages)

        logger.info(
            f"Query classified: '{query[:50]}...' -> {result.route}"
        )

        return result
