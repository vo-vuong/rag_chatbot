"""
Agent configuration settings.
"""

import os
from dataclasses import dataclass
from enum import Enum

from config.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_NUM_RETRIEVAL,
    DEFAULT_SCORE_THRESHOLD,
)


class WorkflowType(str, Enum):
    """Available workflow types."""

    AGENTIC_RAG = "agentic_rag"
    SIMPLE_RAG = "simple_rag"  # Future: linear retrieve-then-generate


@dataclass
class AgentConfig:
    """Configuration for agent workflows."""

    # LLM settings
    model: str = DEFAULT_LLM_MODEL
    temperature: float = 0.2
    max_tokens: int = 2000

    # Agent behavior
    max_iterations: int = 5  # Max tool-calling loops

    # Retrieval defaults
    default_top_k: int = DEFAULT_NUM_RETRIEVAL
    default_score_threshold: float = DEFAULT_SCORE_THRESHOLD

    # Persistence
    enable_checkpointing: bool = True
    checkpointer_type: str = "memory"  # "memory", "sqlite", "redis"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        # Read AGENT_MEMORY_ENABLED (default: True for production)
        memory_enabled_str = os.getenv("AGENT_MEMORY_ENABLED", "true").lower()
        enable_checkpointing = memory_enabled_str in ("true", "1", "yes")

        return cls(
            model=DEFAULT_LLM_MODEL,
            default_top_k=DEFAULT_NUM_RETRIEVAL,
            default_score_threshold=DEFAULT_SCORE_THRESHOLD,
            enable_checkpointing=enable_checkpointing,
        )
