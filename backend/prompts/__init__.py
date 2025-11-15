"""
Prompt Management Module for RAG Chatbot.

This module provides centralized prompt template management, including:
- PromptTemplate: Template class with variable substitution
- PromptManager: Singleton manager for loading and managing templates
- PromptBuilder: Helper for constructing complex prompts
"""

from backend.prompts.prompt_builder import PromptBuilder
from backend.prompts.prompt_manager import PromptManager
from backend.prompts.prompt_template import PromptTemplate

__all__ = ['PromptTemplate', 'PromptManager', 'PromptBuilder']
