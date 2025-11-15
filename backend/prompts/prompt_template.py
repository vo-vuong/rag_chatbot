"""
Prompt Template class with variable substitution and validation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    Represents a prompt template with variable substitution.

    Supports Python string formatting with {variable} syntax.
    """

    def __init__(
        self,
        name: str,
        template: str,
        description: str = "",
        category: str = "general",
        language: str = "multi",
        variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a prompt template.

        Args:
            name: Unique identifier for the template
            template: Template string with {variable} placeholders
            description: Human-readable description
            category: Category (system, rag, chat, error)
            language: Language code (en, vi, multi)
            variables: List of required variable names (auto-detected if None)
            metadata: Additional metadata (version, author, tags, etc.)
        """
        self.name = name
        self.template = template
        self.description = description
        self.category = category
        self.language = language
        self.metadata = metadata or {}

        # Auto-detect variables from template if not provided
        self.variables = (
            variables if variables is not None else self._extract_variables()
        )

        logger.debug(f"Created PromptTemplate: {name} with variables: {self.variables}")

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template string."""
        pattern = r'\{(\w+)\}'
        matches = re.findall(pattern, self.template)
        return list(set(matches))

    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Rendered template string

        Raises:
            KeyError: If required variables are missing
        """
        try:
            # Validate variables
            is_valid, missing = self.validate(kwargs)
            if not is_valid:
                raise KeyError(
                    f"Missing required variables for template '{self.name}': {missing}"
                )

            # Render template
            rendered = self.template.format(**kwargs)
            logger.debug(f"Rendered template '{self.name}'")
            return rendered

        except KeyError as e:
            logger.error(f"Error rendering template '{self.name}': {e}")
            raise

    def validate(self, variables: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that all required variables are provided.

        Args:
            variables: Dictionary of variable values

        Returns:
            Tuple of (is_valid, list_of_missing_variables)
        """
        provided_vars = set(variables.keys())
        required_vars = set(self.variables)
        missing_vars = required_vars - provided_vars

        is_valid = len(missing_vars) == 0
        return is_valid, list(missing_vars)

    def get_required_variables(self) -> List[str]:
        """Get list of required variable names."""
        return self.variables.copy()

    def clone(self, new_name: str) -> 'PromptTemplate':
        """
        Create a copy of this template with a new name.

        Args:
            new_name: Name for the cloned template

        Returns:
            New PromptTemplate instance
        """
        return PromptTemplate(
            name=new_name,
            template=self.template,
            description=f"{self.description} (cloned from {self.name})",
            category=self.category,
            language=self.language,
            variables=self.variables.copy(),
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            'name': self.name,
            'template': self.template,
            'description': self.description,
            'category': self.category,
            'language': self.language,
            'variables': self.variables,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create template from dictionary."""
        return cls(
            name=data['name'],
            template=data['template'],
            description=data.get('description', ''),
            category=data.get('category', 'general'),
            language=data.get('language', 'multi'),
            variables=data.get('variables'),
            metadata=data.get('metadata', {}),
        )

    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}', category='{self.category}', variables={self.variables})"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
