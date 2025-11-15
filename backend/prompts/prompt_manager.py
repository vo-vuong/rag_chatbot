"""
Prompt Manager - Singleton class for centralized prompt management.

All prompts are loaded from config/prompts.yaml.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from backend.prompts.prompt_template import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptManager:
    """
    Singleton manager for prompt templates.

    Loads templates from YAML configuration and provides centralized access.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.templates: Dict[str, PromptTemplate] = {}
            self.config_path: Optional[str] = None
            self.custom_prompts_dir: Optional[str] = None
            self._initialized = True

            # Load prompts from YAML (required)
            self._load_yaml_config()

            logger.info(
                f"PromptManager initialized with {len(self.templates)} templates"
            )

    def _load_yaml_config(self) -> None:
        """
        Load prompts from YAML configuration (required).

        Raises:
            FileNotFoundError: If prompts.yaml is not found
            yaml.YAMLError: If YAML parsing fails
        """
        # Look for prompts.yaml in config directory
        base_path = Path(__file__).parent.parent.parent
        yaml_path = base_path / "config" / "prompts.yaml"

        if not yaml_path.exists():
            error_msg = (
                f"❌ REQUIRED: prompts.yaml not found at {yaml_path}\n"
                "The prompt configuration file is required for the system to work.\n"
                "Please ensure config/prompts.yaml exists."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            self.load_templates(str(yaml_path))
            logger.info(f"✅ Loaded {len(self.templates)} prompts from {yaml_path}")
        except Exception as e:
            error_msg = f"❌ Failed to load prompts from {yaml_path}: {e}"
            logger.error(error_msg)
            raise

    def load_templates(self, config_path: str) -> None:
        """
        Load prompt templates from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config_path = config_path

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not config or 'prompts' not in config:
            logger.warning("No prompts found in YAML config")
            return

        # Load each prompt template
        for prompt_data in config['prompts']:
            try:
                template = PromptTemplate.from_dict(prompt_data)
                self.templates[template.name] = template
                logger.debug(f"Loaded template from YAML: {template.name}")
            except Exception as e:
                logger.error(f"Failed to load prompt from YAML: {e}")

        logger.info(f"Loaded {len(config['prompts'])} templates from {config_path}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate if found, None otherwise
        """
        template = self.templates.get(name)
        if template is None:
            logger.warning(f"Template '{name}' not found")
        return template

    def list_templates(self, category: Optional[str] = None) -> List[PromptTemplate]:
        """
        List all templates, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of PromptTemplate objects
        """
        if category:
            return [t for t in self.templates.values() if t.category == category]
        return list(self.templates.values())

    def get_categories(self) -> List[str]:
        """Get list of all unique categories."""
        categories = set(t.category for t in self.templates.values())
        return sorted(list(categories))

    def save_custom_template(
        self, template: PromptTemplate, custom_dir: Optional[str] = None
    ) -> bool:
        """
        Save a custom prompt template to JSON file.

        Args:
            template: PromptTemplate to save
            custom_dir: Directory for custom prompts

        Returns:
            True if successful, False otherwise
        """
        try:
            # Set custom prompts directory
            if custom_dir:
                self.custom_prompts_dir = custom_dir
            elif not self.custom_prompts_dir:
                base_path = Path(__file__).parent.parent.parent
                self.custom_prompts_dir = str(base_path / "config" / "custom_prompts")

            # Create directory if it doesn't exist
            os.makedirs(self.custom_prompts_dir, exist_ok=True)

            # Save template to JSON
            file_path = os.path.join(self.custom_prompts_dir, f"{template.name}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)

            # Add to loaded templates
            self.templates[template.name] = template

            logger.info(f"Saved custom template: {template.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save custom template: {e}")
            return False

    def delete_template(self, name: str) -> bool:
        """
        Delete a template from memory and file system.

        Args:
            name: Template name

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from memory
            if name in self.templates:
                del self.templates[name]

            # Try to delete file
            if self.custom_prompts_dir:
                file_path = os.path.join(self.custom_prompts_dir, f"{name}.json")
                if os.path.exists(file_path):
                    os.remove(file_path)

            logger.info(f"Deleted template: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete template: {e}")
            return False

    def export_templates(
        self, output_path: str, category: Optional[str] = None
    ) -> bool:
        """
        Export templates to YAML file.

        Args:
            output_path: Path for output file
            category: Optional category filter

        Returns:
            True if successful, False otherwise
        """
        try:
            templates_to_export = self.list_templates(category)

            export_data = {
                'version': '1.0',
                'prompts': [t.to_dict() for t in templates_to_export],
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, allow_unicode=True, default_flow_style=False)

            logger.info(
                f"Exported {len(templates_to_export)} templates to {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to export templates: {e}")
            return False

    def import_templates(self, input_path: str) -> bool:
        """
        Import templates from YAML or JSON file.

        Args:
            input_path: Path to import file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                if input_path.endswith('.json'):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)

            # Handle both single template and multiple templates
            if 'prompts' in data:
                templates_data = data['prompts']
            elif 'name' in data:
                templates_data = [data]
            else:
                raise ValueError("Invalid file format")

            count = 0
            for template_data in templates_data:
                template = PromptTemplate.from_dict(template_data)
                self.templates[template.name] = template
                count += 1

            logger.info(f"Imported {count} templates from {input_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import templates: {e}")
            return False

    def __len__(self) -> int:
        """Return number of loaded templates."""
        return len(self.templates)

    def __repr__(self) -> str:
        return f"PromptManager({len(self.templates)} templates, {len(self.get_categories())} categories)"
