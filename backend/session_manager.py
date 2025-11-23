"""
Session state management for the RAG chatbot application.
Implements Singleton pattern for centralized state management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import streamlit as st

from backend.document_processor import DocumentProcessor, ProcessingResult
from backend.prompts.prompt_manager import PromptManager
from config.constants import PAGE_CHAT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionManager:
    """Singleton class to manage Streamlit session state."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialize_defaults()
            self._initialized = True
            logger.info("SessionManager initialized with default values")

    def _initialize_defaults(self):
        """Set default values for all session state variables."""
        defaults = {
            # ============================================================
            # PAGE NAVIGATION
            # ============================================================
            'current_page': PAGE_CHAT,  # Current page: 'chat' or 'upload'
            # ============================================================
            # LANGUAGE & EMBEDDING CONFIGURATION
            # ============================================================
            'language': None,  # 'en', 'vi', or None
            # Embedding Strategy
            'embedding_strategy': None,  # EmbeddingStrategy instance
            'embedding_provider': None,  # "openai" or "local"
            'embedding_model_name': None,  # Model identifier
            'embedding_api_key': None,  # API key for online providers
            'embedding_dimension': None,  # Vector dimension
            # ============================================================
            # LLM CONFIGURATION
            # ============================================================
            'llm_type': None,  # 'online_llm' or 'local_llm'
            'llm_name': None,  # Model name
            'online_llms': None,  # Online LLM instance
            'llm_api_key': None,  # API key for online LLM
            'llm_choice': None,  # 'Online' or 'Local (Ollama)'
            'local_llms': None,  # Local LLM instance
            'model_version': None,  # Specific model version
            'selected_model_display': None,  # Display name
            'llm_model': None,  # OpenAILLM instance
            'llm_model_name': None,  # Model name (e.g., "gpt-4o-mini")
            'system_prompt': None,  # System prompt
            'temperature': 0.7,  # LLM temperature
            'chat_mode': 'rag',  # 'rag' or 'llm_only'
            # ============================================================
            # VECTOR DATABASE (QDRANT)
            # ============================================================
            'qdrant_manager': None,  # QdrantManager instance
            'collection_name': None,  # Collection name
            # ============================================================
            # DATA MANAGEMENT
            # ============================================================
            'chunks_df': pd.DataFrame(),  # DataFrame with chunks
            'doc_ids': [],  # List of document IDs
            'data_saved_success': False,  # Data saved to vector DB
            'source_data': 'UPLOAD',  # 'UPLOAD' or 'DB'
            # ============================================================
            # CHUNKING CONFIGURATION
            # ============================================================
            'chunk_size': 200,  # Chunk size in tokens
            'chunk_overlap': 20,  # Overlap between chunks
            'chunkOption': None,  # Chunking strategy
            'semantic_embedding_option': 'TF-IDF',
            # ============================================================
            # PDF PROCESSING CONFIGURATION
            # ============================================================
            'pdf_processing_strategy': 'auto',  # auto, ocr, fast, fallback
            'pdf_semantic_chunking': True,  # Use semantic chunking for PDFs
            'pdf_chunk_size': 1000,  # Chunk size for PDF semantic chunking
            'pdf_chunk_overlap': 100,  # Overlap for PDF chunks
            'pdf_combine_small_chunks': 1000,  # Threshold to combine small chunks
            'pdf_processing_progress': {},  # Progress tracking for PDF processing
            'uploaded_files_info': [],  # Information about uploaded files
            'document_processor': None,  # DocumentProcessor instance (lazy initialization)
            'pdf_processing_stats': {},  # PDF processing statistics
            'pdf_error_states': {},  # Error states for recovery
            # ============================================================
            # SEARCH & RETRIEVAL
            # ============================================================
            'search_option': 'Vector Search',
            'number_docs_retrieval': 3,
            'score_threshold': 0.5,  # Minimum similarity score
            # ============================================================
            # CHAT HISTORY
            # ============================================================
            'chat_history': [],
            # ============================================================
            # UI STATE
            # ============================================================
            'open_dialog': None,
            'show_settings': False,  # Show advanced settings
            # ============================================================
            # PROMPT MANAGEMENT
            # ============================================================
            'prompt_manager': None,  # PromptManager instance
            'active_system_template': 'system_helpful_assistant',
            'active_rag_template': 'rag_qa_with_history',
            'active_chat_template': 'chat_conversational',
            'custom_prompts_enabled': False,
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                logger.debug("Initialized '%s' with default value", key)

    # ============================================================
    # BASIC GETTERS & SETTERS
    # ============================================================

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session state."""
        value = st.session_state.get(key, default)
        logger.debug("Getting '%s' = %s", key, value)
        return value

    def set(self, key: str, value: Any) -> None:
        """Set value in session state."""
        logger.debug("Setting '%s' = %s", key, value)
        st.session_state[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple session state values."""
        logger.debug("Updating %s values", len(updates))
        for key, value in updates.items():
            st.session_state[key] = value
            logger.debug("Updated '%s' = %s", key, value)

    def delete(self, key: str) -> None:
        """Delete a key from session state."""
        if key in st.session_state:
            del st.session_state[key]
            logger.debug("Deleted '%s' from session state", key)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in st.session_state

    def clear(self) -> None:
        """Clear all session state and reinitialize."""
        logger.info("Clearing all session state")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_defaults()
        logger.info("Session state cleared and reinitialized")

    # ============================================================
    # SPECIFIC CLEAR METHODS
    # ============================================================

    def clear_chat_history(self) -> None:
        """Clear only chat history."""
        logger.info("Clearing chat history")
        st.session_state['chat_history'] = []

    def clear_data(self) -> None:
        """Clear data-related state."""
        logger.info("Clearing data-related state")
        self.update(
            {
                'chunks_df': pd.DataFrame(),
                'doc_ids': [],
                'collection': None,
                'collection_name': None,
                'data_saved_success': False,
            }
        )

    # ============================================================
    # PROMPT MANAGEMENT METHODS
    # ============================================================

    def initialize_prompt_manager(self) -> None:
        """Initialize prompt manager if not already initialized."""
        if self.get('prompt_manager') is None:
            try:
                prompt_manager = PromptManager()
                self.set('prompt_manager', prompt_manager)
                logger.info("PromptManager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PromptManager: {e}")

    def get_prompt_template(self, name: str):
        """
        Get a prompt template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate instance or None
        """
        prompt_manager = self.get('prompt_manager')
        if prompt_manager is None:
            self.initialize_prompt_manager()
            prompt_manager = self.get('prompt_manager')

        if prompt_manager:
            return prompt_manager.get_template(name)
        return None

    def get_active_system_prompt(self) -> str:
        """Get the rendered active system prompt."""
        template_name = self.get('active_system_template', 'system_helpful_assistant')
        template = self.get_prompt_template(template_name)

        if template:
            try:
                return template.render()
            except Exception as e:
                logger.error(f"Error rendering system prompt: {e}")
        return ""

    def get_active_rag_template(self):
        """Get the active RAG prompt template."""
        template_name = self.get('active_rag_template', 'rag_qa_with_history')
        return self.get_prompt_template(template_name)

    def get_active_chat_template(self):
        """Get the active chat prompt template."""
        template_name = self.get('active_chat_template', 'chat_conversational')
        return self.get_prompt_template(template_name)

    # ============================================================
    # PDF PROCESSING STATE MANAGEMENT
    # ============================================================

    def initialize_document_processor(self) -> Optional['DocumentProcessor']:
        """
        Initialize DocumentProcessor with current configuration.

        Returns:
            DocumentProcessor instance or None if initialization fails
        """
        if self.get('document_processor') is None:
            try:
                # Get current PDF configuration
                config = self.get_pdf_config()

                # Create processor instance
                processor = DocumentProcessor(config=config)
                self.set('document_processor', processor)

                logger.info("DocumentProcessor initialized successfully")
                return processor

            except Exception as e:
                logger.error(f"Failed to initialize DocumentProcessor: {e}")
                self.set_pdf_error('processor_init', str(e))
                return None

        return self.get('document_processor')

    def get_pdf_config(self) -> Dict[str, Any]:
        """
        Get current PDF processing configuration.

        Returns:
            Dictionary containing PDF processing configuration
        """
        return {
            "pdf": {
                "strategy": self.get("pdf_processing_strategy", "auto"),
                "infer_table_structure": True,
                "extract_images": True,  # Enable image extraction by default
                "chunk_after_extraction": False,  # Handle chunking separately
            },
            "ocr": {
                "languages": [self.get("language", "en")],
                "enabled": True,  # OCR is always available for strategies that need it
            },
            "chunking": {
                "chunk_size": self.get("pdf_chunk_size", 1000),
                "chunk_overlap": self.get("pdf_chunk_overlap", 100),
                "max_characters": self.get("pdf_chunk_size", 1000) * 2,
                "combine_text_under_n_chars": self.get(
                    "pdf_combine_small_chunks", 1000
                ),
                "new_after_n_chars": 3000,
                "multipage_sections": True,
                "enforce_strict": False,
            },
        }

    def set_pdf_config(self, config: Dict[str, Any]) -> None:
        """
        Set PDF processing configuration.

        Args:
            config: Configuration dictionary
        """
        # Update relevant session state variables
        if "pdf" in config:
            pdf_config = config["pdf"]
            if "strategy" in pdf_config:
                self.set("pdf_processing_strategy", pdf_config["strategy"])

        # OCR configuration is now handled automatically by strategies
        # No longer need to store OCR enabled state in session

        if "chunking" in config:
            chunking_config = config["chunking"]
            self.set("pdf_chunk_size", chunking_config.get("chunk_size", 1000))
            self.set("pdf_chunk_overlap", chunking_config.get("chunk_overlap", 100))
            self.set(
                "pdf_combine_small_chunks",
                chunking_config.get("combine_text_under_n_chars", 1000),
            )

        # Reset document processor to apply new configuration
        self.set('document_processor', None)
        logger.info(
            "PDF configuration updated, DocumentProcessor will be reinitialized"
        )

    def update_processing_progress(
        self, file_name: str, progress: float, status: str = "processing"
    ) -> None:
        """
        Update processing progress for a file.

        Args:
            file_name: Name of the file being processed
            progress: Progress percentage (0-100)
            status: Current status (processing, completed, error)
        """
        current_progress = self.get('pdf_processing_progress', {})
        current_progress[file_name] = {
            'progress': progress,
            'status': status,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        self.set('pdf_processing_progress', current_progress)
        logger.debug(f"Updated progress for {file_name}: {progress}% ({status})")

    def get_file_processing_progress(self, file_name: str) -> Dict[str, Any]:
        """
        Get processing progress for a specific file.

        Args:
            file_name: Name of the file

        Returns:
            Dictionary containing progress information
        """
        progress = self.get('pdf_processing_progress', {})
        return progress.get(file_name, {'progress': 0, 'status': 'not_started'})

    def update_processing_statistics(
        self, file_name: str, stats: Dict[str, Any]
    ) -> None:
        """
        Update processing statistics for a file.

        Args:
            file_name: Name of the processed file
            stats: Processing statistics dictionary
        """
        current_stats = self.get('pdf_processing_stats', {})
        current_stats[file_name] = {
            **stats,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        self.set('pdf_processing_stats', current_stats)
        logger.debug(f"Updated statistics for {file_name}: {stats}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get overall processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        stats = self.get('pdf_processing_stats', {})

        # Calculate aggregates
        total_files = len(stats)
        successful_files = sum(1 for s in stats.values() if s.get('success', False))
        failed_files = total_files - successful_files
        total_chunks = sum(s.get('chunks_created', 0) for s in stats.values())

        # Get document processor stats if available
        processor = self.get('document_processor')
        processor_stats = processor.get_processing_stats() if processor else {}

        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': (successful_files / max(1, total_files)) * 100,
            'total_chunks': total_chunks,
            'file_details': stats,
            'processor_stats': processor_stats,
        }

    def set_pdf_error(self, error_key: str, error_message: str) -> None:
        """
        Set PDF processing error state.

        Args:
            error_key: Key identifying the error
            error_message: Error message
        """
        error_states = self.get('pdf_error_states', {})
        error_states[error_key] = {
            'message': error_message,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        self.set('pdf_error_states', error_states)
        logger.error(f"PDF processing error ({error_key}): {error_message}")

    def get_pdf_errors(self) -> Dict[str, Any]:
        """
        Get PDF processing error states.

        Returns:
            Dictionary containing error states
        """
        return self.get('pdf_error_states', {})

    def clear_pdf_errors(self) -> None:
        """Clear PDF processing error states."""
        self.set('pdf_error_states', {})
        logger.info("PDF processing errors cleared")

    def reset_pdf_processing_state(self) -> None:
        """Reset PDF processing state while preserving configuration."""
        self.update(
            {
                'pdf_processing_progress': {},
                'pdf_processing_stats': {},
                'pdf_error_states': {},
                'uploaded_files_info': [],
            }
        )

        # Reset document processor to force reinitialization
        self.set('document_processor', None)

        logger.info("PDF processing state reset")

    def is_pdf_processor_available(self) -> bool:
        """
        Check if PDF processor is available and initialized.

        Returns:
            True if PDF processor is available
        """
        processor = self.get('document_processor')
        if processor is None:
            processor = self.initialize_document_processor()

        return processor is not None

    def get_pdf_processor_info(self) -> Dict[str, Any]:
        """
        Get information about the PDF processor.

        Returns:
            Dictionary containing processor information
        """
        if not self.is_pdf_processor_available():
            return {'available': False, 'error': 'Processor not initialized'}

        processor = self.get('document_processor')
        try:
            return {
                'available': True,
                'config_info': processor.get_config_info(),
                'stats': processor.get_processing_stats(),
                'supported_files': processor.get_supported_file_types(),
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def process_document_with_session(
        self,
        file_path: Union[str, Path],
        file_content: Optional[bytes] = None,
        original_filename: Optional[str] = None,
        **kwargs,
    ) -> Optional['ProcessingResult']:  # ProcessingResult from document_processor
        """
        Process a document using session-managed processor.

        Args:
            file_path: Path to the document
            file_content: Optional file content bytes
            original_filename: Original filename to override metadata
            **kwargs: Additional processing parameters

        Returns:
            ProcessingResult or None if processing fails
        """
        try:
            # Initialize processor if needed
            processor = self.initialize_document_processor()
            if processor is None:
                raise Exception("Failed to initialize document processor")

            # Update progress
            file_name = str(file_path).split('/')[-1].split('\\')[-1]
            self.update_processing_progress(file_name, 0, "starting")

            # Process document
            self.update_processing_progress(file_name, 50, "processing")
            result = processor.process_document(file_path, original_filename=original_filename, **kwargs)

            # Update statistics based on result
            if result.success:
                self.update_processing_progress(file_name, 100, "completed")
                self.update_processing_statistics(
                    file_name,
                    {
                        'success': True,
                        'chunks_created': len(result.elements),
                        'strategy_used': result.metadata.get(
                            'strategy_used', 'unknown'
                        ),
                        'ocr_used': result.metadata.get('ocr_used', False),
                        'total_pages': result.metadata.get('total_pages', 0),
                    },
                )
            else:
                self.update_processing_progress(file_name, 100, "error")
                error_message = result.error_message or "Unknown processing error"
                self.set_pdf_error(f"processing_{file_name}", error_message)
                self.update_processing_statistics(
                    file_name, {'success': False, 'error_message': error_message}
                )

            return result

        except Exception as e:
            file_name = str(file_path).split('/')[-1].split('\\')[-1]
            self.update_processing_progress(file_name, 100, "error")
            self.set_pdf_error(f"processing_{file_name}", str(e))
            logger.error(f"Document processing failed for {file_name}: {e}")
            return None

    # ============================================================
    # VALIDATION & CHECK METHODS
    # ============================================================

    def is_embedding_configured(self) -> bool:
        """Check if embedding strategy is configured."""
        embedding_strategy = self.get('embedding_strategy')
        is_configured = embedding_strategy is not None
        logger.debug("Embedding configured: %s", is_configured)
        return is_configured

    def is_llm_configured(self) -> bool:
        """Check if LLM is configured."""
        llm_model = self.get('llm_model')
        is_configured = llm_model is not None
        # online_llms = self.get('online_llms')
        # local_llms = self.get('local_llms')
        # is_configured = online_llms is not None or local_llms is not None
        logger.debug("LLM configured: %s", is_configured)
        return is_configured

    def is_qdrant_connected(self) -> bool:
        """Check if Qdrant is connected."""
        qdrant_manager = self.get('qdrant_manager')
        is_connected = qdrant_manager is not None
        logger.debug("Qdrant connected: %s", is_connected)
        return is_connected

    def has_documents(self) -> bool:
        """Check if there are documents in Qdrant."""
        if not self.is_qdrant_connected():
            return False

        qdrant_manager = self.get('qdrant_manager')
        try:
            stats = qdrant_manager.get_statistics()
            doc_count = stats.get('total_documents', 0)
            return doc_count > 0
        except Exception:
            return False

    def is_ready_for_chat(self) -> bool:
        """Check if system is ready for chat (with or without RAG)."""
        # Minimum: Need LLM configured
        llm_ok = self.is_llm_configured()
        embedding_ok = self.is_embedding_configured()

        # Can chat without documents (LLM only mode)
        is_ready = llm_ok and embedding_ok

        logger.debug(
            "Ready for chat: %s (llm:%s, embedding:%s)",
            is_ready,
            llm_ok,
            embedding_ok,
        )
        return is_ready

    # ============================================================
    # CONFIGURATION SUMMARY
    # ============================================================

    def get_config_summary(self) -> Dict[str, str]:
        """Get configuration summary for display."""
        qdrant_manager = self.get('qdrant_manager')

        if qdrant_manager:
            try:
                stats = qdrant_manager.get_statistics()
                doc_count = stats.get('total_documents', 0)
                collection_name = stats.get('collection_name', 'N/A')
            except Exception:
                doc_count = 0
                collection_name = 'N/A'
        else:
            doc_count = 0
            collection_name = 'Not connected'

        # check llm_model
        llm_model = self.get('llm_model')
        llm_name = llm_model.get_model_name() if llm_model else 'Not configured'

        embedding_strategy = self.get('embedding_strategy')
        embedding_name = (
            embedding_strategy.get_model_name()
            if embedding_strategy
            else 'Not configured'
        )

        return {
            'llm_model': llm_name,
            'embedding_model': embedding_name,
            'collection': collection_name,
            'total_docs': str(doc_count),
            'num_retrieval': str(self.get('number_docs_retrieval', 3)),
            'temperature': str(self.get('temperature', 0.7)),
            'score_threshold': str(self.get('score_threshold', 0.5)),
        }

    def get_status_summary(self) -> Dict[str, bool]:
        """Get boolean status of major components."""
        return {
            'Embedding': self.is_embedding_configured(),
            'LLM': self.is_llm_configured(),
            'Qdrant': self.is_qdrant_connected(),
            'Documents': self.has_documents(),
            'Ready': self.is_ready_for_chat(),
        }


def get_session_manager() -> SessionManager:
    """Convenience function to get SessionManager instance."""
    return SessionManager()
