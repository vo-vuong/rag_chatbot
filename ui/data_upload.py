"""
Data Upload UI - Enhanced with PDF processing support.

Supports both CSV and PDF file uploads with advanced processing strategies,
and comprehensive progress tracking.
"""

import logging
import os
import tempfile
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import pypdf
import streamlit as st

from backend.chunking.csv_grouping_chunker import CSVGroupingChunker
from backend.session_manager import SessionManager
from backend.strategies.csv_strategy import CSVProcessingStrategy
from backend.vector_db.qdrant_manager import QdrantManager
from config.constants import (
    CSV_UI_MESSAGES,
    DEFAULT_CSV_CONFIG,
    EN,
    ENGLISH,
    PDF_PROCESSING_STRATEGIES,
    PDF_SIZE_LIMIT_MB,
    PDF_SIZE_WARNING_MB,
    VI,
    VIETNAMESE,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataUploadUI:
    """UI component for data upload with language selection."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def render(self, header_number: int) -> None:
        """Render upload interface."""
        # Language selection
        st.subheader(f"{header_number}.1. Select Document Language", divider=True)
        self._render_language_selection()

        # Check prerequisites
        if not self._check_prerequisites():
            return

        # Upload section
        st.subheader(f"{header_number}.2. Upload Files", divider=True)
        self._render_upload_section()

        # Save button
        if (
            self.session_manager.get("chunks_df") is not None
            and not self.session_manager.get("chunks_df").empty
            and not self.session_manager.get("data_saved_success")
        ):
            st.divider()
            if st.button("💾 Save to Vector Database", type="primary"):
                self._handle_save_data()

    def _render_language_selection(self) -> None:
        """Render language selection for this upload."""
        st.info(
            "💡 Select the language of the documents you're about to upload. "
            "This will be stored as metadata."
        )

        language_choice = st.radio(
            "Document Language:",
            [ENGLISH, VIETNAMESE],
            index=0,
            horizontal=True,
            help="Select the primary language of your documents",
        )

        lang_code = EN if language_choice == ENGLISH else VI
        self.session_manager.set("language", lang_code)
        st.success(f"✅ Language: **{language_choice}** (`{lang_code}`)")

    def _check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        if not self.session_manager.is_embedding_configured():
            st.warning(
                "⚠️ Embeddings not configured. Please enter API key in sidebar first."
            )
            return False

        return True

    def _render_upload_section(self) -> None:
        """Render enhanced file upload section with PDF support."""
        # Enhanced file uploader with both CSV and PDF support
        uploaded_files = st.file_uploader(
            "Upload CSV and PDF files",
            type=["csv", "pdf"],
            accept_multiple_files=True,
            help="Upload one or more CSV or PDF files. PDFs will be processed with OCR "
            "and semantic chunking.",
        )

        if uploaded_files:
            # Analyze uploaded files
            files_info = self._analyze_uploaded_files(uploaded_files)
            self.session_manager.set("uploaded_files_info", files_info)

            # Display file analysis
            self._display_file_analysis(files_info)

            # Show PDF processing options if PDFs are present
            csv_files = [f for f in uploaded_files if f.name.lower().endswith(".csv")]
            pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]

            if pdf_files:
                st.subheader("📄 PDF Processing Configuration", divider="gray")
                self._render_pdf_processing_options()

            if csv_files:
                st.subheader("📊 CSV Processing Configuration", divider="gray")
                self._render_csv_column_selection(csv_files)

            # Process button
            st.divider()
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                self._process_uploaded_files(uploaded_files)

    def _analyze_uploaded_files(self, uploaded_files: List) -> List[Dict]:
        """
        Analyze uploaded files and extract metadata.

        Args:
            uploaded_files: List of uploaded file objects

        Returns:
            List of dictionaries with file information
        """
        files_info = []
        total_size_mb = 0

        for uploaded_file in uploaded_files:
            file_info = {
                "name": uploaded_file.name,
                "type": (
                    "CSV" if uploaded_file.name.lower().endswith(".csv") else "PDF"
                ),
                "size_mb": round(uploaded_file.size / (1024 * 1024), 2),
                "size_bytes": uploaded_file.size,
                "warnings": [],
            }

            # Add size warnings for PDFs
            if file_info["type"] == "PDF":
                if file_info["size_mb"] > PDF_SIZE_LIMIT_MB:
                    file_info["warnings"].append(
                        f"⚠️ File too large ({file_info['size_mb']}MB >"
                        f"{PDF_SIZE_LIMIT_MB}MB limit)"
                    )
                elif file_info["size_mb"] > PDF_SIZE_WARNING_MB:
                    file_info["warnings"].append(
                        f"⚠️ Large file ({file_info['size_mb']}MB). Processing may take time."
                    )

            total_size_mb += file_info["size_mb"]
            files_info.append(file_info)

        return files_info

    def _display_file_analysis(self, files_info: List[Dict]) -> None:
        """Display analysis of uploaded files."""
        st.subheader("📋 File Analysis", divider="gray")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(files_info))
        with col2:
            csv_count = sum(1 for f in files_info if f["type"] == "CSV")
            st.metric("CSV Files", csv_count)
        with col3:
            pdf_count = sum(1 for f in files_info if f["type"] == "PDF")
            st.metric("PDF Files", pdf_count)

        # Detailed file information
        for file_info in files_info:
            file_type_icon = "📊" if file_info["type"] == "CSV" else "📄"

            with st.expander(
                f"{file_type_icon} {file_info['name']} ({file_info['size_mb']} MB)",
                expanded=True,
            ):
                st.write(f"**Type:** {file_info['type']}")
                st.write(f"**Size:** {file_info['size_mb']} MB")

                # File type specific information
                if file_info["type"] == "PDF":
                    st.info(
                        "🤖 This PDF will be processed with OCR and semantic chunking"
                    )
                else:
                    st.info("📊 This CSV will be processed with standard chunking")

                if file_info["warnings"]:
                    for warning in file_info["warnings"]:
                        st.warning(warning)
                else:
                    st.success("✅ Ready to process")

    def _render_pdf_processing_options(self) -> None:
        """Render PDF processing configuration options."""
        # Processing Strategy Selection
        st.info("💡 **PDF Processing Strategy** - Choose how to process your PDF files")

        # Get available strategies
        strategy_options = list(PDF_PROCESSING_STRATEGIES.keys())

        strategy_choice = st.selectbox(
            "Processing Strategy:",
            options=strategy_options,
            index=0,  # Default to first option
            help="Auto: Automatically selects the best strategy based on PDF content",
            key="pdf_strategy_selectbox",
        )

        self.session_manager.set("pdf_processing_strategy", strategy_choice)

        # Advanced options in expander
        with st.expander("🔧 Advanced PDF Processing Options", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                semantic_chunking = st.checkbox(
                    "Enable Semantic Chunking",
                    value=self.session_manager.get("pdf_semantic_chunking", True),
                    help="Use title-based semantic chunking for better content understanding",
                )
                self.session_manager.set("pdf_semantic_chunking", semantic_chunking)

            with col2:
                if semantic_chunking:
                    chunk_size = st.slider(
                        "Chunk Size (characters):",
                        min_value=500,
                        max_value=3000,
                        value=self.session_manager.get("pdf_chunk_size", 1000),
                        step=100,
                        help="Maximum size of each chunk",
                    )
                    self.session_manager.set("pdf_chunk_size", chunk_size)

                    chunk_overlap = st.slider(
                        "Chunk Overlap (characters):",
                        min_value=0,
                        max_value=500,
                        value=self.session_manager.get("pdf_chunk_overlap", 100),
                        step=50,
                        help="Overlap between chunks for context preservation",
                    )
                    self.session_manager.set("pdf_chunk_overlap", chunk_overlap)

        # Strategy information
        strategy_info = {
            "auto": "🤖 Automatically detects the best processing method based on PDF characteristics",
            "ocr_only": "🔍 Forces OCR processing for image-based PDFs and scanned documents",
            "hi_res": "🔬 High-resolution processing with OCR for image-based PDFs",
            "fast": "⚡ Skips OCR for faster processing of text-only PDFs",
            "fallback": "🛡️ Uses basic PDF processing as fallback for problematic files",
        }

        st.info(
            f"**{PDF_PROCESSING_STRATEGIES[strategy_choice]}**: {strategy_info[strategy_choice]}"
        )

    def _render_csv_column_selection(self, csv_files: List) -> Dict[str, Any]:
        """
        Render column selection interface for CSV files.

        Args:
            csv_files: List of uploaded CSV files

        Returns:
            Dictionary containing CSV configuration for each file
        """
        csv_configs = {}

        # Help message
        st.info(CSV_UI_MESSAGES["upload_help"])

        # Process each CSV file
        for csv_file in csv_files:
            with st.expander(f"⚙️ Configure {csv_file.name}", expanded=True):
                try:
                    # Analyze CSV structure - reset file pointer first
                    csv_file.seek(0)  # Reset file pointer to beginning
                    df_sample = pd.read_csv(csv_file, nrows=100)

                    # Display basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📏 Rows", len(df_sample))
                    with col2:
                        st.metric("📊 Columns", len(df_sample.columns))
                    with col3:
                        file_size_mb = round(csv_file.size / (1024 * 1024), 2)
                        st.metric("💾 Size", f"{file_size_mb} MB")

                    # Preview CSV structure - using tabs instead of nested expander
                    preview_tab, column_info_tab = st.tabs(
                        ["📋 Data Preview", "📊 Column Information"]
                    )

                    with preview_tab:
                        st.dataframe(df_sample.head(), use_container_width=True)

                    with column_info_tab:
                        # Column information
                        column_info = []
                        for col in df_sample.columns:
                            null_count = df_sample[col].isnull().sum()
                            unique_count = df_sample[col].nunique()
                            sample_values = df_sample[col].dropna().head(2).tolist()

                            column_info.append(
                                {
                                    'Column': col,
                                    'Type': str(df_sample[col].dtype),
                                    'Unique Values': unique_count,
                                    'Null Count': null_count,
                                    'Sample Values': ', '.join(
                                        str(v) for v in sample_values
                                    ),
                                }
                            )

                        col_info_df = pd.DataFrame(column_info)
                        st.dataframe(col_info_df, use_container_width=True)

                    # Column selection
                    selected_columns = st.multiselect(
                        "🏷️ Select Grouping Columns",
                        options=list(df_sample.columns),
                        help=CSV_UI_MESSAGES["column_selection_help"],
                        format_func=lambda x: f"📊 {x} ({df_sample[x].dtype})",
                    )

                    # Configuration options
                    col1, col2 = st.columns(2)

                    with col1:
                        max_rows_per_chunk = st.slider(
                            "📏 Maximum Rows per Chunk",
                            min_value=1,
                            max_value=50,
                            value=DEFAULT_CSV_CONFIG["max_rows_per_chunk"],
                            help="Limit the number of rows per chunk to prevent overly "
                            "large chunks",
                        )

                    with col2:
                        include_headers = st.checkbox(
                            "📋 Include Column Headers",
                            value=DEFAULT_CSV_CONFIG["include_headers"],
                            help="Add column headers to each chunk for better context",
                        )

                    # Preview configuration using tabs instead of nested expander
                    if selected_columns:
                        # Create a preview section without nested expanders
                        st.markdown("### 👁️ Chunk Preview")
                        self._render_csv_chunk_preview(
                            df_sample,
                            {
                                "selected_columns": selected_columns,
                                "max_rows_per_chunk": max_rows_per_chunk,
                                "include_headers": include_headers,
                                "csv_file_name": csv_file.name,  # Add file name for unique keys
                            },
                        )

                    # Store configuration
                    csv_configs[csv_file.name] = {
                        "selected_columns": selected_columns,
                        "max_rows_per_chunk": max_rows_per_chunk,
                        "include_headers": include_headers,
                        "file_size_mb": file_size_mb,
                        "total_columns": len(df_sample.columns),
                        "sample_rows": len(df_sample),
                    }

                except Exception as e:
                    st.error(f"❌ Error analyzing {csv_file.name}: {str(e)}")
                    # Fallback configuration
                    csv_configs[csv_file.name] = {
                        "selected_columns": [],
                        "max_rows_per_chunk": DEFAULT_CSV_CONFIG["max_rows_per_chunk"],
                        "include_headers": DEFAULT_CSV_CONFIG["include_headers"],
                        "error": str(e),
                    }

        # Store configurations in session state
        st.session_state.csv_processing_configs = csv_configs
        return csv_configs

    def _render_csv_chunk_preview(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> None:
        """
        Render preview of chunks based on column selection.

        Args:
            df: Sample DataFrame
            config: Chunking configuration
        """
        selected_columns = config.get("selected_columns", [])
        max_rows_per_chunk = config.get("max_rows_per_chunk", 10)
        csv_file_name = config.get("csv_file_name", "preview")

        if not selected_columns:
            st.info("Select columns to see chunk preview")
            return

        try:
            # Create chunker and generate sample chunks
            chunker = CSVGroupingChunker()
            sample_chunks = chunker.chunk_dataframe(
                df.head(20),  # Limit preview to first 20 rows
                group_columns=selected_columns,
                max_rows_per_chunk=max_rows_per_chunk,
            )

            if not sample_chunks:
                st.warning("No chunks could be generated with current selection")
                return

            # Display preview chunks using columns instead of nested expanders
            st.write(f"**Preview: {len(sample_chunks)} chunk(s) will be created**")

            # Create tabs for each chunk to avoid nesting
            if sample_chunks:
                chunk_tabs = st.tabs(
                    [f"Chunk {i+1}" for i in range(min(3, len(sample_chunks)))]
                )

                for i, (chunk, tab) in enumerate(zip(sample_chunks[:3], chunk_tabs)):
                    with tab:
                        chunk_size = len(chunk.get("chunk", ""))
                        row_count = chunk.get("row_count", 0)

                        st.write(
                            f"**Rows:** {row_count} | **Characters:** {chunk_size}"
                        )

                        st.text_area(
                            f"Chunk {i+1} Content:",
                            chunk.get("chunk", ""),
                            height=150,
                            disabled=True,
                            key=f"chunk_content_{i}_{csv_file_name}",
                        )

                        # Show metadata as collapsible section using st.accordion
                        if chunk.get("metadata"):
                            st.write("**Metadata:**")
                            st.json(chunk["metadata"])

        except Exception as e:
            st.error(f"Error generating preview: {str(e)}")

    def _process_uploaded_files(self, uploaded_files: List) -> None:
        """
        Process uploaded files with enhanced PDF support and progress tracking.

        Args:
            uploaded_files: List of uploaded file objects
        """
        try:
            # Initialize progress tracking
            progress_container = st.container()
            with progress_container:
                st.subheader("🔄 Processing Files", divider="gray")
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_text = st.empty()

            all_chunks = []
            processing_stats = {
                "total_files": len(uploaded_files),
                "processed_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "processing_time": 0.0,
            }

            start_time = time.time()

            for i, uploaded_file in enumerate(uploaded_files):
                file_extension = uploaded_file.name.split(".")[-1].lower()
                file_progress = (i + 1) / len(uploaded_files)

                # Update progress
                progress_bar.progress(file_progress)
                status_text.markdown(
                    f"**Processing file {i+1}/{len(uploaded_files)}**: `{uploaded_file.name}`"
                )
                details_text.info(f"🔍 Analyzing file type: {file_extension.upper()}")

                try:
                    if file_extension == "csv":
                        chunks = self._process_csv_file_enhanced(
                            uploaded_file, details_text
                        )
                        if chunks:
                            all_chunks.extend(chunks)
                            processing_stats["processed_files"] += 1
                            details_text.success(
                                f"✅ CSV file processed: {len(chunks)} chunks created"
                            )

                    elif file_extension == "pdf":
                        chunks = self._process_pdf_file(uploaded_file, details_text)
                        if chunks:
                            all_chunks.extend(chunks)
                            processing_stats["processed_files"] += 1
                            details_text.success(
                                f"✅ PDF file processed: {len(chunks)} chunks created"
                            )

                    else:
                        details_text.warning(
                            f"⚠️ Skipping unsupported file type: {file_extension}"
                        )
                        processing_stats["failed_files"] += 1

                except Exception as file_error:
                    processing_stats["failed_files"] += 1
                    details_text.error(
                        f"❌ Error processing {uploaded_file.name}: {str(file_error)}"
                    )
                    continue

            # Complete progress
            processing_stats["processing_time"] = round(time.time() - start_time, 2)
            processing_stats["total_chunks"] = len(all_chunks)

            progress_bar.progress(1.0)
            status_text.markdown("**Processing Complete!**")

            # Clear progress container
            time.sleep(2)
            progress_container.empty()

            # Display results
            if all_chunks:
                self._display_processing_results(all_chunks, processing_stats)
                self._render_enhanced_chunking_section()
            else:
                st.error("❌ No valid content was extracted from the uploaded files")

        except Exception as e:
            st.error(f"❌ Critical Error: {str(e)}")
            st.code(traceback.format_exc())

    def _process_csv_file(self, uploaded_file, status_text) -> List[Dict]:
        """Process a single CSV file."""
        status_text.info("📊 Reading CSV file...")

        # Reset file pointer to ensure we read from the beginning
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        chunks = []

        for idx, row in df.iterrows():
            # Create a chunk for each row
            chunk_text = " | ".join(
                [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            )

            chunk = {
                "chunk": chunk_text,
                "source_file": uploaded_file.name,
                "file_type": "CSV",
                "row_index": idx,
                "doc_id": str(uuid.uuid4()),
                "metadata": {
                    "source_row": idx,
                    "total_columns": len(df.columns),
                    "columns": list(df.columns),
                },
            }
            chunks.append(chunk)

        return chunks

    def _process_csv_file_enhanced(self, uploaded_file, status_text) -> List[Dict]:
        """
        Process a single CSV file with enhanced column-based chunking.

        Args:
            uploaded_file: Uploaded CSV file object
            status_text: Streamlit status text element for updates

        Returns:
            List of chunk dictionaries
        """
        try:
            status_text.info("🔧 Initializing enhanced CSV processor...")

            # Get CSV configuration from session state
            csv_configs = getattr(st.session_state, 'csv_processing_configs', {})
            csv_config = csv_configs.get(uploaded_file.name, {})

            # Use configuration from UI or defaults
            selected_columns = csv_config.get("selected_columns", [])
            max_rows_per_chunk = csv_config.get(
                "max_rows_per_chunk", DEFAULT_CSV_CONFIG["max_rows_per_chunk"]
            )
            include_headers = csv_config.get(
                "include_headers", DEFAULT_CSV_CONFIG["include_headers"]
            )

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                # Reset file pointer before reading
                uploaded_file.seek(0)
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            try:
                strategy_config = {
                    "max_rows_per_chunk": max_rows_per_chunk,
                    "include_headers": include_headers,
                    "encoding": "utf-8",
                    "delimiter": ",",
                }

                strategy = CSVProcessingStrategy(config=strategy_config)

                status_text.info(
                    f"📊 Processing CSV with column grouping:"
                    f"{selected_columns if selected_columns else 'Row by row'}"
                )

                # Process CSV with strategy
                processing_result = strategy.extract_elements(
                    temp_file_path,
                    selected_columns=selected_columns,
                    max_rows_per_chunk=max_rows_per_chunk,
                    include_headers=include_headers,
                )

                if not processing_result.success:
                    status_text.error(
                        f"❌ CSV processing failed: {processing_result.error_message}"
                    )
                    return []

                # Convert processing result elements to chunk format
                chunks = []
                for element in processing_result.elements:
                    chunk = {
                        "chunk": element.get("text", ""),
                        "source_file": uploaded_file.name,
                        "file_type": "CSV",
                        "doc_id": element.get("metadata", {}).get(
                            "doc_id", str(uuid.uuid4())
                        ),
                        "metadata": {
                            **element.get("metadata", {}),
                            "processing_strategy": "enhanced_csv",
                            "selected_columns": selected_columns,
                            "max_rows_per_chunk": max_rows_per_chunk,
                            "include_headers": include_headers,
                        },
                    }
                    chunks.append(chunk)

                # Log processing statistics
                metadata = processing_result.metadata
                status_text.success(
                    f"✅ Enhanced CSV processing complete: {len(chunks)} chunks"
                    f"from {metadata.get('total_rows', 0)} rows"
                )

                return chunks

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            status_text.error(
                f"❌ Enhanced CSV processing failed, trying fallback: {str(e)}"
            )
            logger.error(f"Enhanced CSV processing failed: {e}")

            # Fallback to original CSV processing
            try:
                status_text.info("🔄 Using fallback CSV processing...")
                return self._process_csv_file(uploaded_file, status_text)
            except Exception as fallback_error:
                status_text.error(
                    f"❌ Fallback CSV processing also failed: {str(fallback_error)}"
                )
                return []

    def _process_pdf_file(self, uploaded_file, status_text) -> List[Dict]:
        """Process a single PDF file using the session-managed document processor."""
        try:
            status_text.info("📄 Initializing PDF processor...")

            # Check if session manager has PDF processor available
            if not self.session_manager.is_pdf_processor_available():
                status_text.warning(
                    "⚠️ PDF processor not available. Using basic PDF processing."
                )
                return self._fallback_pdf_processing(uploaded_file, status_text)

            status_text.info("🔍 Extracting content from PDF...")

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            try:
                # Process the PDF using session-managed processor
                with st.spinner("Processing PDF..."):
                    result = self.session_manager.process_document_with_session(
                        temp_file_path,
                        languages=[self.session_manager.get("language", "en")],
                        original_filename=uploaded_file.name,
                    )

                if result and result.success:
                    status_text.success("✅ PDF content extracted successfully")

                    # Convert processing result to chunks
                    chunks = []
                    for i, element in enumerate(result.elements):
                        chunk = {
                            "chunk": element.text,
                            "source_file": uploaded_file.name,
                            "file_type": "PDF",
                            "chunk_index": i,
                            "doc_id": str(uuid.uuid4()),
                            "metadata": {
                                "processing_strategy": result.metadata.get(
                                    "strategy_used", "unknown"
                                ),
                                "ocr_used": result.metadata.get("ocr_used", False),
                                "total_pages": result.metadata.get("total_pages", 0),
                                "chunk_type": "extracted_content",
                                "element_type": getattr(element, 'category', 'text'),
                                "page_number": self._extract_page_number_from_element(
                                    element
                                ),
                            },
                        }
                        chunks.append(chunk)

                    return chunks

                else:
                    error_msg = result.error_message if result else "Unknown error"
                    status_text.error(f"❌ PDF processing failed: {error_msg}")
                    return []

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            status_text.error(f"❌ PDF processing error: {str(e)}")
            logger.error(
                f"PDF processing error for {uploaded_file.name}: {e}", exc_info=True
            )
            return []

    def _fallback_pdf_processing(self, uploaded_file, status_text) -> List[Dict]:
        """Fallback PDF processing using basic libraries."""
        status_text.info("🔄 Attempting fallback PDF processing...")

        try:
            # Read PDF content
            pdf_reader = pypdf.PdfReader(uploaded_file)
            text_content = ""

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text_content += f"\n\n--- Page {page_num + 1} ---\n{page_text}"

            if text_content.strip():
                chunks = [
                    {
                        "chunk": text_content.strip(),
                        "source_file": uploaded_file.name,
                        "file_type": "PDF",
                        "chunk_index": 0,
                        "doc_id": str(uuid.uuid4()),
                        "metadata": {
                            "processing_strategy": "fallback_pypdf2",
                            "ocr_used": False,
                            "total_pages": len(pdf_reader.pages),
                            "chunk_type": "fallback_extraction",
                        },
                    }
                ]
                return chunks
            else:
                status_text.warning("⚠️ No text content found in PDF")
                return []

        except Exception as e:
            status_text.error(f"❌ Fallback processing failed: {str(e)}")
            return []

    def _display_processing_results(self, chunks: List[Dict], stats: Dict) -> None:
        """Display processing results and statistics."""
        st.success("🎉 Files processed successfully!")

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Files Processed",
                f"{stats['processed_files']}/{stats['total_files']}",
            )
        with col2:
            st.metric("Total Chunks", stats["total_chunks"])
        with col3:
            st.metric("Failed Files", stats["failed_files"])
        with col4:
            st.metric("Processing Time", f"{stats['processing_time']}s")

        # Create DataFrame for chunks
        chunks_df = pd.DataFrame(chunks)
        self.session_manager.set("chunks_df", chunks_df)

        # Display sample chunks
        if len(chunks_df) > 0:
            st.subheader("📝 Extracted Content Preview", divider="gray")

            # Show first few chunks
            display_count = min(5, len(chunks_df))
            for idx, (i, chunk_row) in enumerate(
                chunks_df.head(display_count).iterrows()
            ):
                with st.expander(
                    f"Chunk {idx+1} - {chunk_row['source_file']}",
                    expanded=idx == 0,
                ):
                    st.write("**Source:**", chunk_row["source_file"])
                    st.write("**Type:**", chunk_row["file_type"])
                    st.write("**Content Preview:**")
                    st.text_area(
                        "Content",
                        (
                            chunk_row["chunk"][:500] + "..."
                            if len(chunk_row["chunk"]) > 500
                            else chunk_row["chunk"]
                        ),
                        height=150,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"chunk_content_{i}",
                    )
                    if "metadata" in chunk_row:
                        st.write("**Metadata:**")
                        st.json(chunk_row["metadata"])

            if len(chunks_df) > display_count:
                st.info(
                    f"📊 Showing first {display_count} of {len(chunks_df)} total chunks"
                )

    def _render_enhanced_chunking_section(self) -> None:
        """Render enhanced chunking configuration section."""
        chunks_df = self.session_manager.get("chunks_df")

        if chunks_df is None or chunks_df.empty:
            st.warning("No chunks available for further processing")
            return

    def _extract_page_number_from_element(self, element) -> Optional[int]:
        """
        Extract page number from an unstructured element.

        Args:
            element: Unstructured document element

        Returns:
            Page number if available, None otherwise
        """
        try:
            # Try to get page number from element metadata
            if hasattr(element, 'metadata') and hasattr(
                element.metadata, 'page_number'
            ):
                page_num = element.metadata.page_number
                if page_num is not None:
                    return int(page_num)

            # Fallback: check for page_number attribute directly on element
            if hasattr(element, 'page_number'):
                page_num = element.page_number
                if page_num is not None:
                    return int(page_num)

            return None
        except (ValueError, TypeError, AttributeError):
            return None

    def _handle_save_data(self) -> None:
        """Save enhanced chunks data to Qdrant."""
        chunks_df = self.session_manager.get("chunks_df")

        if chunks_df is None or chunks_df.empty:
            st.warning("No data to save")
            return

        embedding_strategy = self.session_manager.get("embedding_strategy")
        if not embedding_strategy:
            st.error("❌ Embeddings not configured")
            return

        language = self.session_manager.get("language")
        if not language:
            st.error("❌ Language not selected")
            return

        try:
            # Enhanced save progress tracking
            progress_container = st.container()
            with progress_container:
                st.subheader("💾 Saving to Vector Database", divider="gray")
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_text = st.empty()

            total_chunks = len(chunks_df)

            # Step 1: Connect to Qdrant
            progress_bar.progress(0.1)
            status_text.markdown("**Step 1/5: Connecting to Qdrant...**")
            details_text.info("🔄 Initializing vector database connection...")

            qdrant_manager = QdrantManager()

            if not qdrant_manager.is_healthy():
                progress_container.empty()
                st.error(
                    "❌ Qdrant not running. Start with:\n"
                    "```bash\ndocker-compose up -d\n```"
                )
                return

            details_text.success("✅ Connected to Qdrant successfully")

            # Step 2: Setup Collection
            progress_bar.progress(0.2)
            status_text.markdown("**Step 2/5: Setting up collection...**")
            details_text.info("🔄 Creating/verifying vector collection...")

            dimension = self.session_manager.get("embedding_dimension")

            if not qdrant_manager.ensure_collection(dimension):
                progress_container.empty()
                st.error("❌ Failed to create collection")
                return

            details_text.success(
                f"✅ Collection '{qdrant_manager.collection_name}' ready"
            )

            # Step 3: Generate Embeddings
            progress_bar.progress(0.3)
            status_text.markdown("**Step 3/5: Generating embeddings...**")
            details_text.info(f"🔄 Processing {total_chunks} chunks...")

            chunks = chunks_df["chunk"].tolist()

            # Embeddings with progress
            try:
                embeddings = embedding_strategy.embed_texts(chunks)
                details_text.success(f"✅ Generated {len(embeddings)} embeddings")
            except Exception as e:
                progress_container.empty()
                st.error(f"❌ Embedding error: {str(e)}")
                return

            # Step 4: Prepare Enhanced Metadata
            progress_bar.progress(0.6)
            status_text.markdown("**Step 4/5: Preparing enhanced metadata...**")
            details_text.info("🔄 Processing document metadata...")

            # Enhanced source file tracking
            source_files = (
                chunks_df["source_file"].unique()
                if "source_file" in chunks_df.columns
                else ["uploaded_files"]
            )
            file_types = (
                chunks_df["file_type"].value_counts().to_dict()
                if "file_type" in chunks_df.columns
                else {}
            )

            # Create enhanced source description
            source_description = f"uploaded_{len(source_files)}_files"
            if file_types:
                type_descriptions = [
                    f"{count}_{file_type.lower()}"
                    for file_type, count in file_types.items()
                ]
                source_description = "_".join(type_descriptions)

            details_text.success(
                f"✅ Metadata prepared for {len(source_files)} source files"
            )

            # Step 5: Upload to Database
            progress_bar.progress(0.8)
            status_text.markdown("**Step 5/5: Uploading to database...**")
            details_text.info(f"🔄 Storing {total_chunks} document vectors...")

            success = qdrant_manager.add_documents(
                chunks_df=chunks_df,
                embeddings=embeddings,
                language=language,
                source_file=source_description,
            )

            if not success:
                progress_container.empty()
                st.error("❌ Upload failed")
                return

            # Complete progress
            progress_bar.progress(1.0)
            status_text.markdown("**Save Complete!**")
            details_text.success("✅ All documents saved to vector database")

            # Save to session with enhanced information
            self.session_manager.update(
                {
                    "qdrant_manager": qdrant_manager,
                    "collection_name": qdrant_manager.collection_name,
                    "data_saved_success": True,
                    "source_data": "UPLOAD",
                    "uploaded_file_types": file_types,
                    "last_upload_stats": {
                        "total_chunks": total_chunks,
                        "source_files_count": len(source_files),
                        "file_types": file_types,
                        "language": language,
                    },
                }
            )

            # Clear progress and show success
            time.sleep(2)
            progress_container.empty()

            st.success("🎉 Data saved successfully!")

            # Enhanced statistics display
            stats = qdrant_manager.get_statistics()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
            with col2:
                st.metric("Collection", qdrant_manager.collection_name)
            with col3:
                st.metric("Language Saved", language.upper())
            with col4:
                st.metric("Chunks Uploaded", total_chunks)

            # File type breakdown
            if file_types:
                st.subheader("📄 File Type Breakdown", divider="gray")
                type_cols = st.columns(len(file_types))
                for i, (file_type, count) in enumerate(file_types.items()):
                    with type_cols[i]:
                        st.metric(f"{file_type} Files", count)

            # Success message with processing details
            files_info = self.session_manager.get("uploaded_files_info", [])
            if files_info:
                st.info(
                    f"📊 Successfully processed and saved content from "
                    f"{len(files_info)} uploaded files"
                )

        except Exception as e:
            st.error(f"❌ Save Error: {str(e)}")
            st.code(traceback.format_exc())
