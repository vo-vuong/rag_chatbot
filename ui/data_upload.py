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
    PDF_PROCESSING_MODES,
    PDF_SIZE_LIMIT_MB,
    PDF_SIZE_WARNING_MB,
    VI,
    VIETNAMESE,
)
from ui.api_client import (
    PreviewChunk,
    PreviewImage,
    PreviewResult,
    SaveResult,
    UploadResult,
    get_api_client,
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

        # Check for preview data and show save button
        if self.session_manager.get("upload_preview_data"):
            self._display_preview_results()

            st.divider()
            if st.button("💾 Save to Vector Database", type="primary", use_container_width=True):
                self._handle_save_preview_data()

        # Legacy: Save button for old workflow (chunks_df)
        if (
            self.session_manager.get("chunks_df") is not None
            and not self.session_manager.get("chunks_df").empty
            and not self.session_manager.get("data_saved_success")
            and not self.session_manager.get("upload_preview_data")
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
        # Enhanced file uploader with CSV, PDF, and DOCX support
        uploaded_files = st.file_uploader(
            "Upload CSV, PDF, and DOCX files",
            type=["csv", "pdf", "docx"],
            accept_multiple_files=True,
            help="Upload CSV, PDF, or DOCX files. PDFs support optional OCR."
        )

        if uploaded_files:
            # Analyze uploaded files
            files_info = self._analyze_uploaded_files(uploaded_files)
            self.session_manager.set("uploaded_files_info", files_info)

            # Display file analysis
            self._display_file_analysis(files_info)

            # Categorize files by type
            csv_files = [f for f in uploaded_files if f.name.lower().endswith(".csv")]
            pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
            docx_files = [f for f in uploaded_files if f.name.lower().endswith(".docx")]

            # Combined document files (PDF + DOCX)
            document_files = pdf_files + docx_files

            if document_files:
                st.subheader("📄 Document Processing Configuration", divider="gray")

                # Show OCR options only if PDFs are present
                if pdf_files:
                    self._render_pdf_processing_options()
                else:
                    # DOCX only - show info that OCR not needed
                    st.info("💡 DOCX files have embedded text - no OCR required.")

            if csv_files:
                st.subheader("📊 CSV Processing Configuration", divider="gray")
                self._render_csv_column_selection(csv_files)

            # Process button - uses preview API for two-step workflow
            st.divider()
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                self._process_files_for_preview(uploaded_files)

    def _get_file_type(self, filename: str) -> str:
        """Determine file type from filename.

        Args:
            filename: Name of the file

        Returns:
            File type string (CSV, PDF, DOCX, or Unknown)
        """
        name_lower = filename.lower()
        if name_lower.endswith(".csv"):
            return "CSV"
        elif name_lower.endswith(".pdf"):
            return "PDF"
        elif name_lower.endswith(".docx"):
            return "DOCX"
        return "Unknown"

    def _get_file_icon(self, file_type: str) -> str:
        """Get icon for file type.

        Args:
            file_type: Type of file (CSV, PDF, DOCX)

        Returns:
            Emoji icon for the file type
        """
        icons = {"CSV": "📊", "PDF": "📄", "DOCX": "📝"}
        return icons.get(file_type, "📁")

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
            file_type = self._get_file_type(uploaded_file.name)
            file_info = {
                "name": uploaded_file.name,
                "type": file_type,
                "size_mb": round(uploaded_file.size / (1024 * 1024), 2),
                "size_bytes": uploaded_file.size,
                "warnings": [],
            }

            # Add size warnings for PDFs and DOCX
            if file_info["type"] in ("PDF", "DOCX"):
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

        # Summary metrics with DOCX support
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(files_info))
        with col2:
            csv_count = sum(1 for f in files_info if f["type"] == "CSV")
            st.metric("CSV Files", csv_count)
        with col3:
            pdf_count = sum(1 for f in files_info if f["type"] == "PDF")
            st.metric("PDF Files", pdf_count)
        with col4:
            docx_count = sum(1 for f in files_info if f["type"] == "DOCX")
            st.metric("DOCX Files", docx_count)

        # Detailed file information
        for file_info in files_info:
            file_type_icon = self._get_file_icon(file_info["type"])

            with st.expander(
                f"{file_type_icon} {file_info['name']} ({file_info['size_mb']} MB)",
                expanded=True,
            ):
                st.write(f"**Type:** {file_info['type']}")
                st.write(f"**Size:** {file_info['size_mb']} MB")

                if file_info["warnings"]:
                    for warning in file_info["warnings"]:
                        st.warning(warning)
                else:
                    st.success("✅ Ready to process")

    def _render_pdf_processing_options(self) -> None:
        """Render PDF processing configuration options."""
        # Processing Strategy Selection
        st.info("💡 **PDF Processing Mode** - Choose how to process your PDF files")

        # Get available modes
        mode_options = list(PDF_PROCESSING_MODES.keys())

        mode_choice = st.selectbox(
            "Processing Mode:",
            options=mode_options,
            index=0,  # Default to first option
            help="Auto: Automatically selects the best mode based on PDF content",
            key="pdf_mode_selectbox",
        )

        self.session_manager.set("pdf_processing_mode", mode_choice)

        # Image Captioning Settings
        with st.expander("⚙️ Advanced Image Captioning Settings", expanded=False):
            st.markdown("### Caption Failure Handling")

            # Valid failure modes
            VALID_FAILURE_MODES = ["graceful", "strict", "skip"]

            failure_mode = st.radio(
                "What should happen if an image caption fails?",
                options=VALID_FAILURE_MODES,
                format_func=lambda x: {
                    "graceful": "🛡️ Graceful (Recommended) - Use fallback caption",
                    "strict": "⚠️ Strict - Abort entire upload",
                    "skip": "⏭️ Skip - Ignore failed images",
                }[x],
                index=0,  # Default to graceful
                help=(
                    "**Graceful**: Failed images get fallback caption 'Image (caption unavailable)'\n\n"
                    "**Strict**: Any caption failure aborts entire PDF upload\n\n"
                    "**Skip**: Failed images are not stored (only successful captions)"
                ),
                key="caption_failure_mode_radio",
            )

            # Validate failure mode
            if failure_mode not in VALID_FAILURE_MODES:
                st.error(f"❌ Invalid failure mode: {failure_mode}")
                failure_mode = "graceful"  # Fallback to safe default

            self.session_manager.set("caption_failure_mode", failure_mode)

            # Show warning for strict mode
            if failure_mode == "strict":
                st.warning(
                    "⚠️ **Strict mode**: If any image fails to caption "
                    "(due to API error, rate limit, etc.), the entire PDF upload will fail. "
                    "Use only if you require 100% caption coverage."
                )
            elif failure_mode == "graceful":
                st.success(
                    "✅ **Graceful mode** (Recommended): Upload continues even if some images fail to caption. "
                    "Failed images will have a placeholder caption."
                )
            elif failure_mode == "skip":
                st.info(
                    "ℹ️ **Skip mode**: Failed images will be skipped entirely. "
                    "Only successfully captioned images will be stored."
                )

        # Mode information
        mode_info = {
            "no_ocr": "⚡ Fast processing for text-based PDFs (no OCR needed)",
            "ocr": "🔍 Uses EasyOCR for scanned documents and image-based PDFs",
        }

        st.info(
            f"**{PDF_PROCESSING_MODES[mode_choice]}**: {mode_info[mode_choice]}"
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

    def _process_file_via_api(
        self,
        uploaded_file,
        status_text,
    ) -> UploadResult:
        """Process a single file via the upload API.

        Args:
            uploaded_file: Streamlit UploadedFile object
            status_text: Streamlit element for status updates

        Returns:
            UploadResult with processing status
        """
        api_client = get_api_client()

        # Get language from session
        language = self.session_manager.get("language", "en")

        # Get processing mode for PDFs
        pdf_mode = self.session_manager.get("pdf_processing_mode", "no_ocr")
        processing_mode = "ocr" if pdf_mode == "ocr" else "fast"

        # Get vision failure mode
        vision_failure_mode = self.session_manager.get(
            "caption_failure_mode", "graceful"
        )

        # Get CSV columns if configured
        csv_columns = None
        csv_configs = getattr(st.session_state, "csv_processing_configs", {})
        if uploaded_file.name in csv_configs:
            selected_cols = csv_configs[uploaded_file.name].get("selected_columns", [])
            if selected_cols:
                csv_columns = ",".join(selected_cols)

        # Read file content
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for potential reuse

        status_text.info(f"📤 Uploading {uploaded_file.name} to API...")

        # Upload via API
        result = api_client.upload_file(
            file_content=file_content,
            file_name=uploaded_file.name,
            language=language,
            processing_mode=processing_mode,
            csv_columns=csv_columns,
            vision_failure_mode=vision_failure_mode,
        )

        return result

    def _process_files_for_preview(self, uploaded_files: List) -> None:
        """Process files via preview API and store results for user confirmation.

        This implements the first step of the two-step upload workflow.

        Args:
            uploaded_files: List of uploaded file objects
        """
        try:
            # Clear any previous preview data
            self.session_manager.set("upload_preview_data", None)
            self.session_manager.set("data_saved_success", False)

            # Initialize progress tracking
            progress_container = st.container()
            with progress_container:
                st.subheader("🔄 Processing Files for Preview", divider="gray")
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_text = st.empty()

                col1, col2, col3 = st.columns(3)
                with col1:
                    chunks_metric = st.empty()
                with col2:
                    images_metric = st.empty()
                with col3:
                    time_metric = st.empty()

            api_client = get_api_client()

            # Get settings
            language = self.session_manager.get("language", "en")
            pdf_mode = self.session_manager.get("pdf_processing_mode", "no_ocr")
            processing_mode = "ocr" if pdf_mode == "ocr" else "fast"
            vision_failure_mode = self.session_manager.get(
                "caption_failure_mode", "graceful"
            )

            # Track results
            all_preview_data: List[PreviewResult] = []
            total_chunks = 0
            total_images = 0
            processed_files = 0
            failed_files = 0
            start_time = time.time()

            for i, uploaded_file in enumerate(uploaded_files):
                file_progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(file_progress)
                status_text.markdown(
                    f"**Processing file {i+1}/{len(uploaded_files)}**: `{uploaded_file.name}`"
                )

                # Get CSV columns if configured
                csv_columns = None
                csv_configs = getattr(st.session_state, "csv_processing_configs", {})
                if uploaded_file.name in csv_configs:
                    selected_cols = csv_configs[uploaded_file.name].get("selected_columns", [])
                    if selected_cols:
                        csv_columns = ",".join(selected_cols)

                # Read file content
                file_content = uploaded_file.read()
                uploaded_file.seek(0)

                details_text.info(f"📤 Processing {uploaded_file.name}...")

                # Call preview API
                result = api_client.preview_upload(
                    file_content=file_content,
                    file_name=uploaded_file.name,
                    language=language,
                    processing_mode=processing_mode,
                    csv_columns=csv_columns,
                    vision_failure_mode=vision_failure_mode,
                )

                if result.success:
                    processed_files += 1
                    total_chunks += result.total_chunks_count
                    total_images += result.total_images_count
                    all_preview_data.append(result)
                    details_text.success(
                        f"✅ {result.file_name}: {result.total_chunks_count} chunks, "
                        f"{result.total_images_count} images ({result.processing_time:.1f}s)"
                    )
                else:
                    failed_files += 1
                    details_text.error(f"❌ {result.file_name}: {result.error}")

                # Update metrics
                chunks_metric.metric("Total Chunks", total_chunks)
                images_metric.metric("Total Images", total_images)
                elapsed = round(time.time() - start_time, 1)
                time_metric.metric("Elapsed Time", f"{elapsed}s")

            # Complete progress
            progress_bar.progress(1.0)
            total_time = round(time.time() - start_time, 2)
            status_text.markdown("**Processing Complete!**")

            # Brief pause then clear progress
            time.sleep(1)
            progress_container.empty()

            # Store preview data in session for save step
            if all_preview_data:
                self.session_manager.set("upload_preview_data", all_preview_data)
                st.success(
                    f"🎉 Processed {processed_files} file(s): "
                    f"{total_chunks} chunks, {total_images} images"
                )
                st.info(
                    "👁️ Review the preview below, then click "
                    "**Save to Vector Database** to store the data."
                )
            else:
                st.error("❌ No files were processed successfully")

            if failed_files > 0:
                st.warning(f"⚠️ {failed_files} file(s) failed to process")

        except Exception as e:
            st.error(f"❌ Processing Error: {str(e)}")
            logger.error(f"Preview processing error: {e}", exc_info=True)

    def _display_preview_results(self) -> None:
        """Display preview results from the preview API call."""
        preview_data: List[PreviewResult] = self.session_manager.get("upload_preview_data", [])

        if not preview_data:
            return

        st.subheader("📝 Preview Results", divider="gray")

        # Summary metrics
        total_chunks = sum(p.total_chunks_count for p in preview_data)
        total_images = sum(p.total_images_count for p in preview_data)
        total_preview_chunks = sum(len(p.preview_chunks) for p in preview_data)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files", len(preview_data))
        with col2:
            st.metric("Total Chunks", total_chunks)
        with col3:
            st.metric("Total Images", total_images)
        with col4:
            st.metric("Preview Chunks", total_preview_chunks)

        if total_chunks > total_preview_chunks:
            st.info(
                f"📊 Showing first {total_preview_chunks} of {total_chunks} chunks. "
                f"All {total_chunks} chunks will be saved."
            )

        # Display preview chunks per file
        for preview_result in preview_data:
            with st.expander(
                f"📄 {preview_result.file_name} ({preview_result.total_chunks_count} chunks)",
                expanded=len(preview_data) == 1,
            ):
                # Tabs for chunks and images
                if preview_result.preview_images:
                    chunk_tab, image_tab = st.tabs(["📝 Text Chunks", "🖼️ Images"])
                else:
                    chunk_tab = st.container()
                    image_tab = None

                with chunk_tab:
                    if preview_result.preview_chunks:
                        # Show first few preview chunks
                        display_count = min(5, len(preview_result.preview_chunks))
                        for idx, chunk in enumerate(preview_result.preview_chunks[:display_count]):
                            st.markdown(f"**Chunk {chunk.chunk_index + 1}:**")
                            preview_text = chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text
                            st.text_area(
                                f"Content",
                                preview_text,
                                height=100,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"preview_{preview_result.file_name}_{idx}",
                            )

                            # Basic info
                            basic_info = f"Page: {chunk.page_number or 'N/A'} | Type: {chunk.element_type}"
                            st.caption(basic_info)

                            # Metadata toggle (avoid nested expander)
                            show_meta = st.toggle(
                                "Show metadata",
                                key=f"meta_toggle_{preview_result.file_name}_{idx}",
                            )
                            if show_meta:
                                meta_col1, meta_col2 = st.columns(2)
                                with meta_col1:
                                    st.markdown(f"**Processing Strategy:** `{chunk.processing_strategy or 'N/A'}`")
                                    st.markdown(f"**Chunk Type:** `{chunk.chunk_type}`")
                                    st.markdown(f"**Source:** `{chunk.source}`")
                                    st.markdown(f"**OCR Used:** `{chunk.ocr_used}`")
                                with meta_col2:
                                    st.markdown(f"**Token Count:** `{chunk.token_count or 'N/A'}`")
                                    st.markdown(f"**File Type:** `{chunk.file_type}`")
                                    if chunk.headings:
                                        st.markdown(f"**Headings:** {', '.join(chunk.headings)}")
                                    if chunk.bbox:
                                        st.markdown(f"**BBox:** `{chunk.bbox}`")

                            st.divider()

                        if len(preview_result.preview_chunks) > display_count:
                            st.info(
                                f"Showing {display_count} of {len(preview_result.preview_chunks)} preview chunks"
                            )

                # Show images with metadata
                if image_tab and preview_result.preview_images:
                    with image_tab:
                        st.markdown(f"**{len(preview_result.preview_images)} image(s) extracted:**")
                        for img_idx, img in enumerate(preview_result.preview_images[:5]):
                            st.markdown(f"**Image {img_idx + 1}:**")
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                # Try to display image if path exists
                                if img.image_path:
                                    try:
                                        st.image(img.image_path, width=150)
                                    except Exception:
                                        st.caption(f"📷 {img.image_path}")
                            with col2:
                                st.markdown(f"**Caption:** {img.caption[:200]}..." if len(img.caption) > 200 else f"**Caption:** {img.caption}")
                                st.caption(f"Page: {img.page_number or 'N/A'} | Hash: {img.image_hash[:8]}...")

                                # Image metadata toggle (avoid nested expander)
                                show_img_meta = st.toggle(
                                    "Show image metadata",
                                    key=f"img_meta_toggle_{preview_result.file_name}_{img_idx}",
                                )
                                if show_img_meta:
                                    st.markdown(f"**Processing Strategy:** `{img.processing_strategy}`")
                                    st.markdown(f"**File Type:** `{img.file_type}`")
                                    st.markdown(f"**Language:** `{img.language}`")
                                    st.markdown(f"**Caption Cost:** `${img.caption_cost:.4f}`")
                                    if img.docling_caption:
                                        st.markdown(f"**Docling Caption:** {img.docling_caption[:100]}...")
                                    if img.surrounding_context:
                                        st.markdown(f"**Context:** {img.surrounding_context[:100]}...")
                                    if img.image_metadata:
                                        st.markdown("**Image Info:**")
                                        st.json(img.image_metadata)
                                    if img.bbox:
                                        st.markdown(f"**BBox:** `{img.bbox}`")

                            st.divider()

                        if len(preview_result.preview_images) > 5:
                            st.info(f"Showing 5 of {len(preview_result.preview_images)} images")

    def _handle_save_preview_data(self) -> None:
        """Save preview data to Qdrant via the save API.

        This implements the second step of the two-step upload workflow.
        """
        preview_data: List[PreviewResult] = self.session_manager.get("upload_preview_data", [])

        if not preview_data:
            st.warning("No preview data to save")
            return

        try:
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                st.subheader("💾 Saving to Vector Database", divider="gray")
                progress_bar = st.progress(0)
                status_text = st.empty()

            api_client = get_api_client()
            total_chunks_saved = 0
            total_images_saved = 0
            saved_files = 0
            failed_files = 0
            text_collection = ""
            image_collection = ""

            for i, preview_result in enumerate(preview_data):
                file_progress = (i + 1) / len(preview_data)
                progress_bar.progress(file_progress)
                status_text.markdown(f"**Saving {preview_result.file_name}...**")

                # Call save API
                save_result = api_client.save_upload(
                    file_name=preview_result.file_name,
                    file_type=preview_result.file_type,
                    language=preview_result.language,
                    chunks_data=preview_result.full_chunks_data,
                    images_data=preview_result.full_images_data,
                )

                if save_result.success:
                    saved_files += 1
                    total_chunks_saved += save_result.chunks_count
                    total_images_saved += save_result.images_count
                    text_collection = save_result.text_collection
                    image_collection = save_result.image_collection
                else:
                    failed_files += 1
                    st.error(f"❌ Failed to save {preview_result.file_name}: {save_result.error}")

            # Complete progress
            progress_bar.progress(1.0)
            status_text.markdown("**Save Complete!**")

            time.sleep(1)
            progress_container.empty()

            # Clear preview data
            self.session_manager.set("upload_preview_data", None)
            self.session_manager.set("data_saved_success", True)

            # Show success message
            if saved_files > 0:
                st.success(
                    f"🎉 Successfully saved {saved_files} file(s)!\n\n"
                    f"- **{total_chunks_saved}** chunks → `{text_collection}`\n"
                    f"- **{total_images_saved}** images → `{image_collection}`"
                )

            if failed_files > 0:
                st.warning(f"⚠️ {failed_files} file(s) failed to save")

        except Exception as e:
            st.error(f"❌ Save Error: {str(e)}")
            logger.error(f"Save error: {e}", exc_info=True)

    def _process_uploaded_files_via_api(
        self, uploaded_files: List
    ) -> None:
        """Process uploaded files via API with progress tracking.

        Args:
            uploaded_files: List of uploaded file objects
        """
        try:
            # Initialize progress tracking
            progress_container = st.container()
            with progress_container:
                st.subheader("🔄 Processing & Uploading Files", divider="gray")
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_text = st.empty()

                # Metrics display
                col1, col2, col3 = st.columns(3)
                with col1:
                    chunks_metric = st.empty()
                with col2:
                    images_metric = st.empty()
                with col3:
                    time_metric = st.empty()

            # Track results
            results = []
            total_chunks = 0
            total_images = 0
            processed_files = 0
            failed_files = 0
            start_time = time.time()

            for i, uploaded_file in enumerate(uploaded_files):
                file_progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(file_progress)
                status_text.markdown(
                    f"**Processing file {i+1}/{len(uploaded_files)}**: `{uploaded_file.name}`"
                )

                # Process via API
                result = self._process_file_via_api(
                    uploaded_file=uploaded_file,
                    status_text=details_text,
                )

                results.append(result)

                if result.success:
                    processed_files += 1
                    total_chunks += result.chunks_count
                    total_images += result.images_count
                    details_text.success(
                        f"✅ {result.file_name}: {result.chunks_count} chunks, "
                        f"{result.images_count} images ({result.processing_time:.1f}s)"
                    )
                else:
                    failed_files += 1
                    details_text.error(f"❌ {result.file_name}: {result.error}")

                # Update metrics
                chunks_metric.metric("Total Chunks", total_chunks)
                images_metric.metric("Total Images", total_images)
                elapsed = round(time.time() - start_time, 1)
                time_metric.metric("Elapsed Time", f"{elapsed}s")

            # Complete progress
            progress_bar.progress(1.0)
            total_time = round(time.time() - start_time, 2)
            status_text.markdown("**Processing Complete!**")

            # Brief pause then clear progress
            time.sleep(1)
            progress_container.empty()

            # Display final results
            self._display_api_upload_results(
                results=results,
                total_chunks=total_chunks,
                total_images=total_images,
                processed_files=processed_files,
                failed_files=failed_files,
                total_time=total_time,
            )

        except Exception as e:
            st.error(f"❌ Upload Error: {str(e)}")
            logger.error(f"API upload error: {e}", exc_info=True)

    def _display_api_upload_results(
        self,
        results: List,
        total_chunks: int,
        total_images: int,
        processed_files: int,
        failed_files: int,
        total_time: float,
    ) -> None:
        """Display results from API-based file upload.

        Args:
            results: List of UploadResult objects
            total_chunks: Total chunks uploaded
            total_images: Total images uploaded
            processed_files: Number of successfully processed files
            failed_files: Number of failed files
            total_time: Total processing time in seconds
        """
        if processed_files > 0:
            st.success(
                f"🎉 Successfully uploaded {processed_files} file(s)!"
            )

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Files Processed", f"{processed_files}/{len(results)}")
            with col2:
                st.metric("Total Chunks", total_chunks)
            with col3:
                st.metric("Total Images", total_images)
            with col4:
                st.metric("Processing Time", f"{total_time}s")

            # Mark as saved (API already saved to Qdrant)
            self.session_manager.set("data_saved_success", True)

        if failed_files > 0:
            st.warning(f"⚠️ {failed_files} file(s) failed to process")

            # Show failed file details
            with st.expander("Failed Files Details", expanded=True):
                for result in results:
                    if not result.success:
                        st.error(f"**{result.file_name}**: {result.error}")

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

                # Add cost tracking metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    cost_metric = st.empty()
                with col2:
                    image_metric = st.empty()
                with col3:
                    chunk_metric = st.empty()

            all_chunks = []
            processing_stats = {
                "total_files": len(uploaded_files),
                "processed_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "total_images": 0,
                "total_cost": 0.0,
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
                        chunks, pdf_metadata = self._process_pdf_file(
                            uploaded_file, details_text
                        )
                        if chunks:
                            all_chunks.extend(chunks)
                            processing_stats["processed_files"] += 1
                            processing_stats["total_images"] += pdf_metadata.get(
                                "image_count", 0
                            )
                            processing_stats["total_cost"] += pdf_metadata.get(
                                "caption_cost", 0.0
                            )

                            # Store image data for later upload to Qdrant
                            if "all_image_data" not in processing_stats:
                                processing_stats["all_image_data"] = []
                            if pdf_metadata.get("image_data"):
                                processing_stats["all_image_data"].extend(
                                    pdf_metadata["image_data"]
                                )

                            # Update metrics in real-time
                            cost_metric.metric(
                                "Caption Cost",
                                f"${processing_stats['total_cost']:.4f}",
                                help="Total GPT-4o Mini Vision API cost",
                            )
                            image_metric.metric(
                                "Images",
                                processing_stats["total_images"],
                                help="Total images captioned",
                            )
                            chunk_metric.metric(
                                "Chunks", len(all_chunks), help="Total chunks created"
                            )

                            details_text.success(
                                f"✅ PDF file processed: {len(chunks)} chunks created"
                            )

                    elif file_extension == "docx":
                        chunks, docx_metadata = self._process_docx_file(
                            uploaded_file, details_text
                        )
                        if chunks:
                            all_chunks.extend(chunks)
                            processing_stats["processed_files"] += 1
                            processing_stats["total_images"] += docx_metadata.get(
                                "image_count", 0
                            )
                            processing_stats["total_cost"] += docx_metadata.get(
                                "caption_cost", 0.0
                            )

                            # Store image data for later upload to Qdrant
                            if "all_image_data" not in processing_stats:
                                processing_stats["all_image_data"] = []
                            if docx_metadata.get("image_data"):
                                processing_stats["all_image_data"].extend(
                                    docx_metadata["image_data"]
                                )

                            # Update metrics in real-time
                            cost_metric.metric(
                                "Caption Cost",
                                f"${processing_stats['total_cost']:.4f}",
                                help="Total GPT-4o Mini Vision API cost",
                            )
                            image_metric.metric(
                                "Images",
                                processing_stats["total_images"],
                                help="Total images captioned",
                            )
                            chunk_metric.metric(
                                "Chunks", len(all_chunks), help="Total chunks created"
                            )

                            details_text.success(
                                f"✅ DOCX file processed: {len(chunks)} chunks created"
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

    def _process_pdf_file(self, uploaded_file, status_text) -> tuple:
        """
        Process a single PDF file using the session-managed document processor with enhanced progress tracking.

        Returns:
            Tuple of (chunks, metadata_dict) where metadata_dict contains:
                - image_count: number of images processed
                - caption_cost: total caption cost
                - total_chunks: number of chunks created
        """
        try:
            status_text.info("📄 Step 1/5: Initializing PDF processor...")

            # Check if session manager has PDF processor available
            if not self.session_manager.is_pdf_processor_available():
                status_text.warning(
                    "⚠️ PDF processor not available. Using basic PDF processing."
                )
                return self._fallback_pdf_processing(uploaded_file, status_text), {}

            status_text.info("📄 Step 2/5: Parsing PDF structure...")

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file_path = temp_file.name  # Assign path first for cleanup
                temp_file.write(uploaded_file.read())

            try:
                status_text.info("🔍 Step 3/5: Extracting text and images from PDF...")

                # Process the PDF using session-managed processor
                result = self.session_manager.process_document_with_session(
                    temp_file_path,
                    languages=[self.session_manager.get("language", "en")],
                    original_filename=uploaded_file.name,
                    caption_failure_mode=self.session_manager.get(
                        "caption_failure_mode", "graceful"
                    ),
                )

                if result and result.success:
                    # Check if images were extracted and captioned
                    image_count = result.metadata.get('total_images', 0)
                    caption_cost = result.metadata.get('caption_total_cost', 0.0)

                    if image_count > 0:
                        status_text.info(
                            f"🖼️ Step 4/5: Captioned {image_count} images "
                            f"(Cost: ${caption_cost:.4f})"
                        )
                    else:
                        status_text.info("📝 Step 4/5: No images found in PDF")

                    status_text.info(
                        f"✅ Step 5/5: Creating {len(result.elements)} text chunks..."
                    )

                    # Convert processing result to chunks
                    chunks = []
                    for i, element in enumerate(result.elements):
                        # Preserve existing metadata from DoclingChunker
                        elem_metadata = (
                            element.metadata.copy()
                            if hasattr(element, 'metadata') and isinstance(element.metadata, dict)
                            else {}
                        )

                        # Add/override with process-level metrics
                        elem_metadata.update({
                            "processing_strategy": (
                                result.metrics.strategy_used
                                if hasattr(result, 'metrics') and result.metrics
                                else elem_metadata.get("source", "docling")
                            ),
                            "ocr_used": (
                                result.metrics.ocr_used
                                if hasattr(result, 'metrics') and result.metrics
                                else False
                            ),
                            "total_pages": (
                                result.metrics.pages_processed
                                if hasattr(result, 'metrics') and result.metrics
                                else 0
                            ),
                        })

                        chunk = {
                            "chunk": element.text,
                            "source_file": uploaded_file.name,
                            "file_type": "PDF",
                            "chunk_index": i,
                            "doc_id": str(uuid.uuid4()),
                            "metadata": elem_metadata,
                        }
                        chunks.append(chunk)

                    # Display final summary
                    total_pages = (
                        result.metrics.pages_processed
                        if hasattr(result, 'metrics') and result.metrics
                        else 0
                    )
                    status_text.success(
                        f"✅ PDF processing complete! Created {len(chunks)} chunks "
                        f"from {total_pages} pages. "
                        f"Images: {image_count}, Caption cost: ${caption_cost:.4f}"
                    )

                    # Return chunks, metadata, and image data for Qdrant upload
                    metadata = {
                        "image_count": image_count,
                        "caption_cost": caption_cost,
                        "total_chunks": len(chunks),
                        "image_data": (
                            result.image_data if hasattr(result, 'image_data') else []
                        ),
                    }
                    return chunks, metadata

                else:
                    error_msg = result.error_message if result else "Unknown error"
                    status_text.error(f"❌ PDF processing failed: {error_msg}")
                    return [], {}

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            status_text.error(f"❌ PDF processing error: {str(e)}")
            logger.error(
                f"PDF processing error for {uploaded_file.name}: {e}", exc_info=True
            )
            return [], {}

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

    def _process_docx_file(self, uploaded_file, status_text) -> tuple:
        """Process a single DOCX file using the session-managed document processor.

        Uses same pipeline as PDF but without OCR options.

        Args:
            uploaded_file: Uploaded DOCX file object
            status_text: Streamlit status text element for updates

        Returns:
            Tuple of (chunks, metadata_dict) where metadata_dict contains:
                - image_count: number of images processed
                - caption_cost: total caption cost
                - total_chunks: number of chunks created
                - image_data: list of image data for Qdrant upload
        """
        try:
            status_text.info("📝 Step 1/4: Initializing DOCX processor...")

            # Check if session manager has document processor available
            if not self.session_manager.is_pdf_processor_available():
                status_text.error(
                    "❌ Document processor not available. Cannot process DOCX files."
                )
                return [], {}

            status_text.info("📝 Step 2/4: Parsing DOCX structure...")

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file_path = temp_file.name  # Assign path first for cleanup
                temp_file.write(uploaded_file.read())

            try:
                status_text.info("🔍 Step 3/4: Extracting text and images from DOCX...")

                # Process the DOCX using session-managed processor
                result = self.session_manager.process_document_with_session(
                    temp_file_path,
                    languages=[self.session_manager.get("language", "en")],
                    original_filename=uploaded_file.name,
                    caption_failure_mode=self.session_manager.get(
                        "caption_failure_mode", "graceful"
                    ),
                )

                if result and result.success:
                    # Check if images were extracted and captioned
                    image_count = result.metadata.get('total_images', 0)
                    caption_cost = result.metadata.get('caption_total_cost', 0.0)

                    if image_count > 0:
                        status_text.info(
                            f"🖼️ Captioned {image_count} images "
                            f"(Cost: ${caption_cost:.4f})"
                        )

                    status_text.info(
                        f"✅ Step 4/4: Creating {len(result.elements)} text chunks..."
                    )

                    # Convert processing result to chunks
                    chunks = []
                    for i, element in enumerate(result.elements):
                        # Preserve existing metadata from DoclingChunker
                        elem_metadata = (
                            element.metadata.copy()
                            if hasattr(element, 'metadata') and isinstance(element.metadata, dict)
                            else {}
                        )

                        # Add/override with process-level metrics
                        elem_metadata.update({
                            "processing_strategy": (
                                result.metrics.strategy_used
                                if hasattr(result, 'metrics') and result.metrics
                                else elem_metadata.get("source", "docling")
                            ),
                            "ocr_used": False,  # DOCX never uses OCR
                        })

                        chunk = {
                            "chunk": element.text,
                            "source_file": uploaded_file.name,
                            "file_type": "DOCX",
                            "chunk_index": i,
                            "doc_id": str(uuid.uuid4()),
                            "metadata": elem_metadata,
                        }
                        chunks.append(chunk)

                    status_text.success(
                        f"✅ DOCX processing complete! Created {len(chunks)} chunks. "
                        f"Images: {image_count}, Caption cost: ${caption_cost:.4f}"
                    )

                    # Return chunks, metadata, and image data for Qdrant upload
                    metadata = {
                        "image_count": image_count,
                        "caption_cost": caption_cost,
                        "total_chunks": len(chunks),
                        "image_data": (
                            result.image_data if hasattr(result, 'image_data') else []
                        ),
                    }
                    return chunks, metadata

                else:
                    error_msg = result.error_message if result else "Unknown error"
                    status_text.error(f"❌ DOCX processing failed: {error_msg}")
                    return [], {}

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            status_text.error(f"❌ DOCX processing error: {str(e)}")
            logger.error(
                f"DOCX processing error for {uploaded_file.name}: {e}", exc_info=True
            )
            return [], {}

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

        # Store image data for Qdrant upload
        if stats.get("all_image_data"):
            self.session_manager.set("pending_image_data", stats["all_image_data"])
            logger.info(f"Stored {len(stats['all_image_data'])} images for upload")

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

            details_text.success(
                f"✅ Uploaded {total_chunks} text chunks to '{qdrant_manager.collection_name}'"
            )

            # Step 5.5: Upload images to image collection (if any)
            pending_images = self.session_manager.get("pending_image_data", [])
            images_uploaded = 0

            if pending_images:
                try:
                    details_text.info(
                        f"🖼️ Uploading {len(pending_images)} image captions..."
                    )

                    # Use DocumentProcessor's upload_to_qdrant method
                    from backend.strategies.results import ProcessingResult

                    # Create a minimal ProcessingResult with image data
                    result = ProcessingResult(
                        success=True,
                        elements=[],  # No text elements for image-only upload
                        image_data=pending_images,
                    )

                    # Get document processor from session
                    doc_processor = self.session_manager.get('document_processor')
                    if not doc_processor:
                        doc_processor = (
                            self.session_manager.initialize_document_processor()
                        )

                    if doc_processor:
                        # Upload images using document processor
                        upload_result = doc_processor.upload_to_qdrant(
                            processing_result=result,
                            embeddings=[],  # No text embeddings
                            source_file=source_description,
                        )

                        images_uploaded = upload_result.get("images", 0)

                        if images_uploaded > 0:
                            details_text.success(
                                f"✅ Uploaded {images_uploaded} image captions to 'rag_chatbot_images'"
                            )
                        # Clear pending images
                        self.session_manager.set("pending_image_data", [])
                    else:
                        logger.warning(
                            "Document processor not available for image upload"
                        )

                except Exception as img_error:
                    logger.error(f"Failed to upload images: {img_error}")
                    details_text.warning(
                        f"⚠️ Text chunks uploaded successfully, but image upload failed: {str(img_error)}"
                    )

            # Complete progress
            progress_bar.progress(1.0)
            status_text.markdown("**Save Complete!**")

            if images_uploaded > 0:
                details_text.success(
                    f"✅ All data saved: {total_chunks} text chunks + {images_uploaded} images"
                )
            else:
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
