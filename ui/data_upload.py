"""
Data Upload UI - Enhanced with PDF processing support.

Supports both CSV and PDF file uploads with advanced processing strategies,
and comprehensive progress tracking.
"""

import logging
import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from backend.chunking.csv_grouping_chunker import CSVGroupingChunker
from backend.session_manager import SessionManager
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
from ui.api_client import PreviewResult, get_api_client

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
            if st.button(
                "💾 Save to Vector Database", type="primary", use_container_width=True
            ):
                self._handle_save_preview_data()

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
            help="Upload CSV, PDF, or DOCX files. PDFs support optional OCR.",
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

        st.info(f"**{PDF_PROCESSING_MODES[mode_choice]}**: {mode_info[mode_choice]}")

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
                    selected_cols = csv_configs[uploaded_file.name].get(
                        "selected_columns", []
                    )
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
        preview_data: List[PreviewResult] = self.session_manager.get(
            "upload_preview_data", []
        )

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
                        for idx, chunk in enumerate(
                            preview_result.preview_chunks[:display_count]
                        ):
                            st.markdown(f"**Chunk {chunk.chunk_index + 1}:**")
                            preview_text = (
                                chunk.text[:500] + "..."
                                if len(chunk.text) > 500
                                else chunk.text
                            )
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
                                    st.markdown(
                                        f"**Processing Strategy:** `{chunk.processing_strategy or 'N/A'}`"
                                    )
                                    st.markdown(f"**Chunk Type:** `{chunk.chunk_type}`")
                                    st.markdown(f"**Source:** `{chunk.source}`")
                                    st.markdown(f"**OCR Used:** `{chunk.ocr_used}`")
                                with meta_col2:
                                    st.markdown(
                                        f"**Token Count:** `{chunk.token_count or 'N/A'}`"
                                    )
                                    st.markdown(f"**File Type:** `{chunk.file_type}`")
                                    if chunk.headings:
                                        st.markdown(
                                            f"**Headings:** {', '.join(chunk.headings)}"
                                        )
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
                        st.markdown(
                            f"**{len(preview_result.preview_images)} image(s) extracted:**"
                        )
                        for img_idx, img in enumerate(
                            preview_result.preview_images[:5]
                        ):
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
                                st.markdown(
                                    f"**Caption:** {img.caption[:200]}..."
                                    if len(img.caption) > 200
                                    else f"**Caption:** {img.caption}"
                                )
                                st.caption(
                                    f"Page: {img.page_number or 'N/A'} | Hash: {img.image_hash[:8]}..."
                                )

                                # Image metadata toggle (avoid nested expander)
                                show_img_meta = st.toggle(
                                    "Show image metadata",
                                    key=f"img_meta_toggle_{preview_result.file_name}_{img_idx}",
                                )
                                if show_img_meta:
                                    st.markdown(
                                        f"**Processing Strategy:** `{img.processing_strategy}`"
                                    )
                                    st.markdown(f"**File Type:** `{img.file_type}`")
                                    st.markdown(f"**Language:** `{img.language}`")
                                    st.markdown(
                                        f"**Caption Cost:** `${img.caption_cost:.4f}`"
                                    )
                                    if img.docling_caption:
                                        st.markdown(
                                            f"**Docling Caption:** {img.docling_caption[:100]}..."
                                        )
                                    if img.surrounding_context:
                                        st.markdown(
                                            f"**Context:** {img.surrounding_context[:100]}..."
                                        )
                                    if img.image_metadata:
                                        st.markdown("**Image Info:**")
                                        st.json(img.image_metadata)
                                    if img.bbox:
                                        st.markdown(f"**BBox:** `{img.bbox}`")

                            st.divider()

                        if len(preview_result.preview_images) > 5:
                            st.info(
                                f"Showing 5 of {len(preview_result.preview_images)} images"
                            )

    def _handle_save_preview_data(self) -> None:
        """Save preview data to Qdrant via the save API.

        This implements the second step of the two-step upload workflow.
        """
        preview_data: List[PreviewResult] = self.session_manager.get(
            "upload_preview_data", []
        )

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
                    st.error(
                        f"❌ Failed to save {preview_result.file_name}: {save_result.error}"
                    )

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
