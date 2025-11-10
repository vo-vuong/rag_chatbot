import uuid
from typing import List, Optional

import pandas as pd
import streamlit as st

from backend.session_manager import SessionManager


class DataUploadUI:
    """UI component for data upload and chunking."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self, header_number: int) -> None:
        st.header(f"{header_number}. Setup Data Source")

        # Upload section
        st.subheader(f"{header_number}.1. Upload Data", divider=True)
        self._render_upload_section()

        # Load from database section (placeholder)
        st.subheader(f"{header_number}.2. Or Load from Saved Collection", divider=True)
        st.info("TODO: Implement Qdrant db backend")

        # Save data button
        if (
            self.session_manager.get("chunks_df") is not None
            and not self.session_manager.get("chunks_df").empty
        ):
            if st.button("Save Data", type="primary"):
                self._handle_save_data()

    def _render_upload_section(self) -> None:
        uploaded_files = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload one or more files to process",
        )

        if uploaded_files:
            self._process_uploaded_files(uploaded_files)

    def _process_uploaded_files(self, uploaded_files: List) -> None:
        """
        Process uploaded files and display data.

        Args:
            uploaded_files: List of uploaded file objects
        """
        try:
            all_data = []

            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                # Simple CSV processing for now
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    all_data.append(df)
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                else:
                    st.warning(
                        f"⚠️ {file_extension.upper()} support coming soon. Skipping {uploaded_file.name}"
                    )

            if all_data:
                # Combine all dataframes
                combined_df = pd.concat(all_data, ignore_index=True)

                # Generate document IDs
                doc_ids = [str(uuid.uuid4()) for _ in range(len(combined_df))]
                combined_df['doc_id'] = doc_ids

                # Store in session
                self.session_manager.set("doc_ids", doc_ids)

                # Display data
                st.success(
                    f"📊 Loaded {len(combined_df)} rows from {len(all_data)} file(s)"
                )
                st.dataframe(combined_df.head(10), use_container_width=True)

                if len(combined_df) > 10:
                    st.caption(f"Showing first 10 rows. Total: {len(combined_df)} rows")

                # Chunking section
                self._render_chunking_section(combined_df)
            else:
                st.warning("No valid files were loaded. Please upload CSV files.")

        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")

    def _render_chunking_section(self, df: pd.DataFrame) -> None:
        """
        Render chunking configuration section.

        Args:
            df: DataFrame to chunk
        """
        st.subheader("📑 Chunking Configuration")

        if df.empty:
            st.warning("DataFrame is empty")
            return

        # Select column for indexing
        index_column = st.selectbox(
            "Choose the column to index (for vector search):",
            df.columns.tolist(),
            help="Select the text column to create embeddings from",
        )

        st.info(f"Selected column: **{index_column}**")

        # Chunking options (simplified)
        chunk_option = st.radio(
            "Select chunking strategy:",
            ["No Chunking", "Simple Split (by sentences)"],
            help="Choose how to split your documents",
            key="chunkOption",
        )

        # Process chunks button
        if st.button("Process Chunks", type="primary"):
            chunks_df = self._process_chunks(df, index_column, chunk_option)

            if chunks_df is not None and not chunks_df.empty:
                st.success(f"✅ Created {len(chunks_df)} chunks!")
                st.dataframe(chunks_df.head(10), use_container_width=True)
                self.session_manager.set("chunks_df", chunks_df)

    def _process_chunks(
        self, df: pd.DataFrame, index_column: str, chunk_option: str
    ) -> Optional[pd.DataFrame]:
        """
        Process dataframe into chunks.

        Args:
            df: Source dataframe
            index_column: Column to chunk
            chunk_option: Chunking strategy

        Returns:
            DataFrame with chunks or None
        """
        try:
            chunk_records = []
            progress_bar = st.progress(0, text="Processing chunks...")
            total = len(df)

            for idx, (row_idx, row) in enumerate(df.iterrows()):
                selected_value = row[index_column]

                # Skip invalid values
                if not isinstance(selected_value, str) or len(selected_value) == 0:
                    continue

                # Simple chunking logic
                if chunk_option == "No Chunking":
                    chunks = [selected_value]
                else:
                    # Simple sentence split
                    import re

                    sentences = re.split(r'[.!?]+', selected_value)
                    chunks = [s.strip() for s in sentences if s.strip()]

                # Create chunk records - FIX: exclude index_column instead of 'chunk'
                for chunk in chunks:
                    chunk_record = {
                        'chunk': chunk,
                        **{k: v for k, v in row.to_dict().items() if k != index_column},
                    }
                    chunk_records.append(chunk_record)

                # Update progress
                progress_bar.progress(
                    (idx + 1) / total, text=f"Processing {idx + 1}/{total}"
                )

            progress_bar.empty()

            if not chunk_records:
                st.warning("No valid chunks created. Check your data.")
                return None

            return pd.DataFrame(chunk_records)

        except Exception as e:
            st.error(f"Error processing chunks: {str(e)}")
            import traceback

            st.error(traceback.format_exc())  # Thêm để debug
            return None

    def _handle_save_data(self) -> None:
        """Handle save data button click."""
        chunks_df = self.session_manager.get("chunks_df")

        if chunks_df is None or chunks_df.empty:
            st.warning("No data to save. Please upload and process data first.")
            return

        try:
            # Generate collection name
            collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"

            # For now, just mark as saved (vector DB integration comes later)
            self.session_manager.update(
                {
                    "collection_name": collection_name,
                    "data_saved_success": True,
                    "source_data": "UPLOAD",
                }
            )

            st.success("✅ Data prepared successfully!")
            st.info(f"📦 Collection name: `{collection_name}`")
            st.info("TODO: Implement Qdrant db backend")

        except Exception as e:
            st.error(f"❌ Error saving data: {str(e)}")
