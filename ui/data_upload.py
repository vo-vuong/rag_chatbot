import re
import traceback
import uuid
from typing import List, Optional

import pandas as pd
import streamlit as st

from backend.session_manager import SessionManager
from backend.vector_db.qdrant_manager import QdrantManager


class DataUploadUI:
    """UI component for data upload and vector database integration."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self, header_number: int) -> None:
        st.header(f"{header_number}. Setup Data Source")

        # Check prerequisites
        if not self._check_prerequisites():
            return

        # Upload section
        st.subheader(f"{header_number}.1. Upload Data", divider=True)
        self._render_upload_section()

        # Save data button (only if chunks are ready)
        if (
            self.session_manager.get("chunks_df") is not None
            and not self.session_manager.get("chunks_df").empty
            and not self.session_manager.get("data_saved_success")
        ):
            st.divider()
            if st.button("💾 Save to Vector Database", type="primary"):
                self._handle_save_data()

    def _check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        if not self.session_manager.is_embedding_configured():
            st.warning("⚠️ Please configure embeddings first (Step 1)")
            return False

        return True

    def _render_upload_section(self) -> None:
        """Render file upload section."""
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload one or more CSV files to process",
        )

        if uploaded_files:
            self._process_uploaded_files(uploaded_files)

    def _process_uploaded_files(self, uploaded_files: List) -> None:
        """Process uploaded files and display data."""
        try:
            all_data = []

            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    all_data.append(df)
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                else:
                    st.warning(
                        f"⚠️ {file_extension.upper()} support coming soon. "
                        f"Skipping {uploaded_file.name}"
                    )

            if all_data:
                # Combine dataframes
                combined_df = pd.concat(all_data, ignore_index=True)

                # Generate document IDs
                doc_ids = [str(uuid.uuid4()) for _ in range(len(combined_df))]
                combined_df['doc_id'] = doc_ids

                # Store in session
                self.session_manager.set("doc_ids", doc_ids)

                # Display data
                st.success(
                    f"📊 Loaded {len(combined_df)} rows from "
                    f"{len(all_data)} file(s)"
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
            st.code(traceback.format_exc())

    def _render_chunking_section(self, df: pd.DataFrame) -> None:
        """Render chunking configuration."""
        st.subheader("📝 Chunking Configuration")

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
        if st.button("🔄 Process Chunks", type="primary"):
            chunks_df = self._process_chunks(df, index_column, chunk_option)

            if chunks_df is not None and not chunks_df.empty:
                st.success(f"✅ Created {len(chunks_df)} chunks!")
                st.dataframe(chunks_df.head(10), use_container_width=True)
                self.session_manager.set("chunks_df", chunks_df)

    def _process_chunks(
        self, df: pd.DataFrame, index_column: str, chunk_option: str
    ) -> Optional[pd.DataFrame]:
        """Process dataframe into chunks."""
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
                    sentences = re.split(r'[.!?]+', selected_value)
                    chunks = [s.strip() for s in sentences if s.strip()]

                # Create chunk records
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
            st.code(traceback.format_exc())
            return None

    def _handle_save_data(self) -> None:
        """Handle save data to Qdrant vector database."""
        chunks_df = self.session_manager.get("chunks_df")

        if chunks_df is None or chunks_df.empty:
            st.warning("No data to save. Please upload and process data first.")
            return

        # Check embedding configuration
        embedding_strategy = self.session_manager.get("embedding_strategy")
        if not embedding_strategy:
            st.error(
                "❌ Embedding strategy not configured. Please setup embeddings first."
            )
            return

        try:
            with st.spinner("Saving to vector database..."):
                # Step 1: Initialize Qdrant
                st.info("🔄 Step 1/4: Connecting to Qdrant...")
                qdrant_manager = QdrantManager()

                if not qdrant_manager.is_healthy():
                    st.error(
                        "❌ Cannot connect to Qdrant. "
                        "Make sure Qdrant is running:\n"
                        "```bash\n"
                        "docker-compose up -d\n"
                        "```"
                    )
                    return

                # Step 2: Ensure collection exists
                st.info("🔄 Step 2/4: Setting up collection...")
                language = self.session_manager.get("language")
                dimension = self.session_manager.get("embedding_dimension")

                if not qdrant_manager.ensure_collection(dimension, language):
                    st.error("❌ Failed to create/verify collection")
                    return

                # Step 3: Generate embeddings
                st.info("🔄 Step 3/4: Generating embeddings...")
                chunks = chunks_df['chunk'].tolist()

                try:
                    embeddings = embedding_strategy.embed_texts(chunks)
                except Exception as e:
                    st.error(f"❌ Failed to generate embeddings: {str(e)}")
                    return

                # Step 4: Upload to Qdrant
                st.info("🔄 Step 4/4: Uploading to vector database...")

                success = qdrant_manager.add_documents(
                    chunks_df=chunks_df,
                    embeddings=embeddings,
                    language=language,
                    source_file="uploaded_csv",
                )

                if not success:
                    st.error("❌ Failed to upload documents")
                    return

                # Save to session
                self.session_manager.update(
                    {
                        "qdrant_manager": qdrant_manager,
                        "collection_name": qdrant_manager.collection_name,
                        "data_saved_success": True,
                        "source_data": "UPLOAD",
                    }
                )

                # Show success
                st.success("✅ Data saved successfully to vector database!")

                # Show statistics
                stats = qdrant_manager.get_statistics()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Documents", stats.get("total_documents", 0))
                with col2:
                    st.metric("Collection", qdrant_manager.collection_name)

        except Exception as e:
            st.error(f"❌ Error saving data: {str(e)}")

            st.code(traceback.format_exc())
