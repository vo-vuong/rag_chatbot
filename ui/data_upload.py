"""
Data Upload UI - Kept language selection in upload flow.

Language is selected per-upload and stored in metadata.
"""

import re
import traceback
import uuid
from typing import List, Optional

import pandas as pd
import streamlit as st

from backend.session_manager import SessionManager
from backend.vector_db.qdrant_manager import QdrantManager
from config.constants import EN, ENGLISH, NONE, VI, VIETNAMESE


class DataUploadUI:
    """UI component for data upload with language selection."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

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
            [NONE, ENGLISH, VIETNAMESE],
            index=0,
            horizontal=True,
            help="Select the primary language of your documents",
        )

        if language_choice != NONE:
            lang_code = EN if language_choice == ENGLISH else VI
            self.session_manager.set("language", lang_code)
            st.success(f"✅ Language: **{language_choice}** (`{lang_code}`)")
        else:
            self.session_manager.set("language", None)

    def _check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        if not self.session_manager.is_embedding_configured():
            st.warning(
                "⚠️ Embeddings not configured. " "Please enter API key in sidebar first."
            )
            return False

        language = self.session_manager.get("language")
        if not language:
            st.warning("⚠️ Please select a document language first.")
            return False

        return True

    def _render_upload_section(self) -> None:
        """Render file upload section."""
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload one or more CSV files",
        )

        if uploaded_files:
            self._process_uploaded_files(uploaded_files)

    def _process_uploaded_files(self, uploaded_files: List) -> None:
        """Process uploaded files."""
        try:
            all_data = []

            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    all_data.append(df)
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ Skipping unsupported file: {uploaded_file.name}")

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)

                doc_ids = [str(uuid.uuid4()) for _ in range(len(combined_df))]
                combined_df['doc_id'] = doc_ids

                self.session_manager.set("doc_ids", doc_ids)

                st.success(
                    f"📊 Loaded {len(combined_df)} rows from {len(all_data)} file(s)"
                )
                st.dataframe(combined_df.head(10), use_container_width=True)

                if len(combined_df) > 10:
                    st.caption(f"Showing first 10 rows. Total: {len(combined_df)}")

                self._render_chunking_section(combined_df)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.code(traceback.format_exc())

    def _render_chunking_section(self, df: pd.DataFrame) -> None:
        """Render chunking configuration."""
        st.subheader("📝 Chunking Configuration")

        if df.empty:
            st.warning("DataFrame is empty")
            return

        index_column = st.selectbox(
            "Choose column to index:",
            df.columns.tolist(),
            help="Select the text column for vector search",
        )

        st.info(f"Selected: **{index_column}**")

        chunk_option = st.radio(
            "Chunking strategy:",
            ["No Chunking", "Simple Split (by sentences)"],
            help="How to split documents",
            key="chunkOption",
        )

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
            progress_bar = st.progress(0, text="Processing...")
            total = len(df)

            for idx, (_, row) in enumerate(df.iterrows()):
                selected_value = row[index_column]

                if not isinstance(selected_value, str) or not selected_value:
                    continue

                if chunk_option == "No Chunking":
                    chunks = [selected_value]
                else:
                    sentences = re.split(r'[.!?]+', selected_value)
                    chunks = [s.strip() for s in sentences if s.strip()]

                for chunk in chunks:
                    chunk_record = {
                        'chunk': chunk,
                        **{k: v for k, v in row.to_dict().items() if k != index_column},
                    }
                    chunk_records.append(chunk_record)

                progress_bar.progress((idx + 1) / total)

            progress_bar.empty()

            if not chunk_records:
                st.warning("No valid chunks created")
                return None

            return pd.DataFrame(chunk_records)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())
            return None

    def _handle_save_data(self) -> None:
        """Save data to Qdrant."""
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
            with st.spinner("Saving..."):
                # Step 1: Qdrant
                st.info("🔄 Step 1/4: Connecting to Qdrant...")
                qdrant_manager = QdrantManager()

                if not qdrant_manager.is_healthy():
                    st.error(
                        "❌ Qdrant not running. Start with:\n"
                        "```bash\ndocker-compose up -d\n```"
                    )
                    return

                # Step 2: Collection
                st.info("🔄 Step 2/4: Setting up collection...")
                dimension = self.session_manager.get("embedding_dimension")

                if not qdrant_manager.ensure_collection(dimension):
                    st.error("❌ Failed to create collection")
                    return

                # Step 3: Embeddings
                st.info("🔄 Step 3/4: Generating embeddings...")
                chunks = chunks_df['chunk'].tolist()

                try:
                    embeddings = embedding_strategy.embed_texts(chunks)
                except Exception as e:
                    st.error(f"❌ Embedding error: {str(e)}")
                    return

                # Step 4: Upload
                st.info("🔄 Step 4/4: Uploading to database...")

                success = qdrant_manager.add_documents(
                    chunks_df=chunks_df,
                    embeddings=embeddings,
                    language=language,
                    source_file="uploaded_csv",
                )

                if not success:
                    st.error("❌ Upload failed")
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

                st.success("✅ Data saved successfully!")

                # Stats
                stats = qdrant_manager.get_statistics()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Documents", stats.get("total_documents", 0))
                with col2:
                    st.metric("Collection", qdrant_manager.collection_name)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.code(traceback.format_exc())
