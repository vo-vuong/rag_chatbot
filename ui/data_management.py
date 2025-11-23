"""
Data Management UI - Collection Management and Data Operations.

This page provides tools for managing vector collections, documents,
and data operations within the RAG system.
"""

import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from backend.session_manager import SessionManager
from backend.vector_db.qdrant_manager import QdrantManager


class PaginationManager:
    """
    Centralized pagination state management for collection data tables.

    Handles both normal pagination and search mode pagination with collection-specific
    session state keys to ensure proper isolation between different collections.
    """

    PAGE_SIZE_OPTIONS = [10, 25, 50, 100]
    DEFAULT_PAGE_SIZE = 25
    SEARCH_LIMIT = 1000
    MEMORY_PAGINATION_THRESHOLD = 1000

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._initialize_state_keys()

    def _initialize_state_keys(self) -> None:
        """Initialize collection-specific session state keys."""
        self.page_key = f'data_page_{self.collection_name}'
        self.search_page_key = f'search_page_{self.collection_name}'
        self.offset_key = f'data_offset_{self.collection_name}'
        self.search_active_key = f'search_active_{self.collection_name}'
        self.cache_key = f'_all_points_cache_{self.collection_name}'

    def initialize_session_state(self) -> None:
        """Initialize all required session state variables if not present."""
        if self.page_key not in st.session_state:
            st.session_state[self.page_key] = 0
        if self.search_page_key not in st.session_state:
            st.session_state[self.search_page_key] = 0
        if self.offset_key not in st.session_state:
            st.session_state[self.offset_key] = None
        if self.search_active_key not in st.session_state:
            st.session_state[self.search_active_key] = False
        if self.cache_key not in st.session_state:
            st.session_state[self.cache_key] = []

    def get_current_page(self, is_search_mode: bool = False) -> int:
        """Get current page number for normal or search mode."""
        return (
            st.session_state[self.search_page_key]
            if is_search_mode
            else st.session_state[self.page_key]
        )

    def set_page(self, page: int, is_search_mode: bool = False) -> None:
        """Set current page number for normal or search mode."""
        key = self.search_page_key if is_search_mode else self.page_key
        st.session_state[key] = page

    def is_search_active(self) -> bool:
        """Check if search mode is currently active."""
        return st.session_state.get(self.search_active_key, False)

    def set_search_active(self, active: bool) -> None:
        """Set search mode active state."""
        st.session_state[self.search_active_key] = active

    def reset_search(self) -> None:
        """Reset search mode and return to normal pagination."""
        self.set_search_active(False)
        st.session_state[self.search_page_key] = 0

    def get_cached_points(self) -> List[Dict[str, Any]]:
        """Get cached points for the collection."""
        return st.session_state.get(self.cache_key, [])

    def set_cached_points(self, points: List[Dict[str, Any]]) -> None:
        """Cache points for the collection."""
        st.session_state[self.cache_key] = points


class CollectionsOverviewTab:
    """Component for rendering collections overview and creation functionality."""

    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant_manager = qdrant_manager

    def render(self) -> None:
        """Render the collections overview tab."""
        self._render_collection_list()

    def _render_collection_list(self) -> None:
        """Render list of all collections with management options."""
        st.write("### Vector Collections Overview")

        # Refresh button
        if st.button("🔄 Refresh Collections", key="refresh_collections"):
            st.rerun()

        try:
            collections = self.qdrant_manager.list_collections()

            if not collections:
                st.info(
                    "📭 No collections found. Create your first collection in the 'Create Collection' tab."
                )
                return

            self._display_collections_table(collections)
            self._display_summary_statistics(collections)

        except Exception as e:
            st.error(f"❌ Error loading collections: {str(e)}")

    def _display_collections_table(self, collections: List[Dict[str, Any]]) -> None:
        """Display collections in a structured table format."""
        for i, collection in enumerate(collections, 1):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                with col1:
                    st.write(f"**{collection['name']}**")
                    if collection.get('error'):
                        st.caption(f"⚠️ {collection['error']}")
                    else:
                        st.caption(f"Status: {collection['status']}")

                with col2:
                    if not collection.get('error'):
                        points_count = collection.get('points_count', 0)
                        st.metric(
                            "Points",
                            f"{points_count:,}" if points_count is not None else "N/A",
                        )

                with col3:
                    if not collection.get('error'):
                        indexed_vectors = collection.get('indexed_vectors_count', 0)
                        st.metric(
                            "Indexed Vectors",
                            (
                                f"{indexed_vectors:,}"
                                if indexed_vectors is not None
                                else "N/A"
                            ),
                        )

                with col4:
                    if (
                        not collection.get('error')
                        and collection['name'] != self.qdrant_manager.collection_name
                    ):
                        if st.button(
                            "🗑️",
                            key=f"delete_{collection['name']}",
                            help="Delete collection",
                        ):
                            st.session_state[f"confirm_delete_{collection['name']}"] = (
                                True
                            )
                            st.rerun()

                        # Show confirmation dialog if needed
                        if st.session_state.get(
                            f"confirm_delete_{collection['name']}", False
                        ):
                            self._show_delete_confirmation(collection['name'])

                st.divider()

    def _display_summary_statistics(self, collections: List[Dict[str, Any]]) -> None:
        """Display summary statistics across all collections."""
        total_points = sum(
            col.get('points_count', 0) or 0
            for col in collections
            if not col.get('error')
        )
        st.metric("📊 Total Points Across All Collections", f"{total_points:,}")

    def _show_delete_confirmation(self, collection_name: str) -> None:
        """Show deletion confirmation dialog."""
        with st.container():
            st.error(f"⚠️ **Delete Collection: {collection_name}**")
            st.warning(
                "This action cannot be undone. All vectors and metadata in this collection will be permanently deleted."
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🚫 Cancel", key=f"cancel_delete_{collection_name}"):
                    if f"confirm_delete_{collection_name}" in st.session_state:
                        del st.session_state[f"confirm_delete_{collection_name}"]
                    st.rerun()

            with col2:
                if st.button(
                    "🗑️ Delete Forever",
                    key=f"confirm_delete_btn_{collection_name}",
                    type="primary",
                ):
                    self._delete_collection(collection_name)

    def _delete_collection(self, collection_name: str) -> None:
        """Delete a collection after confirmation."""
        try:
            with st.spinner(f"Deleting collection '{collection_name}'..."):
                success = self.qdrant_manager.delete_collection_by_name(collection_name)

                if success:
                    st.success(
                        f"✅ Collection '{collection_name}' deleted successfully!"
                    )
                    # Clean up session state
                    if f"confirm_delete_{collection_name}" in st.session_state:
                        del st.session_state[f"confirm_delete_{collection_name}"]
                    # Auto-refresh after a short delay
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Failed to delete collection '{collection_name}'.")

        except Exception as e:
            st.error(f"❌ Error deleting collection: {str(e)}")


class CollectionCreationTab:
    """Component for rendering collection creation functionality."""

    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant_manager = qdrant_manager

    def render(self) -> None:
        """Render the collection creation tab."""
        st.write("### Create New Collection")
        self._render_creation_form()

    def _render_creation_form(self) -> None:
        """Render the collection creation form."""
        with st.form("create_collection_form"):
            col1, col2 = st.columns(2)

            with col1:
                collection_name = st.text_input(
                    "Collection Name *",
                    placeholder="e.g., my_documents",
                    help="Unique name for the new collection",
                )

            with col2:
                dimension = st.number_input(
                    "Vector Dimension *",
                    min_value=1,
                    max_value=10000,
                    value=384,
                    help="Size of the embedding vectors (e.g., 384 for sentence-transformers/all-MiniLM-L6-v2)",
                )

            distance_metric = st.selectbox(
                "Distance Metric",
                options=["Cosine", "Euclid", "Dot"],
                index=0,
                help="Method for calculating vector similarity",
            )

            submitted = st.form_submit_button("🚀 Create Collection", type="primary")

            if submitted:
                self._handle_collection_creation(
                    collection_name, dimension, distance_metric
                )

    def _handle_collection_creation(
        self, collection_name: str, dimension: int, distance: str
    ) -> None:
        """Handle the collection creation process."""
        if not collection_name.strip():
            st.error("❌ Collection name is required")
        elif dimension <= 0:
            st.error("❌ Vector dimension must be greater than 0")
        else:
            self._create_collection(collection_name.strip(), int(dimension), distance)

    def _create_collection(
        self, collection_name: str, dimension: int, distance: str
    ) -> None:
        """Create a new collection with user-specified parameters."""
        try:
            with st.spinner(f"Creating collection '{collection_name}'..."):
                success = self.qdrant_manager.create_collection(
                    collection_name, dimension, distance
                )

                if success:
                    st.success(
                        f"✅ Collection '{collection_name}' created successfully!"
                    )
                    # Auto-refresh after a short delay
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(
                        f"❌ Failed to create collection '{collection_name}'. It may already exist."
                    )

        except Exception as e:
            st.error(f"❌ Error creating collection: {str(e)}")


class CollectionDetailsTab:
    """Component for rendering collection details and data points functionality."""

    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant_manager = qdrant_manager

    def render(self) -> None:
        """Render the collection details tab."""
        st.write("### Collection Details")

        try:
            collections = self.qdrant_manager.list_collections()
            if not collections:
                st.info("📭 No collections available.")
                return

            collection_names = [
                col['name'] for col in collections if not col.get('error')
            ]
            if not collection_names:
                st.warning("⚠️ No accessible collections found.")
                return

            selected_collection = st.selectbox(
                "Select Collection",
                options=collection_names,
                index=0,
                help="Choose a collection to view detailed information",
            )

            if selected_collection:
                self._display_collection_details(selected_collection)

        except Exception as e:
            st.error(f"❌ Error loading collection details: {str(e)}")

    def _display_collection_details(self, collection_name: str) -> None:
        """Display detailed information about a collection."""
        try:
            details = self.qdrant_manager.get_detailed_collection_info(collection_name)

            if not details:
                st.error(
                    f"❌ Could not retrieve details for collection: {collection_name}"
                )
                return

            # Create tabs for different views
            overview_tab, data_tab = st.tabs(["📊 Overview", "📋 Data Points"])

            with overview_tab:
                self._display_collection_overview(details, collection_name)

            with data_tab:
                data_points_renderer = CollectionDataPointsRenderer(
                    self.qdrant_manager, collection_name
                )
                data_points_renderer.render()

        except Exception as e:
            st.error(f"❌ Error displaying collection details: {str(e)}")

    def _display_collection_overview(
        self, details: Dict[str, Any], collection_name: str
    ) -> None:
        """Display collection overview information."""
        # Main collection info
        col1, col2, col3 = st.columns(3)

        with col1:
            points_count = details.get('points_count', 0)
            st.metric(
                "📄 Points", f"{points_count:,}" if points_count is not None else "N/A"
            )

        with col2:
            vectors_count = details.get('vectors_count', 0)
            st.metric(
                "🔢 Vectors",
                f"{vectors_count:,}" if vectors_count is not None else "N/A",
            )

        with col3:
            segments_count = details.get('segments_count', 0)
            st.metric(
                "📊 Segments",
                f"{segments_count:,}" if segments_count is not None else "N/A",
            )

        # Additional metrics
        col1, col2 = st.columns(2)

        with col1:
            indexed_count = details.get('indexed_vectors_count', 0)
            st.metric(
                "🔢 Indexed Vectors",
                f"{indexed_count:,}" if indexed_count is not None else "N/A",
            )

        with col2:
            # Calculate percentage of indexed vectors if we have both counts
            points_count = details.get('points_count', 0)
            indexed_count = details.get('indexed_vectors_count', 0)
            if points_count and points_count > 0 and indexed_count is not None:
                percentage = (indexed_count / points_count) * 100
                st.metric("📊 Indexed %", f"{percentage:.1f}%")
            else:
                st.metric("📊 Indexed %", "N/A")

        # Vector configuration
        st.write("#### Vector Configuration")
        vector_config = details.get('vector_config', {})

        if vector_config:
            for config_name, config_info in vector_config.items():
                with st.expander(
                    f"{'Default' if config_name == 'default' else config_name} Vector"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Size", config_info.get('size', 'N/A'))
                    with col2:
                        st.metric("Distance", config_info.get('distance', 'N/A'))
        else:
            st.info("No vector configuration information available.")

        # Collection status
        st.write("#### Collection Status")
        status = details.get('status', 'unknown').lower()
        if 'green' in status:
            st.success("✅ Collection is healthy")
        elif 'yellow' in status:
            st.warning("⚠️ Collection has warnings")
        elif 'red' in status:
            st.error("❌ Collection has errors")
        else:
            st.info(f"ℹ️ Status: {details.get('status', 'unknown')}")


class CollectionDataPointsRenderer:
    """Component for rendering collection data points with pagination and search."""

    def __init__(self, qdrant_manager: QdrantManager, collection_name: str):
        self.qdrant_manager = qdrant_manager
        self.collection_name = collection_name
        self.pagination_manager = PaginationManager(collection_name)

    def render(self) -> None:
        """Render the data points table with pagination and search."""
        st.write("### Data Points")

        # Initialize pagination state
        self.pagination_manager.initialize_session_state()

        # Render pagination and filtering controls
        page_size, search_term = self._render_controls()

        # Handle search state
        is_search_mode = self._handle_search_state(search_term)

        try:
            # Get and display data points
            paginated_points, total_points = self._get_paginated_points(
                page_size, search_term, is_search_mode
            )

            if paginated_points:
                self._display_data_table(paginated_points)
                self._render_pagination_controls(
                    page_size, total_points, is_search_mode, paginated_points
                )
            else:
                st.info("📭 No data points found in this collection.")

        except Exception as e:
            st.error(f"❌ Error loading data points: {str(e)}")

        # Reset search if search term is cleared
        if not search_term and self.pagination_manager.is_search_active():
            self.pagination_manager.reset_search()
            st.rerun()

    def _render_controls(self) -> tuple[int, str]:
        """Render pagination and filtering controls."""
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            page_size = st.selectbox(
                "Page Size",
                options=PaginationManager.PAGE_SIZE_OPTIONS,
                index=1,
                help="Number of points to display per page",
            )

        with col2:
            search_term = st.text_input(
                "Search in Content",
                placeholder="Search text chunks...",
                help="Filter data points by content",
            )

        with col3:
            if st.button("🔍 Search", key=f"search_points_{self.collection_name}"):
                st.session_state[f'search_page_{self.collection_name}'] = 0
                self.pagination_manager.set_search_active(True)

        return page_size, search_term

    def _handle_search_state(self, search_term: str) -> bool:
        """Handle search mode state and return whether we're in search mode."""
        is_search_mode = self.pagination_manager.is_search_active() and bool(
            search_term
        )
        return is_search_mode

    def _get_paginated_points(
        self, page_size: int, search_term: str, is_search_mode: bool
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get paginated points based on current mode and search state."""
        # Get collection info once
        collection_info = self.qdrant_manager.get_detailed_collection_info(
            self.collection_name
        )
        total_collection_points = (
            collection_info.get('points_count', 0) if collection_info else 0
        )

        if is_search_mode:
            return self._get_search_results(page_size, search_term)
        else:
            return self._get_normal_pagination_points(
                page_size, total_collection_points
            )

    def _get_search_results(
        self, page_size: int, search_term: str
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get points filtered by search term."""
        # Get all points for searching
        points_data = self.qdrant_manager.get_collection_points(
            collection_name=self.collection_name,
            limit=PaginationManager.SEARCH_LIMIT,
            offset=0,
        )

        # Filter points based on search term
        all_points = points_data.get('points', [])
        if all_points:
            filtered_points = self._filter_points_by_search_term(
                all_points, search_term
            )

            # Paginate search results
            current_search_page = self.pagination_manager.get_current_page(
                is_search_mode=True
            )
            start_idx = current_search_page * page_size
            end_idx = start_idx + page_size
            paginated_points = filtered_points[start_idx:end_idx]
            total_points = len(filtered_points)

            # Display search results info
            st.info(f"🔍 Found {total_points} points matching '{search_term}'")
            if total_points > 0:
                display_start = start_idx + 1
                display_end = min(start_idx + len(paginated_points), total_points)
                st.info(
                    f"📋 Showing {display_start}-{display_end} of {total_points} matching points"
                )
            else:
                st.info("📋 No matching points found")

            return paginated_points, total_points
        else:
            st.info("📋 No matching points found")
            return [], 0

    def _filter_points_by_search_term(
        self, points: List[Dict[str, Any]], search_term: str
    ) -> List[Dict[str, Any]]:
        """Filter points based on search term."""
        filtered_points = []
        search_lower = search_term.lower()

        for point in points:
            payload = point.get('payload', {})
            chunk_content = payload.get('chunk', '')
            if search_lower in chunk_content.lower():
                filtered_points.append(point)

        return filtered_points

    def _get_normal_pagination_points(
        self, page_size: int, total_collection_points: int
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get points using normal pagination."""
        current_page = self.pagination_manager.get_current_page(is_search_mode=False)

        if total_collection_points <= PaginationManager.MEMORY_PAGINATION_THRESHOLD:
            # Use in-memory pagination for smaller collections
            return self._get_memory_paginated_points(
                page_size, current_page, total_collection_points
            )
        else:
            # Use offset-based pagination for larger collections
            return self._get_offset_paginated_points(
                page_size, current_page, total_collection_points
            )

    def _get_memory_paginated_points(
        self, page_size: int, current_page: int, total_collection_points: int
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get points using in-memory pagination (for smaller collections)."""
        # Get all points once if not cached
        cached_points = self.pagination_manager.get_cached_points()
        if not cached_points:
            all_data = self.qdrant_manager.get_collection_points(
                collection_name=self.collection_name,
                limit=PaginationManager.SEARCH_LIMIT,
                offset=0,
            )
            points = all_data.get('points', [])
            self.pagination_manager.set_cached_points(points)
        else:
            points = cached_points

        # Paginate in memory
        start_idx = current_page * page_size
        end_idx = start_idx + page_size
        paginated_points = points[start_idx:end_idx]

        # Display pagination info
        self._display_pagination_info(
            current_page, page_size, paginated_points, total_collection_points
        )

        return paginated_points, total_collection_points

    def _get_offset_paginated_points(
        self, page_size: int, current_page: int, total_collection_points: int
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get points using offset-based pagination (for larger collections)."""
        # For larger collections, use a simple offset-based approach
        points_data = self.qdrant_manager.get_collection_points(
            collection_name=self.collection_name,
            limit=page_size,
            offset=0,  # We'll handle pagination differently for large collections
        )

        paginated_points = points_data.get('points', [])

        # Display pagination info
        self._display_pagination_info(
            current_page, page_size, paginated_points, total_collection_points
        )

        return paginated_points, total_collection_points

    def _display_pagination_info(
        self,
        current_page: int,
        page_size: int,
        paginated_points: List[Dict[str, Any]],
        total_points: int,
    ) -> None:
        """Display pagination information."""
        if total_points > 0:
            current_start = current_page * page_size
            current_end = min(current_start + len(paginated_points), total_points)
            total_pages = (total_points + page_size - 1) // page_size
            display_page = current_page + 1

            display_start = current_start + 1 if current_start < total_points else 0
            display_end = current_end
            st.info(
                f"📋 Showing {display_start}-{display_end} of {total_points} points (Page {display_page} of {total_pages})"
            )
        else:
            st.info("📋 No points in collection")

    def _display_data_table(self, points: List[Dict[str, Any]]) -> None:
        """Display points in a formatted table."""
        # Prepare data for table
        table_data = []
        for point in points:
            payload = point.get('payload', {})

            # Extract common fields
            row_data = {
                "ID": self._truncate_id(point.get('id', '')),
                "Content": self._truncate_content(payload.get('chunk', '')),
                "Source": payload.get('source_file', 'N/A'),
                "Language": payload.get('language', 'N/A'),
                "Chunk Index": payload.get('chunk_index', 'N/A'),
                "Timestamp": payload.get('timestamp', 'N/A'),
            }

            # Add any additional fields from payload
            self._add_additional_payload_fields(row_data, payload)

            table_data.append(row_data)

        # Convert to DataFrame and display
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    def _truncate_id(self, point_id: str) -> str:
        """Truncate point ID for display."""
        return point_id[:8] + '...' if len(point_id) > 8 else point_id

    def _truncate_content(self, content: str) -> str:
        """Truncate content for display."""
        return content[:100] + '...' if len(content) > 100 else content

    def _add_additional_payload_fields(
        self, row_data: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        """Add additional payload fields to row data."""
        excluded_keys = {'chunk', 'source_file', 'language', 'chunk_index', 'timestamp'}

        for key, value in payload.items():
            if key not in excluded_keys and key not in row_data:
                if isinstance(value, (str, int, float, bool)):
                    str_value = str(value)
                    row_data[key] = (
                        str_value[:50] + '...' if len(str_value) > 50 else str_value
                    )

    def _render_pagination_controls(
        self,
        page_size: int,
        total_points: int,
        is_search_mode: bool,
        paginated_points: List[Dict[str, Any]],
    ) -> None:
        """Render pagination controls."""
        if is_search_mode:
            self._render_search_pagination(page_size, total_points)
        else:
            self._render_normal_pagination(page_size, total_points)

    def _render_search_pagination(self, page_size: int, total_points: int) -> None:
        """Render pagination controls for search results."""
        current_search_page = self.pagination_manager.get_current_page(
            is_search_mode=True
        )
        total_search_pages = (
            (total_points + page_size - 1) // page_size if total_points > 0 else 1
        )

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button(
                "⬅️ Previous",
                key=f"search_prev_{self.collection_name}_{current_search_page}",
                disabled=current_search_page <= 0,
            ):
                self.pagination_manager.set_page(
                    current_search_page - 1, is_search_mode=True
                )
                st.rerun()

        with col2:
            st.write(f"Page {current_search_page + 1} of {total_search_pages}")

        with col3:
            end_idx = (current_search_page + 1) * page_size
            if st.button(
                "Next ➡️",
                key=f"search_next_{self.collection_name}_{current_search_page}",
                disabled=end_idx >= total_points,
            ):
                self.pagination_manager.set_page(
                    current_search_page + 1, is_search_mode=True
                )
                st.rerun()

    def _render_normal_pagination(self, page_size: int, total_points: int) -> None:
        """Render pagination controls for normal mode."""
        current_page = self.pagination_manager.get_current_page(is_search_mode=False)
        total_pages = (
            (total_points + page_size - 1) // page_size if total_points > 0 else 1
        )

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button(
                "⬅️ Previous",
                key=f"data_prev_{self.collection_name}_{current_page}",
                disabled=current_page <= 0,
            ):
                self.pagination_manager.set_page(current_page - 1)
                st.rerun()

        with col2:
            st.write(f"Page {current_page + 1} of {total_pages}")

        with col3:
            is_last_page = (current_page + 1) >= total_pages
            if st.button(
                "Next ➡️",
                key=f"data_next_{self.collection_name}_{current_page}",
                disabled=is_last_page,
            ):
                self.pagination_manager.set_page(current_page + 1)
                st.rerun()


class DataManagementUI:
    """Main UI component for data management operations."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.qdrant_manager = QdrantManager()
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize sub-components."""
        self.overview_tab = CollectionsOverviewTab(self.qdrant_manager)
        self.creation_tab = CollectionCreationTab(self.qdrant_manager)
        self.details_tab = CollectionDetailsTab(self.qdrant_manager)

    def render(self, header_number: int) -> None:
        """
        Render data management interface with modular components.

        Maintains exact same UI behavior and session state keys as the original
        implementation for full backward compatibility.
        """
        st.subheader(f"{header_number}. Data Management", divider=True)

        # Check Qdrant connection
        if not self._check_qdrant_connection():
            st.error(
                "❌ Unable to connect to Qdrant vector database. Please ensure it's running."
            )
            return

        # Main tabs for different management functions - exact same structure as before
        tab1, tab2, tab3 = st.tabs(
            ["📚 Collections", "➕ Create Collection", "🔍 Collection Details"]
        )

        with tab1:
            self.overview_tab.render()

        with tab2:
            self.creation_tab.render()

        with tab3:
            self.details_tab.render()

    def _check_qdrant_connection(self) -> bool:
        """
        Check if Qdrant server is accessible.

        Returns:
            bool: True if Qdrant is healthy, False otherwise
        """
        try:
            return self.qdrant_manager.is_healthy()
        except Exception:
            return False
