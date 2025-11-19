from typing import Any, Callable

import streamlit as st


def list_collection(
    session_state: Any,
    load_callback: Callable[[str], None],
    delete_callback: Callable[[str], None],
) -> None:
    """
    Display list of collections with load/delete actions.

    Args:
        session_state: Streamlit session state
        load_callback: Callback function to load collection
        delete_callback: Callback function to delete collection
    """
    st.info("🚧 Collection management requires Qdrant integration")
    st.markdown(
        """
    **Features coming soon:**
    - 📋 List all saved collections
    - 📥 Load existing collections
    - 🗑️ Delete collections
    - 📊 View collection statistics
    """
    )


def render_collection_dialog(
    session_state: Any, on_select: Callable[[str], None]
) -> None:
    """
    Render collection selection dialog.

    Args:
        session_state: Streamlit session state
        on_select: Callback when collection is selected
    """
    st.info("Collection selection dialog - coming soon")


def export_collection_ui(collection_name: str) -> None:
    """
    Render collection export UI.

    Args:
        collection_name: Name of collection to export
    """
    st.subheader("Export Collection")

    export_format = st.selectbox(
        "Export Format:",
        ["JSON", "CSV", "Parquet"],
        help="Choose format for exported data",
    )

    st.info(f"Export functionality for `{collection_name}` - coming soon")

    if st.button("Export", type="primary", disabled=True):
        st.warning("⏳ Export feature requires backend implementation")
