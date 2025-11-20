"""
Data Management UI - Placeholder for future data management functionality.

This page will provide tools for managing uploaded documents, vector collections,
and data operations within the RAG system.
"""

import streamlit as st

from backend.session_manager import SessionManager


class DataManagementUI:
    """UI component for data management operations."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def render(self, header_number: int) -> None:
        """Render data management interface."""
        st.subheader(f"{header_number}. Data Management", divider=True)
