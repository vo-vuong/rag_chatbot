"""
Session Service - In-memory session management for API layer.

Extracted from ChatMainUI to enable API-based session handling.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class SessionData:
    """Session data container for API layer."""

    chat_history: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)


class SessionService:
    """
    In-memory session store for API layer.

    Provides session management independent of Streamlit's session_state.
    """

    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}

    def get_session(self, session_id: str) -> SessionData:
        """Get or create session by ID."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionData()
        return self._sessions[session_id]

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session."""
        return self.get_session(session_id).chat_history

    def add_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Add message to session chat history."""
        session = self.get_session(session_id)
        session.chat_history.append({"role": role, "content": content})
        session.last_activity = datetime.utcnow()

    def create_session(self) -> str:
        """Create new session and return its ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionData()
        return session_id

    def clear_session(self, session_id: str) -> None:
        """Clear chat history for a session."""
        if session_id in self._sessions:
            self._sessions[session_id].chat_history = []
            self._sessions[session_id].last_activity = datetime.utcnow()

    def delete_session(self, session_id: str) -> bool:
        """Delete session entirely. Returns True if session existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return session_id in self._sessions
