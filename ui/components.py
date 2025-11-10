from typing import List

import streamlit as st


def notify(message: str, notification_type: str = "info") -> None:
    """
    Display notification message.

    Args:
        message: Message to display
        notification_type: Type of notification ('info', 'success', 'warning', 'error')
    """
    if notification_type == "info":
        st.info(message)
    elif notification_type == "success":
        st.success(message)
    elif notification_type == "warning":
        st.warning(message)
    elif notification_type == "error":
        st.error(message)


def display_chat_message(role: str, content: str) -> None:
    """
    Display a chat message.

    Args:
        role: Message role ('user' or 'assistant')
        content: Message content
    """
    with st.chat_message(role):
        st.markdown(content)


def display_chat_history(chat_history: List[dict]) -> None:
    """
    Display chat history.

    Args:
        chat_history: List of message dictionaries with 'role' and 'content'
    """
    for message in chat_history:
        display_chat_message(message["role"], message["content"])
