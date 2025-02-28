import streamlit as st
import pandas as pd
import numpy as np
import asyncio
from controllers.rag import (
    _chatbot_basic,
    _history
)
from models import _environments, _prompts, _constants
import time


def chatbot_response(query):
    try:
        query = query.lower()
        history = _history.load_history()
        answer = _chatbot_basic.chatbot(query, history, 0.5)
        return answer

    except Exception as e:
        content = {"status": 400, "message": "" + str(e)}
        response = f"'{content}'"
        return response


# Khởi tạo session state để lưu lịch sử chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Stavian Group Chatbot")

# Hiển thị lịch sử tin nhắn
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input từ người dùng
if prompt := st.chat_input("Nhập thông tin hỏi đáp..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chatbot_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
