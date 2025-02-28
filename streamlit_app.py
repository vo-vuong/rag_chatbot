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


async def chatbot_response(query):
    try:
        loop = asyncio.get_running_loop()
        history = _history.load_history()
        answer = await loop.run_in_executor(
            None,
            _chatbot_basic.chatbot,
            query,
            history,
            0.5
        )
        yield answer

    except Exception as e:
        content = {"status": 400, "message": "" + str(e)}
        response = f"'{content}'"
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.01)


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
        response = st.write_stream(chatbot_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
