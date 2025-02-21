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




async def chatbot_response(user_input):
    try:
        loop = asyncio.get_running_loop()

        history = _history.load_history(
            # request.user_id, request.collection_id, _constants.NAME_CHATBOT_BASIC
            1, 1, _constants.NAME_CHATBOT_BASIC
        )

        answer = await loop.run_in_executor(
            None,
            _chatbot_basic.chatbot_basic_stream,
            user_input,
            1,
            history,
            1,
            _constants.NAME_CHATBOT_BASIC,
            0.5,
        )
        yield answer

    except Exception as e:
        content = {"status": 400, "message": "Lỗi chatbot basic: " + str(e)}
        response = f"Bot: Lỗi - '{content}'"
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
# user_input = st.chat_input("Nhập tin nhắn...")
# if user_input:
if prompt := st.chat_input("Nhập thông tin hỏi đáp..."):
    # Hiển thị tin nhắn của người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # chat_model = _environments.get_llm_stream(model="gpt-4o", temperature=0.5)
        # response = st.write_stream(chat_model.stream(prompt))
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        # response = st.write_stream(_chatbot_basic.chatbot_basic_stream(user_input, 1, '', '', _constants.NAME_CHATBOT_BASIC, 0.5))
        response = st.write_stream(chatbot_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
