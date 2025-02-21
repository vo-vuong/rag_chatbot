import sys
import os
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time

from dotenv import load_dotenv
import threading
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from controllers.rag import _clean_data
# from controllers.rag import _rag_qdrant, _re_write_query
import asyncio
from models import _environments, _prompts
load_dotenv()


# Stream
def chatbot_basic_stream(query, user_id, history, collection_id, chatbot_name, temperature=0.5):
    history_re_write = "\n\n".join([f"Q: {item['query']}" for item in history])
    re_write_query = query
    # re_write_query = """""" + _re_write_query.re_write_query(query=query, history=history_re_write)
    # re_write_query = _clean_data.validate_and_fix_braces(re_write_query)
    # print("========================================================")
    # print("Re Write Query:\n", re_write_query)
    # print("========================================================\n")

    # history_context = "\n\n".join([f"Q: {item['query']}\nA: {item['answer']}" for item in history])

    # history_context = _clean_data.validate_and_fix_braces(history_context)
    history_context = ""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _prompts.CHATBOT_BASIC.format(history=str(history_context))),
        ("human", str(re_write_query)),
    ])

    chain = prompt | _environments.get_llm_stream(model="gpt-4o", temperature=temperature) | StrOutputParser()

    answer = ""
    # base64_images = []

    async def generate_chat_responses(message):
        nonlocal answer
        chart_answer = ""
        async for chunk in chain.astream(message):
            content = chunk.replace("\n", "<br>")
            answer += content  # Ghép các đoạn nhỏ lại thành câu trả lời hoàn chỉnh
            # yield f"data: {content}\n\n"
            yield content

    #     tmp = re_write_query + " , " + query
    #     text_query = tmp.lower()

        # if "biểu đồ" in text_query or "bieu do" in text_query or "chart" in text_query:
        #     yield f"data: ---chart---\n\n"
        #     chart_answer = _chart.draw_char(history_context, str(re_write_query))

        #     yield f"data: {chart_answer}\n\n"
        #     yield f"data: ---end_chart---\n\n"

        # threading.Thread(
        #     target=_rag_qdrant.save_history,
        #     args=(
        #         answer,
        #         user_id,
        #         re_write_query,
        #         collection_id,
        #         chatbot_name,
        #         base64_images,
        #         chart_answer,
        #     ),
        # ).start()
    # re_write_query = "Hello, how are you?"
    stream_response = generate_chat_responses({"input": str(re_write_query)})

    # return _environments.get_llm_stream(model="gpt-4o", temperature=temperature).stream("Giải thích về Machine Learning")
    return stream_response

# if __name__ == "__main__":
    # query = "Hello, how are you?"
    # user_id = "user123"
    # history = [{"query": "Hi", "answer": "Hello!"}]
    # collection_id = "collection123"
    # chatbot_name = "TestBot"
    # temperature = 0.5

    # stream_response = chatbot_basic_stream(query, user_id, history, collection_id, chatbot_name, temperature)

    # async def collect_responses():
    #     responses = []
    #     async for response in stream_response:
    #         responses.append(response)
    #     print(responses)

    # asyncio.run(collect_responses())
