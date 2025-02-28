import sys
import os
from controllers.rag import _re_write_query, _clean_data, _rag_qdrant
import time
from dotenv import load_dotenv
import threading
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import asyncio
from models import _environments, _prompts, _constants
load_dotenv()


# Stream
def chatbot(query, history, temperature=0.01):
    history_re_write = "\n\n".join([f"Q: {item['query']}" for item in history])
    re_write_query = """""" + _re_write_query.re_write_query(query=query, history=history_re_write)
    re_write_query = _clean_data.validate_and_fix_braces(re_write_query)
    print("========================================================")
    print("Re Write Query:\n", re_write_query)
    print("========================================================\n")

    retrievers = ""
    # for collection in list_collections:
    db = _rag_qdrant.load_vector_db(_constants.COLLECTION_NAME)

    retriever = _rag_qdrant.retriever_question(db, re_write_query, _constants.COLLECTION_NAME)
    retrievers += retriever + "\n\n"
    # print("=========" + retriever + "=========")

    history_context = "\n\n".join([f"Q: {item['query']}\nA: {item['answer']}" for item in history])
    print("========================================================")
    print("retrievers:\n", retrievers)
    history_context = _clean_data.validate_and_fix_braces(history_context)
    contexts = _clean_data.validate_and_fix_braces(retrievers)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _prompts.CHATBOT_QUOTE.format(
                    context=str(contexts), history=str(history_context)
                ),
            ),
            ("human", str(re_write_query)),
        ]
    )

    answer = ""
    chain = prompt | _environments.get_llm_stream(model="gpt-4o", temperature=temperature) | StrOutputParser()
    answer = chain.invoke({"input": str(re_write_query)})

    threading.Thread(
        target=_rag_qdrant.save_history,
        args=(
            answer,
            re_write_query
        ),
    ).start()

    return answer


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
