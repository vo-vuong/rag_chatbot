# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import BGE_M3Embeddings
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
# os.environ["OPENAI_API_KEY"] = 


########################################################################################

# OpenAIEmbeddings embeddings_model
# embeddings_model = OpenAIEmbeddings()

# BAAI/bge-m3 embeddings_model
# embeddings_model = BGE_M3Embeddings()

# HuggingFace BGE embeddings_model
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
# embeddings_model = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

# embeddings_model = OpenAIEmbeddings()

# llm
llm_4o_00_temperature = ChatOpenAI(model="gpt-4o", temperature=0)
llm_4o_05_temperature = ChatOpenAI(model="gpt-4o", temperature=0.5)
llm_4o_10_temperature = ChatOpenAI(model="gpt-4o", temperature=1)
llm_4o_15_temperature = ChatOpenAI(model="gpt-4o", temperature=1.5)

llm_4o_mini_0_temperature = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_4o_mini_05_temperature = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
llm_4o_mini_10_temperature = ChatOpenAI(model="gpt-4o-mini", temperature=1)

llm_35 = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.5)

# llm stream
llm_4o_00_stream_temperature = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
llm_4o_05_stream_temperature = ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)
llm_4o_10_stream_temperature = ChatOpenAI(model="gpt-4o", temperature=1, streaming=True)
llm_4o_15_stream_temperature = ChatOpenAI(model="gpt-4o", temperature=1.5, streaming=True)

llm_4o_mini_0_stream_temperature = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
llm_4o_mini_05_stream_temperature = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True)
llm_4o_mini_10_stream_temperature = ChatOpenAI(model="gpt-4o-mini", temperature=1, streaming=True)

def get_llm_stream(model="gpt-4o", temperature=0.5):
    if model == "gpt-4o":
        if temperature == 0:
            return llm_4o_00_stream_temperature
        elif temperature == 0.5:
            return llm_4o_05_stream_temperature
        elif temperature == 1:
            return llm_4o_10_stream_temperature
        elif temperature == 1.5:
            return llm_4o_15_stream_temperature
        else:
            return ChatOpenAI(model=model, temperature=temperature, streaming=True)

    elif model == "gpt-4o-mini":
        if temperature == 0:
            return llm_4o_mini_0_stream_temperature
        elif temperature == 0.5:
            return llm_4o_mini_05_stream_temperature
        elif temperature == 1:
            return llm_4o_mini_10_stream_temperature
        else:
            return ChatOpenAI(model=model, temperature=temperature, streaming=True)

    else:
        return ChatOpenAI(model=model, temperature=temperature, streaming=True)


# Example usage
# if __name__ == "__main__":
    # print(embeddings_model.embed_query("Hello, world!"))
    # print(embeddings_model.embed_documents("Hello, world!"))
    # chat_model = get_llm_stream(model="gpt-4o", temperature=0.5)
    # chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    # response = chat_model.invoke("Làm thế nào để học lập trình Python?")
    # print(response.content)
    # for chunk in chat_model.stream("Giải thích về Machine Learning"):
    #     print(chunk.content, end="", flush=True)
