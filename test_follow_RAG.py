import chromadb
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama
# from llama_index.vector_stores.chroma import ChromaVectorStore
from unstructured.partition.auto import partition
import re
from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.http import models

def split_text(text, max_length=500):
    # Tách theo dấu chấm câu
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# main function
if __name__ == '__main__':
    # Giả sử file_path là đường dẫn file được cung cấp
    file_path = "CV_VoXuanQuocVuong_Intern_AI_Engineer.pdf"
    elements = partition(filename=file_path)
    #print(elements)

    # text = text.lower()

    # Gom các đoạn văn bản từ elements
    document_text = "\n".join([element.text for element in elements if hasattr(element, "text")])
    # print(document_text)

    text_chunks = split_text(document_text, max_length=200)
    # print(text_chunks)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)

    # Kết nối đến Qdrant
    client = QdrantClient(host="localhost", port=6333)
    print(client.info())
    print(client.collection_exists("documents"))
    # print(client.collection_exists())

    # Tạo collection nếu chưa có (ví dụ: collection tên "documents")
    # collection_name = "documents"
    # client.recreate_collection(
    #     collection_name=collection_name,
    #     vectors_config=models.VectorParams(
    #         size=len(embeddings[0]),   # kích thước vector embedding
    #         distance=models.Distance.COSINE
    #     )
    # )


