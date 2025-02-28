from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
import asyncio
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
import os
from models import _constants
from langchain.schema import Document
import fitz  # PyMuPDF
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


async def upload_files():
    # pdf_loader = PyPDFLoader('Stavian_Group.pdf', extract_images=True)
    pdf_loader = PyPDFLoader('Stavian_Group.pdf')
    # pdf_loader = PyPDFLoader('Luat.pdf')
    docs_file = pdf_loader.load()


    # Convert the text in the documents to lowercase
    for doc in docs_file:
        doc.page_content = doc.page_content.lower()

    # chunk_size = 300
    # chunk_overlap = 30
    # separator: List[str] = ['\n\n', '\n', '. ', ' ']

    # char_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     separators=separator,
    #     length_function=len,
    #     is_separator_regex=False
    # )

    # text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
    # embedding = HuggingFaceEmbeddings()
    embeddings_model = OpenAIEmbeddings()
    # docs = char_splitter.split_documents(docs_file)
    # docs = text_splitter.split_documents(docs_file)
    text_splitter = SemanticChunker(embeddings_model, breakpoint_threshold_type="percentile")
    docs = text_splitter.split_documents(docs_file)

    ## Add new docs and embedding to langchain_qdrant
    qdrant_doc = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings_model,
        url="http://localhost:6333",
        prefer_grpc=False,
        collection_name=_constants.COLLECTION_NAME,
    )


if __name__ == "__main__":
    asyncio.run(upload_files())
    # embedding = HuggingFaceEmbeddings()
    embeddings_model = OpenAIEmbeddings()
    qdrant = Qdrant(client=QdrantClient(url="http://localhost", port="6333"), collection_name=_constants.COLLECTION_NAME, embeddings=embeddings_model)
    query = "When: Khi làm việc và phục vụ khách hàng"
    similar_docs = qdrant.similarity_search(query, k=3)
    for doc in similar_docs:
        print(f"Document: {doc.page_content}\n")
