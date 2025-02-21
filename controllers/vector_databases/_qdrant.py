
import sys
import os
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import qdrant_client
from qdrant_client.http.models import PointStruct, VectorParams
# from qdrant_client import Qdrant
from langchain_qdrant import Qdrant
from models import _constants, _environments
from langchain.schema import Document

class QdrantHandler:
    def __init__(self, url=_constants.QDRANT_SERVER, port=_constants.QDRANT_PORT):
        self.client = qdrant_client.QdrantClient(url=url, port=port)

    def get_collections(self):
        collections = self.client.get_collections()
        return collections

    def create_collection(self, collection_name, vector_size):
        vectors_config = VectorParams(size=vector_size, distance="Cosine")
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )

    def get_next_id(self, collection_name):
        "Get the next available id in the collection"
        all_points = self.client.scroll(collection_name=collection_name, limit=100, with_payload=False, with_vectors=False)
    
        if all_points[0]:
            max_id = max(point.id for point in all_points[0])
            return max_id + 1
        return 1

    def insert_vector(self, collection_name, vector, payload=None):
        point = PointStruct(id=1, vector=vector, payload=payload)
        self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )

    def search_vector(self, collection_name, query_vector, top_k=10):
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return search_result


    def save_vector_db_as_ids_single(self, docs, collection_name, point_ids):
        """
        Save documents to a Qdrant vector database with specified point IDs.

        Args:
            docs (list): A list of documents to be saved in the vector database.
            collection_name (str): The name of the collection in the Qdrant database.
            point_ids (list): A list of point IDs corresponding to the documents.

        Returns:
            Qdrant: An instance of the Qdrant class containing the saved documents.

        """
        qdrant_doc = Qdrant.from_documents(
            documents=docs,
            embedding=_environments.embeddings_model,
            url=_constants.QDRANT_SERVER,
            prefer_grpc=False,
            collection_name=collection_name,
            # api_key=_constants.QDRANT_API_KEY,
            ids=point_ids,
        )
        return qdrant_doc

    def save_vector_db_as_ids(self, docs, collection_name, point_ids, file_chunk_size):
        """
        Save documents to a vector database with specified point IDs.
        Args:
            docs (list): List of documents to be saved.
            collection_name (str): Name of the collection in the vector database.
            point_ids (list): List of point IDs corresponding to the documents.
            file_chunk_size (int): Size of each chunk to be processed. If 1, process documents individually.
        Returns:
            list: List of results from saving documents in chunks.
        Raises:
            ValueError: If the length of docs and point_ids are not equal.
        """
        if len(docs) != len(point_ids):
            raise ValueError("len(docs) != len(point_ids)")

        if file_chunk_size == 1:
            saves = self.save_vector_db_as_ids_single(docs, collection_name, point_ids)
            return saves

        chunks = len(docs) // file_chunk_size + (
            1 if len(docs) % file_chunk_size != 0 else 0
        )

        qdrant_docs = []
        for i in range(chunks):
            # Lấy ra các phần tử con theo kích thước file_chunk_size
            doc_chunk = docs[i * file_chunk_size : (i + 1) * file_chunk_size]
            point_id_chunk = point_ids[i * file_chunk_size : (i + 1) * file_chunk_size]

            qdrant_doc = self.save_vector_db_as_ids_single(
                doc_chunk, collection_name, point_id_chunk
            )
            qdrant_docs.append(qdrant_doc)

        # saves = run_bots_in_parallel(qdrant_docs)
        return qdrant_docs

# Example usage
if __name__ == "__main__":
    handler = QdrantHandler()
    # handler.create_collection("example_collection", vector_size=128)
    # handler.insert_vector("example_collection", [0.1] * 128, payload={"example": "data"})
    # results = handler.search_vector("example_collection", [0.1] * 128)
    # handler.insert_vector("my_collection", [0.1] * 128, payload={"example": "data", "image_name": "image.jpg"})
    # collections = handler.get_collections()
    # max_id = handler.get_next_id("my_collection")
    # print(collections)
    # print(max_id)
    # handler.delete_collection("example_collection")

    # create example for save_vector_db_as_ids_single
    docs = [
        Document("John is 25 years old."),
        Document("Jane is 30 years old."),
        Document(
            page_content="ChatGPT là một mô hình AI mạnh mẽ này.",
            metadata={"source": "Wikipedia", "language": "vi"},
        ),
    ]
    collection_name = "example_collection"
    point_ids = [4, 5, 6]
    handler.save_vector_db_as_ids_single(docs, collection_name, point_ids)
