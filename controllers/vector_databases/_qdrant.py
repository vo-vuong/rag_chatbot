
import sys
import os
import json
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

    def get_point_from_ids(self, collection_name, point_ids):
        id = self.client.retrieve(collection_name=collection_name, ids=point_ids)
        return id

    def get_value_points(self, point_ids, collection_name):
        points = self.get_point_from_ids(
            db=self, collection_name=collection_name, point_ids=point_ids
        )
        values = []

        for point in points:
            value = {
                "page_content": point.payload["metadata"]["page_content"],
                "metadata": {
                    "file_name": point.payload["metadata"]["file_name"],
                    "page_number": point.payload["metadata"]["page"],
                },
            }
            values.append(value)
        return values

    # def load_vector_db(self, collection_names):
    #     try:
    #         client = self.client.from_existing_collection(
    #             embedding=_environments.embeddings_model,
    #             collection_name=collection_names,
    #             # url=_constants.QDRANT_SERVER,
    #             # api_key=_constants.QDRANT_API_KEY,
    #         )
    #         return client
    #     except Exception:
    #         a = "None"
    #         return a
        
def load_vector_db(collection_names):
    try:
        client = Qdrant.from_existing_collection(
            embedding=_environments.embeddings_model,
            collection_name=collection_names,
            url=_constants.QDRANT_SERVER,
            port=_constants.QDRANT_PORT,
            # api_key=_constants.QDRANT_API_KEY,
        )
        print(client)
        return client
    except Exception:
        a = "None"
        return a

def similarity_search_qdrant_data(db, query, k=3):
    docs = db.similarity_search(query=query, k=k)
    return docs

def get_point_from_ids(db, collection_name, point_ids):
    id = db.client.retrieve(collection_name=collection_name, ids=point_ids)
    return id

def count_list_collections_qdrant(user_id, chatbot_name, _path=_constants.DATAS_PATH):
    # Đường dẫn đến file JSON
    json_file_path = os.path.join(
        _path, chatbot_name, str(user_id), "list_collections_qdrant.json"
    )

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    # Đọc dữ liệu hiện có từ file JSON
    if os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = []

    # Tìm tổng số dữ liệu trong file
    file_count = len(data)

    return int(file_count)

def save_vector_db_as_ids_single(docs, collection_name, point_ids):
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


def save_vector_db_as_ids(docs, collection_name, point_ids, file_chunk_size):
    # if len(docs) != len(point_ids):
    #     raise ValueError("len(docs) != len(point_ids)")

    if file_chunk_size == 1:
        saves = save_vector_db_as_ids_single(docs, collection_name, point_ids)
        return saves

    chunks = len(docs) // file_chunk_size + (
        1 if len(docs) % file_chunk_size != 0 else 0
    )

    qdrant_docs = []
    for i in range(chunks):
        # Lấy ra các phần tử con theo kích thước file_chunk_size
        doc_chunk = docs[i * file_chunk_size : (i + 1) * file_chunk_size]
        point_id_chunk = point_ids[i * file_chunk_size : (i + 1) * file_chunk_size]

        qdrant_doc = save_vector_db_as_ids_single(
            doc_chunk, collection_name, point_id_chunk
        )
        qdrant_docs.append(qdrant_doc)

    # saves = run_bots_in_parallel(qdrant_docs)
    return qdrant_docs




# Example usage
# if __name__ == "__main__":
#     handler = QdrantHandler()
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
    # docs = [
    #     Document("John is 25 years old."),
    #     Document("Jane is 30 years old."),
    #     Document(
    #         page_content="ChatGPT là một mô hình AI mạnh mẽ này.",
    #         metadata={"source": "Wikipedia", "language": "vi"},
    #     ),
    # ]
    # collection_name = "example_collection"
    # point_ids = [4, 5, 6]
    # handler.save_vector_db_as_ids_single(docs, collection_name, point_ids)

    # get point from ids
    # point_ids = [4, 5, 6]
    # points = handler.get_point_from_ids("example_collection", point_ids)
    # print(points)

    
