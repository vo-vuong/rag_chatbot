from controllers.vector_databases import _qdrant
import os
from dotenv import load_dotenv
import threading
from collections import OrderedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from controllers.load_documents import _pdf

from controllers.rag import (
    _history,
    _node_structed,
    _clean_data,
    _re_write_query,
    _chain_invoke,
)
# from controllers.load_documents import (
#     _pdf,
#     _docx,
#     _txt,
#     _code,
#     _csv,
#     _doc,
#     _html,
#     _md,
#     _xlsx,
# )

from models import _environments, _prompts, _ultils, _constants
from queue import Queue

# import logging

# Cấu hình logging
# logging.basicConfig(
#     level=logging.INFO,  # Mức log tối thiểu
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("app.log"),  # Ghi log vào file
#         logging.StreamHandler()  # Hiển thị log ra console
#     ]
# )

# logger = logging.getLogger(__name__)
load_dotenv()

def save_history(answer, query):
    _history.save_history(query, answer)
    # _history.save_history_public(
    #     user_id, query, answer, collection_id, chatbot_name, base64_images, chart_answer
    # )

def retriever_question(db, query, collection_name):
    docs = _qdrant.similarity_search_qdrant_data(db, query, 3)

    # for doc in docs:
    #     print("\n========================================================")
    #     print("### doc id:\n" + str(doc.metadata["doc_id"]))
    #     print("### doc:\n" + doc.metadata["page_content"])
    #     print("========================================================\n")

    # list_ids = _node_structed.get_ids_3_node(docs)
    # results = _node_structed.merge_lists(list_ids)

    retrievers = ""
    # for result in docs:
    #     print("result: ", result)
    for result in docs:
        # retriever = get_value_branch(db, result, collection_name) + "\n"
        # print("result.metadata['_id']: ", result.metadata["_id"])
        retriever = get_value_branch(db, result, collection_name) + "\n"
        retrievers += retriever

        # nếu không phải là phần tử cuối cùng
        if result != docs[-1]:
            retrievers += "\n\n"

    # print("retrievers data: ======================================")
    # print("retrievers data: ", retrievers)

    return retrievers

def load_vector_db(collection_name):
    db = _qdrant.load_vector_db(collection_name)
    return db

def get_value_branch(db, result, collection_name):
    # print("point_id" + point_ids)
    # points = _qdrant.get_point_from_ids(
    #     db=db, collection_name=collection_name, point_ids=point_ids
    # )
    values = ""
    print("result: ", result)

    context_content = result.page_content
    context_filename = result.metadata["source"]
    context_page_number = result.metadata["page_label"]

    context = (
        str(context_content)
        + "\n\n# Thông tin trên thuộc trang "
        + str(context_page_number)
        + " của tài liệu "
        + str(context_filename)
        + "."
    )

    values += context + "\n\n"

    return values

def save_vector_db(file_path, user_id, language, chatbot_name, file_chunk_size, exactly):
    file_extension = file_path.split(".")[-1].lower()
    if chatbot_name == _constants.NAME_CHATBOT_STAVIAN_GROUP:
        _path = _constants.DATAS_STAVIAN_GROUP_PATH
    else:
        _path = _constants.DATAS_PATH

    path = _path + "/" + chatbot_name
    folder_path = f"{path}/{user_id}"

    loaders = {
        # "csv": _csv.load_documents,
        # "docx": _docx.load_documents,
        # "doc": _doc.load_documents,
        # "html": _html.load_documents,
        # "md": _md.load_documents,
        # "txt": _txt.load_documents,
        # "text": _txt.load_documents,
        # "log": _txt.load_documents,
        # "xlsx": _xlsx.load_documents,
        # "xls": _xlsx.load_documents,
    }
    count_list_collections_qdrant = _qdrant.count_list_collections_qdrant(
        1, "NAME_CHATBOT_STAVIAN_GROUP"
    )

    # Ham xu ly file pdf
    if file_extension in loaders:
        raw_elements = loaders[file_extension](file_path)
    elif file_extension == "pdf":
        raw_elements = _ultils.split_and_process_pdf(
            folder_path, language, user_id, count_list_collections_qdrant, chatbot_name, exactly, file_path
        )
        # raw_elements = _pdf.load_documents(file_path)
        # print('test raw_elements')
    else:
        raise ValueError(f"Unsupported file type: .{file_extension}")
    # if raw_elements is None:
    #     raise ValueError("Failed to process PDF file, raw_elements is None")

    if not isinstance(raw_elements, list):
        raise ValueError("raw_elements is not a list")

    # log raw_elements
    # for raw_element in raw_elements:
    #     print("+=========================================+")
    #     print(raw_element)


    threading.Thread(
        target=_ultils.add_totals_point_to_file,
        args=(
            len(raw_elements),
            path
            + "/"
            + user_id
            + "/total_points_"
            + str(count_list_collections_qdrant + 1)
            + ".txt",
        ),
    ).start()
    file_name = file_path.split("\\")[-1]
    file_name = file_name.split("/")[-1]

    datas, summaries, pages, file_names = _node_structed.get_element_data(raw_elements)
    print("Chương trình đang chạy...")
    exit()
    # if file_extension in ["xlsx", "xls", "csv"]:
    #     datas = raw_elements

    # print('test2')
    # log array datas
    # for data in datas:
    #     print(data)
    texts_optimized = _clean_data.clean_data_unstructured(datas)
    # print('test3')

    docs_text, doc_ids = _node_structed.add_text(
        texts_optimized, summaries, pages, file_names
        # datas, summaries, pages, file_names
    )
    # print('test4')
    if (
        # chatbot_name == _constants.NAME_CHATBOT_QUOTE_IMAGES
        # or chatbot_name == _constants.NAME_CHATBOT_FULL_DOCS_IMAGES
        # or chatbot_name == _constants.NAME_CHATBOT_SERVICE_IMAGES
        # or chatbot_name == _constants.NAME_CHATBOT_QUOTE_IMAGES_EXACTLY
        chatbot_name == _constants.NAME_CHATBOT_QUOTE_IMAGES_EXACTLY
    ):

        threading.Thread(
            target=_ultils.add_totals_point_to_file,
            args=(
                len(raw_elements),
                _constants.DATAS_PATH
                + "/"
                + _constants.NAME_CHATBOT_QUOTE_IMAGES_EXACTLY
                + "/"
                + user_id
                + "/total_points_"
                + str(count_list_collections_qdrant + 1)
                + ".txt",
            ),
        ).start()

    # elif chatbot_name == _constants.NAME_CHATBOT_MEKONGAI:
    #     collection_name = chatbot_name


    else:
        collection_name = (
            user_id
            + "_"
            + _constants.NAME_CHATBOT
            + "_"
            + str(count_list_collections_qdrant + 1)
        )

    print("collection_name: ", collection_name)
    _qdrant.save_vector_db_as_ids(docs_text, collection_name, doc_ids, file_chunk_size)

    print("test5")
    return collection_name
