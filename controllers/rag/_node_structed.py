

from collections import OrderedDict
import concurrent.futures
from typing import List
from langchain_core.documents import Document
import re
from unstructured.documents.elements import NarrativeText, Title, Footer, Text, CompositeElement

# add text 1 node
def add_text(
    doc_contents: List[str],
    summaries: List[str],
    pages: List[int],
    file_names: List[str],
) -> tuple[list[Document], list[str]]:
    doc_ids = [i + 1 for i in range(len(doc_contents))]
    docs = []
    # for file_name in filter(None, file_names):
    #     print("file_name -----")
    #     print("file_name -----" + file_name)
    print("doc_contents lenth: =" + str(len(doc_contents)))
    for i, s in enumerate(doc_contents):
        if file_names[i] is None or summaries[i] is None:
            file_names[i] = "Stavian_Group.pdf"
            summaries[i] = "Stavian_Group.pdf"
            # continue
        # print("i --------------" + str(i))
        # print("file_names[i]----" + file_names[i])
        # print("summaries[i]" + summaries[i])
        match = re.search(r"page-(\d+)", file_names[i])

        name = file_names[i]
        split_string = name.split("-page")
        _file_name_ = split_string[0]

        metadata = {
            "type": "text",
            "page_content": s,
            "doc_id": doc_ids[i],
            "page": match.group(1) if match else None,
            "file_name": _file_name_,
        }
        docs.append(Document(page_content=summaries[i], metadata=metadata))
    return docs, doc_ids

# Lấy danh sách các id 3 node
def get_ids_3_node(docs):
    list_ids = []

    for doc in docs:
        id = doc.metadata["doc_id"]
        ids = [
            id - 3,
            id - 2,
            id - 1,
            id,
            id + 1,
            id + 2,
            id + 3,
        ]
        ids = [elem for elem in ids if elem > 0]
        list_ids.append(ids)

    return list_ids

def check_common_elements(l1, l2):
    return any(elem in l1 for elem in l2)

# Bước 2: Hợp nhất các danh sách có phần tử trùng nhau và sắp xếp tăng dần
def merge_lists(lists):
    merged = []
    visited = [False] * len(lists)

    for i in range(len(lists)):
        if visited[i]:
            continue
        current_merge = lists[i]
        visited[i] = True
        for j in range(i + 1, len(lists)):
            if check_common_elements(current_merge, lists[j]):
                current_merge += [
                    elem for elem in lists[j] if elem not in current_merge
                ]
                visited[j] = True
        merged.append(sorted(current_merge))  # Sắp xếp danh sách hiện tại

    return merged

class OrderedSet:
    def __init__(self, iterable):
        self.items = list(OrderedDict.fromkeys(iterable))

    def __repr__(self):
        return "{" + ", ".join(map(str, self.items)) + "}"

def get_element_data(raw_data_elements):
    print("raw_data_elements")
    # for element in raw_data_elements:
    #     print(element)
    datas = [None] * len(raw_data_elements)
    summaries = [None] * len(raw_data_elements)
    pages = [None] * len(raw_data_elements)
    file_names = [None] * len(raw_data_elements)

    def process_element(index, element):
        print("element_struc" + str(type(element)))
        print("e----" + str(element))
        print(element)
        print("element end")

        if isinstance(element, Text):
            file_name = element.metadata.filename
            text = str(element.text)
            page = element.metadata.page_number
            # print("text" + file_name)
            return index, text, text, page, file_name

        if isinstance(element, CompositeElement):
            file_name = element.metadata.filename
            page = sorted(
                {e.metadata.page_number for e in element.metadata.orig_elements}
            )
            ordered_set_numbers = OrderedSet(page)

            data_page = f"{str(element)}"
            return index, data_page, str(element), str(ordered_set_numbers), file_name
        
        if isinstance(element, NarrativeText):
            file_name = element.metadata.filename
            text = str(element.text)
            page = element.metadata.page_number
            summary = text
            print("element---------" + element)
            return index, text, summary, page, file_name

        # Xử lý dữ liệu kiểu Title
        if isinstance(element, Title):
            file_name = element.metadata.filename
            text = str(element.text)
            page = element.metadata.page_number
            summary = text
            print("file_name---------" + file_name)
            return index, text, summary, page, file_name

        # Xử lý dữ liệu kiểu Footer
        if isinstance(element, Footer):
            file_name = element.metadata.filename
            text = str(element.text)
            page = element.metadata.page_number
            summary = text
            print("file_name---------" + file_name)
            return index, text, summary, page, file_name

    with concurrent.futures.ThreadPoolExecutor() as executor:
        print("raw_data_elements")
        futures = [
            executor.submit(process_element, idx, element)
            for idx, element in enumerate(raw_data_elements)
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                index, data, summary, page, file_name = result
                datas[index] = data
                summaries[index] = summary
                pages[index] = page
                file_names[index] = file_name

        print("end raw_data_elements")

    return datas, summaries, pages, file_names

