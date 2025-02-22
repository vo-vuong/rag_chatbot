import os
import re
import json
from PyPDF2 import PdfWriter, PdfReader
import concurrent.futures
from controllers.load_documents import _pdf
from controllers.rag import _node_structed
from models import _constants
from pathlib import Path
from controllers.vector_databases import _qdrant
import threading
import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

lock = Lock()


# Lưu nội dung vào file
def save_content_to_file(content, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(str(content))
        print(f"Content saved to {file_path} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        pass


# Thêm total point vào file
def add_totals_point_to_file(content, file_path):
    print('add_totals_point_to_file')
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(str(content))

        print(f"Content saved to {file_path} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        pass


# Đọc nội dung từ file total point
def read_totals_point_from_file(file_path):
    total = 0
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        print(f"Content read from {file_path} successfully.")
        total = int(content)
    except Exception as e:
        print(f"An error occurred: {e}")
        pass
    return total


# Đọc nội dung từ file
def read_content_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        print(f"Content read from {file_path} successfully.")
        return content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def normalize_string(input_str):
    replacements = {
        r"<QUOTE_DOC>": "QUOTE_DOC",
        r"<PREDICTED_INTENT>QUOTE_DOC</PREDICTED_INTENT>": "QUOTE_DOC",
        r'"QUOTE_DOC"': "QUOTE_DOC",
        r"'QUOTE_DOC'": "QUOTE_DOC",
        r"\[QUOTE_DOC\]": "QUOTE_DOC",
        r"\{QUOTE_DOC\}": "QUOTE_DOC",
        r"<FULL_DOC>": "FULL_DOC",
        r"<PREDICTED_INTENT>FULL_DOC</PREDICTED_INTENT>": "FULL_DOC",
        r'"FULL_DOC"': "FULL_DOC",
        r"'FULL_DOC'": "FULL_DOC",
        r"\[FULL_DOC\]": "FULL_DOC",
        r"\{FULL_DOC\}": "FULL_DOC",
    }

    for pattern, replacement in replacements.items():
        input_str = re.sub(pattern, replacement, input_str)

    return str(input_str)


def split_pdf_files_in_folder(folder_path, file_path):
    pdf_files = []
    path_images = []

    folder_path = Path(folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = folder_path / filename
            with open(file_path, "rb") as pdf_file:
                pdf = PdfReader(pdf_file)

                output_dir = folder_path / "splits" / file_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)

                for i in range(len(pdf.pages)):
                    output = PdfWriter()
                    output.add_page(pdf.pages[i])

                    path_image = (
                        folder_path
                        / "splits"
                        / file_path.stem
                        / f"{file_path.stem}-page-{i+1}"
                    )
                    output_file_path = output_dir / f"{file_path.stem}-page-{i+1}.pdf"
                    with open(output_file_path, "wb") as outputStream:
                        output.write(outputStream)

                    pdf_files.append(str(output_file_path))
                    path_images.append(str(path_image))

                print(f"Saved {filename} to {output_dir}")

    threading.Thread(target=remove_file, args=(file_path,)).start()

    return pdf_files, path_images


def remove_file(file_path_rm):
    try:
        if file_path_rm.exists():
            file_path_rm.unlink()
            print("File removed successfully.")
    except PermissionError as e:
        print("File remove error: ", e)
        pass


def extract_numbers_from_string(input_string):
    # Use regular expressions to find all numbers in the string
    numbers = re.findall(r"\d+", input_string)

    # Check if we have found the necessary numbers
    if len(numbers) < 2:
        return None

    # Extract the required numbers
    user_number = numbers[1]
    page_number = numbers[-1]

    # Concatenate the numbers with an underscore
    result = f"{user_number}_{page_number}"

    return result


def split_and_process_pdf(
    folder_path, language, user_id, count_list_collections_qdrant, chatbot_name, exactly, file_path
):
    try:
        pdf_files, path_images = split_pdf_files_in_folder(folder_path, file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        pdf_files, path_images = [], []

    # print(f"PDF Files: {pdf_files}")
    # print(f"Path Images: {path_images}")
    results = []
    list_img_base64 = []
    list_img_summaries = []
    list_img_file_name = []
    list_img_page = []

    lock = Lock()
    summary_futures = []

    images = None
    if (
        # chatbot_name == _constants.NAME_CHATBOT_QUOTE_IMAGES
        # or chatbot_name == _constants.NAME_CHATBOT_FULL_DOCS_IMAGES
        # or chatbot_name == _constants.NAME_CHATBOT_SERVICE_IMAGES
        # or chatbot_name == _constants.NAME_CHATBOT_QUOTE_IMAGES_EXACTLY
        _constants.NAME_CHATBOT_QUOTE_IMAGES_EXACTLY
    ):
        images = ["Image"]

    # def summary_image(path_image):
    #     path_image = Path(path_image)
    #     if not path_image.exists():
    #         return None

    #     print("Processing images...")

    #     path_str = str(path_image)
    #     page_number = path_str.split("page-")[1].split("-")[0]

    #     img_base64_list, image_summaries, images_file_name = asyncio.run(
    #         _summary.__generate_img_summaries(str(path_image))
    #     )

    #     for idx, (img_base64, summary, file_name) in enumerate(
    #         zip(img_base64_list, image_summaries, images_file_name)
    #     ):
    #         print(
    #             f"Image Summaries: {idx}: {(summary[:50] + '...') if len(summary) > 50 else summary}"
    #         )

    #         if (
    #             summary != "NULL"
    #             and summary != "Answer: NULL"
    #             and summary != "Answer: "
    #             and summary != ""

    #             or chatbot_name == _constants.NAME_CHATBOT_QUOTE_IMAGES_EXACTLY
    #         ):
    #             list_img_base64.append(img_base64)
    #             list_img_summaries.append(summary)
    #             list_img_file_name.append(file_name)
    #             list_img_page.append(page_number)

    #     shutil.rmtree(path_image)

    #     return img_base64_list

    def process_file(pdf_file, language, path_image):
        # print(f"Processing {pdf_file}...")
        try:
            if language == "vie" or language == "eng":
                data = _pdf.load_documents_custom(
                    file_path=pdf_file,
                    path_image=path_image,
                    languages=language,
                    images=images,
                )
            else:
                data = _pdf.load_documents_custom(
                    file_path=pdf_file, path_image=path_image, images=images
                )

            # if images is not None:
            #     with ThreadPoolExecutor() as summary_executor:
            #         summary_future = summary_executor.submit(summary_image, path_image)
            #         summary_futures.append(summary_future)

            return data
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def handle_result(future):
        file = future_to_file[future]
        print(f"Handling result for {file}...")
        try:
            match = re.search(r"page-(\d+)", file)
            print(f"Match: {match}")
            if match:
                name = int(match.group(1))
                result = future.result()
                with lock:
                    results.append((name, result))
                if file.endswith(".pdf"):
                    Path(file).unlink(missing_ok=True)
            else:
                print(f"Filename {file} does not contain a valid page number")
        except Exception as exc:
            print(f"File {file} generated an exception: {exc}")
            pass

        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(process_file, pdf_file, language, path_images[i]): pdf_file
                for i, pdf_file in enumerate(pdf_files)
            }
            for future in as_completed(future_to_file):
                handle_result(future)

        for future in summary_futures:
            future.result()

        # if images is not None:
        #     print(f"Save images to db ...")
        #     collection_name = f"{user_id}_{count_list_collections_qdrant + 1}_images"
        #     threading.Thread(
        #         target=save_images_to_db,
        #         args=(
        #             collection_name,
        #             list_img_base64,
        #             list_img_summaries,
        #             list_img_file_name,
        #             list_img_page,
        #             user_id,
        #             count_list_collections_qdrant,
        #             chatbot_name,
        #         ),
        #     ).start()

        # Sắp xếp kết quả theo thứ tự trang
        sorted_results = sorted(results, key=lambda x: x[0])
        raw_elements = [item for _, sublist in sorted_results for item in sublist]
        return raw_elements


def save_images_to_db(
    collection_name,
    list_img_base64,
    list_img_summaries,
    list_img_file_name,
    list_img_page,
    user_id,
    count_list_collections_qdrant,
    chatbot_name,
):
    if len(list_img_base64) > 0:
        try:
            add_images, id_node = _node_structed.add_images(
                list_img_base64, list_img_summaries, list_img_file_name, list_img_page
            )
            # Chuyển đổi add_images thành chuỗi JSON
            json_payload = json.dumps([doc.metadata for doc in add_images])
            # Tính toán kích thước của payload
            payload_size = len(json_payload.encode("utf-8"))
            file_chunk_size = 1

            if payload_size > _constants.MAX_FILE_SIZE:
                file_chunk_size = int(payload_size // _constants.MAX_FILE_SIZE + 1)

            _qdrant.save_vector_db_as_ids(
                add_images, collection_name, id_node, file_chunk_size
            )

            if chatbot_name == _constants.NAME_CHATBOT_MEKONGAI:
                threading.Thread(
                    target=add_totals_point_to_file,
                    args=(
                        len(list_img_base64),
                        _constants.DATAS_MEKONGAI_PATH
                        + "/"
                        + _constants.NAME_CHATBOT_MEKONGAI
                        + "/"
                        + user_id
                        + "/total_points_img_"
                        + str(count_list_collections_qdrant + 1)
                        + ".txt",
                    ),
                ).start()
            else:
                threading.Thread(
                    target=add_totals_point_to_file,
                    args=(
                        len(list_img_base64),
                        _constants.DATAS_PATH
                        + "/"
                        + _constants.NAME_CHATBOT_FULL_DOCS_IMAGES
                        + "/"
                        + user_id
                        + "/total_points_img_"
                        + str(count_list_collections_qdrant + 1)
                        + ".txt",
                    ),
                ).start()
                threading.Thread(
                    target=add_totals_point_to_file,
                    args=(
                        len(list_img_base64),
                        _constants.DATAS_PATH
                        + "/"
                        + _constants.NAME_CHATBOT_QUOTE_IMAGES
                        + "/"
                        + user_id
                        + "/total_points_img_"
                        + str(count_list_collections_qdrant + 1)
                        + ".txt",
                    ),
                ).start()
                threading.Thread(
                    target=add_totals_point_to_file,
                    args=(
                        len(list_img_base64),
                        _constants.DATAS_PATH
                        + "/"
                        + _constants.NAME_CHATBOT_SERVICE_IMAGES
                        + "/"
                        + user_id
                        + "/total_points_img_"
                        + str(count_list_collections_qdrant + 1)
                        + ".txt",
                    ),
                ).start()
                threading.Thread(
                    target=add_totals_point_to_file,
                    args=(
                        len(list_img_base64),
                        _constants.DATAS_PATH
                        + "/"
                        + _constants.NAME_CHATBOT_QUOTE_IMAGES_EXACTLY
                        + "/"
                        + user_id
                        + "/total_points_img_"
                        + str(count_list_collections_qdrant + 1)
                        + ".txt",
                    ),
                ).start()
        except Exception as e:
            print(f"An error occurred save_images_to_db: {e}")
            pass

    else:
        if chatbot_name == _constants.NAME_CHATBOT_MEKONGAI:
            threading.Thread(
                target=add_totals_point_to_file,
                args=(
                    len(list_img_base64),
                    _constants.DATAS_MEKONGAI_PATH
                    + "/"
                    + _constants.NAME_CHATBOT_MEKONGAI
                    + "/"
                    + user_id
                    + "/total_points_img_"
                    + str(count_list_collections_qdrant + 1)
                    + ".txt",
                ),
            ).start()

        else:
            threading.Thread(
                target=add_totals_point_to_file,
                args=(
                    0,
                    _constants.DATAS_PATH
                    + "/"
                    + _constants.NAME_CHATBOT_FULL_DOCS_IMAGES
                    + "/"
                    + user_id
                    + "/total_points_img_"
                    + str(count_list_collections_qdrant + 1)
                    + ".txt",
                ),
            ).start()
            threading.Thread(
                target=add_totals_point_to_file,
                args=(
                    0,
                    _constants.DATAS_PATH
                    + "/"
                    + _constants.NAME_CHATBOT_QUOTE_IMAGES
                    + "/"
                    + user_id
                    + "/total_points_img_"
                    + str(count_list_collections_qdrant + 1)
                    + ".txt",
                ),
            ).start()
            threading.Thread(
                target=add_totals_point_to_file,
                args=(
                    0,
                    _constants.DATAS_PATH
                    + "/"
                    + _constants.NAME_CHATBOT_SERVICE_IMAGES
                    + "/"
                    + user_id
                    + "/total_points_img_"
                    + str(count_list_collections_qdrant + 1)
                    + ".txt",
                ),
            ).start()
            threading.Thread(
                target=add_totals_point_to_file,
                args=(
                    0,
                    _constants.DATAS_PATH
                    + "/"
                    + _constants.NAME_CHATBOT_QUOTE_IMAGES_EXACTLY
                    + "/"
                    + user_id
                    + "/total_points_img_"
                    + str(count_list_collections_qdrant + 1)
                    + ".txt",
                ),
            ).start()


def extract_pages(text):
    # Tạo pattern để tìm từ "Trang", "Page", ... và theo sau là các số hoặc phạm vi số
    pattern = re.compile(
        r"(?i)(trang|page|số trang|page number|page numbers)\s*:\s*(\d+(-\d+)?(?:,\s*\d+(-\d+)?)*)(?=\s|$)"
    )

    # Tìm tất cả các khớp trong chuỗi
    matches = pattern.findall(text)

    if not matches:
        return []

    # Lấy tất cả các số hoặc phạm vi số từ khớp đầu tiên
    page_str = matches[0][1]

    # Tách các số trang và xử lý phạm vi
    pages = []
    for part in page_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))

    return pages


def check_fields(answer):
    # Chuyển đổi chuỗi về dạng chữ thường để dễ kiểm tra
    lower_answer = answer.lower()

    # Tạo một danh sách các từ khóa cần kiểm tra
    fields = ["số điện thoại", "địa chỉ", "email"]

    # Kiểm tra xem có bất kỳ từ khóa nào xuất hiện trong chuỗi hay không
    for field in fields:
        if field in lower_answer:
            return 1

    return 0
