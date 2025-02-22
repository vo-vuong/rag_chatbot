from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
import time
from transformers import logging
logging.set_verbosity_error()
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"


def load_documents(file_path: str):
    # raw_elements = partition_pdf(filename=file_path,
    #                             #  extract_image_block_types=["Image"],
    #                             #  infer_table_structure=False,
    #                             #  form_extraction_skip_tables=False,
    #                             #  languages=["vie"],
    #                             #  strategy="auto",
    #                             #  hi_res_model_name="yolox",
    #                             #  chunking_strategy="by_title",
    #                             #  partitioning_strategies="auto",
    #                             #  max_characters=3072,
    #                             #  new_after_n_chars=3072,
    #                             #  combine_text_under_n_chars=1024,
    #                             #  multipage_sections=False,
    #                             #  extract_image_block_output_dir=_constants.FILE_IMAGE_OUTPUT_DIR_PATH
    # )
    raw_elements = partition(filename=file_path)

    return raw_elements


def load_documents_vie(file_path: str, path_image: str):
    start_time = time.time()
    raw_elements = partition_pdf(filename=file_path,
                                 extract_images_in_pdf=False,
                                 extract_image_block_types=["Image"],
                                 # extract_image_block_types=["Image"],
                                 # extract_image_block_to_payload=True,
                                 infer_table_structure=True,
                                 form_extraction_skip_tables=True,
                                #  languages=["eng"],
                                 languages=["vie"],
                                 strategy="auto",
                                 hi_res_model_name="yolox",
                                 chunking_strategy="by_title",
                                 partitioning_strategies="auto",
                                 max_characters=3072,
                                 new_after_n_chars=3072,
                                 combine_text_under_n_chars=1024,
                                 multipage_sections=True,
                                 extract_image_block_output_dir=path_image
    )
    end_time = time.time()  # Kết thúc đo thời gian
    print(f"Thời gian xử lý {file_path}: {end_time - start_time} giây")
    return raw_elements


def load_documents_eng(file_path: str, path_image: str):
    start_time = time.time()
    raw_elements = partition_pdf(filename=file_path,
                                 extract_image_block_types=["Image"],
                                 extract_images_in_pdf=False,
                                 # extract_image_block_types=["Image"],
                                 # extract_image_block_to_payload=True,
                                 infer_table_structure=True,
                                 form_extraction_skip_tables=False,
                                 # languages=["eng"],
                                 languages=["eng"],
                                 strategy="auto",
                                 hi_res_model_name="yolox",
                                 chunking_strategy="by_title",
                                 partitioning_strategies="auto",
                                 max_characters=3072,
                                 new_after_n_chars=3072,
                                 combine_text_under_n_chars=1024,
                                 multipage_sections=True,
                                 extract_image_block_output_dir=path_image
    )
    end_time = time.time()  # Kết thúc đo thời gian
    print(f"Thời gian xử lý {file_path}: {end_time - start_time} giây")
    return raw_elements


def load_documents_vie_eng(file_path: str, path_image: str):
    start_time = time.time()
    raw_elements = partition_pdf(filename=file_path,
                                 extract_image_block_types=["Image"],
                                 extract_images_in_pdf=False,
                                 # extract_image_block_types=["Image"],
                                 # extract_image_block_to_payload=True,
                                 infer_table_structure=True,
                                 form_extraction_skip_tables=False,
                                 # languages=["eng"],
                                 languages=["vie+eng"],
                                 strategy="auto",
                                 hi_res_model_name="yolox",
                                 chunking_strategy="by_title",
                                 partitioning_strategies="auto",
                                 max_characters=3072,
                                 new_after_n_chars=3072,
                                 combine_text_under_n_chars=1024,
                                 multipage_sections=True,
                                 extract_image_block_output_dir=path_image
    )
    end_time = time.time()  # Kết thúc đo thời gian
    print(f"# Thời gian xử lý {file_path}: {end_time - start_time} giây")
    return raw_elements


def load_documents_custom(file_path: str, path_image: str, chunk_size=5120, max_chunk_size=10240, chunk_overlap=0, 
                          table=False, images=None, languages="vie+eng"):
    raw_elements = partition_pdf(filename=file_path,
                                 extract_image_block_types=images,
                                 infer_table_structure=table,
                                 form_extraction_skip_tables=False,
                                 languages=[languages],
                                 strategy="auto",
                                 hi_res_model_name="yolox",
                                 chunking_strategy="by_title",
                                 partitioning_strategies="auto",
                                 max_characters=max_chunk_size,
                                 new_after_n_chars=max_chunk_size,
                                 combine_text_under_n_chars=chunk_size,
                                 multipage_sections=True,
                                 overlap=chunk_overlap,
                                 extract_image_block_output_dir=path_image
    )

    return raw_elements


# Test
# raw_data_elements = load_documents_vie_eng("../../files/tests/Prompt_Engineering.pdf")
#
# # for doc in docs:
# #     print(str(type(doc)) + "\n")
# #     print(doc)
# #     print("\n\n==================================================\n\n")
#
# for element in raw_data_elements:
#     if "unstructured.documents.elements.Table" in str(type(element)):
#         table_text = str(element.text)
#         table_html = str(element.metadata.text_as_html)
#
#         print("### Table: \n", table_text)
#         print("### Table HTML: \n", table_html)
#
#     elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
#         print("### CompositeElement: \n", element)
#
#     print("\n\n")




