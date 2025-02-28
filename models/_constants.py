# ====================== QDRANT ======================
# QDRANT_DIR_CRAWL = "./qdrant_db"
QDRANT_SERVER = "http://localhost"
# QDRANT_SERVER = "http://3.147.66.45:6333"
QDRANT_PORT = "6333"
COLLECTION_NAME = "test_collection4"


# ================ Embedding model ===================
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_DEVICE= {'device': 'cuda'}
ENCODE_KWARGS= {'normalize_embeddings': True}

# ====================== CHATBOT NAME ======================
NAME_CHATBOT = "CHATBOT"
NAME_CHATBOT_BASIC = "CHATBOT_BASIC"

NAME_CHATBOT_STAVIAN_GROUP = "STAVIAN_GROUP_CHATBOT"
NAME_CHATBOT_QUOTE_IMAGES_EXACTLY = "CHATBOT_QUOTE_IMAGES_EXACTLY"


# ====================== DATA PATH ======================
DATAS_PATH = "./files/datas/user"
DATAS_PUBLIC_PATH = "./files/datas/public"
DATAS_STAVIAN_GROUP_PATH = "./files/datas/stavian_group"

# ====================== IMAGE ======================
# IMAGES_LINK = "http://3.147.66.45/get-images/files/images_pdf/"
# IMAGES_LINK_1 = "http://3.147.66.45/get-images/files/images_pdf_menuNgonGarden/"

FILE_IMAGE_OUTPUT_DIR_PATH = "./files/images/"

MAX_FILE_SIZE = 20 * 1024 * 1024
TOTAL_POINTS_NAME = "total_points.txt"
