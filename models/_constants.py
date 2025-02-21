# ====================== QDRANT ======================
# QDRANT_DIR_CRAWL = "./qdrant_db"
QDRANT_SERVER = "http://localhost"
# QDRANT_SERVER = "http://3.147.66.45:6333"
QDRANT_PORT = "6333"


# ================ Embedding model ===================
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_DEVICE= {'device': 'cuda'}
ENCODE_KWARGS= {'normalize_embeddings': True}

# ====================== CHATBOT NAME ======================
NAME_CHATBOT = "CHATBOT"
NAME_CHATBOT_BASIC = "CHATBOT_BASIC"

NAME_CHATBOT_STAVIAN_GROUP = "STAVIAN_GROUP_CHATBOT"


# ====================== DATA PATH ======================
DATAS_PATH = "./files/datas/user"
DATAS_PUBLIC_PATH = "./files/datas/public"
DATAS_STAVIAN_GROUP_PATH = "./files/datas/stavian_group"
