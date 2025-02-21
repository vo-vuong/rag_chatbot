# ====================== QDRANT ======================
# QDRANT_DIR_CRAWL = "./qdrant_db"
QDRANT_SERVER = "http://localhost"
# QDRANT_SERVER = "http://3.147.66.45:6333"
QDRANT_PORT = "6333"


# ================ Embedding model ===================
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_DEVICE= {'device': 'cuda'}
ENCODE_KWARGS= {'normalize_embeddings': True}
