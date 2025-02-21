from langchain_huggingface import HuggingFaceEmbeddings
import _constants


class EmbeddingsModel:
    def __init__(self, model_name=_constants.MODEL_NAME, device=_constants.MODEL_DEVICE, 
                 normalize_embeddings=_constants.ENCODE_KWARGS):
        self.model_name = model_name
        self.model_kwargs = device
        self.encode_kwargs = normalize_embeddings
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def embed_query(self, query):
        return self.embeddings_model.embed_query(query)

    def embed_documents(self, documents):
        return self.embeddings_model.embed_documents(documents)

# Example usage
if __name__ == "__main__":
    model = EmbeddingsModel()
    print(model.embed_query("Hello, world!"))
    print(model.embed_documents(["Hello, world!", "How are you?", "This is an AI example."]))
