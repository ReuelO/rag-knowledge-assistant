from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate vector embeddings for text."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
        """Convert text into embedding vectors."""
        return self.model.encode(texts)
