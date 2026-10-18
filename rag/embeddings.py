"""Generate semantic embeddings for document chunks."""

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel:
    """Convert text into numerical vectors."""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings