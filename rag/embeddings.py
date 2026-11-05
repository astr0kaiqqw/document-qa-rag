"""Generate semantic embeddings for document chunks."""

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel:
    """Convert text into numerical vectors."""

    def __init__(self, *, local_files_only=True, model_path=MODEL_NAME):
        self.model = SentenceTransformer(str(model_path), local_files_only=local_files_only)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        lengths = [len(ids) for ids in self.model.tokenizer(
            texts, add_special_tokens=True, truncation=False, verbose=False
        )["input_ids"]]
        if any(length > self.model.max_seq_length for length in lengths):
            raise ValueError(
                f"Embedding input exceeds {self.model.max_seq_length} tokens; "
                "use token-aware chunks (long questions must be shortened explicitly)"
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings