"""FAISS vector storage and similarity search."""

from pathlib import Path
import json

import faiss
import numpy as np


class VectorStore:
    """Store embeddings and retrieve corresponding text chunks."""

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add(self, embeddings: np.ndarray, chunks: list[str]):
        """Add normalized embeddings and their text chunks."""

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Invalid embedding dimensions")

        if len(vectors) != len(chunks):
            raise ValueError("Number of embeddings and chunks must match")

        if not np.all(np.isfinite(vectors)):
            raise ValueError("Embeddings must contain finite values")

        if len(vectors) == 0:
            return

        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms == 0):
            raise ValueError("Embeddings cannot contain zero vectors")

        vectors = np.ascontiguousarray(vectors / norms[:, None])

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """Find the most similar chunks."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.shape == (self.dimension,):
            query = query.reshape(1, -1)

        if query.shape != (1, self.dimension):
            raise ValueError("Invalid query embedding dimensions")

        if not np.all(np.isfinite(query)):
            raise ValueError("Query embedding must contain finite values")

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("Query embedding cannot be zero")

        query = np.ascontiguousarray(query / norm)

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "chunk": self.chunks[int(idx)],
                "score": float(score),
                "index": int(idx),
            })

        return results

    def save(self, directory: str):
        """Save FAISS index and corresponding chunks."""

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "index.faiss")
        )

        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str):
        """Load a previously saved vector store from a trusted directory."""

        path = Path(directory)

        index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list) or not all(
            isinstance(chunk, str) for chunk in chunks
        ):
            raise ValueError("Invalid chunks data")

        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunks count do not match")

        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Expected inner product FAISS index")

        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Expected IndexFlatIP index")

        store = cls(dimension=index.d)
        store.index = index
        store.chunks = chunks

        return store