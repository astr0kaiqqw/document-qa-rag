"""Retrieve relevant document chunks using semantic search."""

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


class Retriever:
    """Find relevant text chunks for a user question."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 3):
        """Return the most relevant chunks for a question."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Convert question into embedding
        query_embedding = self.embedding_model.encode([question])[0]

        # Search FAISS index
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        return results