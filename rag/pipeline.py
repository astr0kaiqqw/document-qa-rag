"""Compose document processing, retrieval, and extractive QA."""

from rag.document_loader import load_document
from rag.chunker import chunk_text
from rag.semantic_chunker import semantic_chunk_text
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.qa_model import QAModel


class RAGPipeline:
    """Answer questions from one loaded document using retrieval-ranked spans.

    Retrieval scores measure similarity, not answer confidence or correctness.
    ``source`` is the chunk containing the selected answer. No available answer
    is represented by empty answer/source strings and a score of 0.0.
    A failed load preserves the previously loaded document, if any.
    """

    def __init__(
        self,
        chunk_size=180,
        overlap=36,
        top_k=3,
        use_semantic_chunking=True,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.use_semantic_chunking = use_semantic_chunking
        self.embedding_model = None
        self.qa_model = None
        self.vector_store = None
        self.retriever = None

    def load_document(self, file_path: str):
        """Build a new document index; loader and indexing errors propagate."""
        text = load_document(file_path)
        chunker = semantic_chunk_text if self.use_semantic_chunking else chunk_text
        chunks = chunker(text, chunk_size=self.chunk_size, overlap=self.overlap)
        if not chunks:
            raise ValueError("Document contains no extractable chunks")
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()
        embeddings = self.embedding_model.encode(chunks)
        store = VectorStore(dimension=embeddings.shape[1])
        store.add(embeddings, chunks)
        retriever = Retriever(self.embedding_model, store)
        self.vector_store = store
        self.retriever = retriever

    def ask(self, question: str) -> dict:
        """Extract from each hit and select by retrieval similarity only."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        if self.retriever is None:
            raise RuntimeError("Load a document before asking questions")
        results = self.retriever.retrieve(question, top_k=self.top_k)
        response = {
            "question": question,
            "answer": "",
            "source": "",
            "retrieval_score": 0.0,
        }
        if not results:
            return response
        if self.qa_model is None:
            self.qa_model = QAModel()
        best_score = float("-inf")
        for result in results:
            answer = self.qa_model.answer(question, result["chunk"])
            score = float(result["score"])
            if answer and answer.strip() and score > best_score:
                best_score = score
                response.update(
                    answer=answer.strip(),
                    source=result["chunk"],
                    retrieval_score=score,
                )
        return response
