"""Compose document processing, retrieval, and extractive QA."""

from rag.document_loader import load_document, load_document_details
from rag.chunker import chunk_text, token_chunk_text
from rag.semantic_chunker import semantic_chunk_text
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.qa_model import QAModel
from rag.answer_selection import rank_candidates, STRATEGIES


class RAGPipeline:
    """Answer questions from one local document using traceable evidence spans.

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
        token_aware_chunking=True,
        embedding_token_limit=96,
        token_overlap=24,
        answer_selection="retrieval",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if embedding_token_limit is not None and embedding_token_limit <= 2:
            raise ValueError("embedding_token_limit must leave room for content")
        if token_overlap < 0:
            raise ValueError("token_overlap must be non-negative")
        if answer_selection not in STRATEGIES:
            raise ValueError("Unknown answer selection strategy")
        self.answer_selection = answer_selection
        self.token_aware_chunking = token_aware_chunking
        self.embedding_token_limit = embedding_token_limit
        self.token_overlap = token_overlap
        self.chunk_metadata = []
        self.document_text = ""
        self.document_pages = ()
        self.extraction_diagnostics = ()
        self.document_path = ""
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
        document = load_document_details(file_path)
        text = document.text
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()
        metadata = []
        if self.token_aware_chunking:
            limit = min(self.embedding_token_limit or self.embedding_model.model.max_seq_length,
                        self.embedding_model.model.max_seq_length)
            metadata = token_chunk_text(
                text, self.embedding_model.model.tokenizer, max_tokens=limit,
                overlap_tokens=self.token_overlap,
                sentence_boundaries=self.use_semantic_chunking,
            )
            chunks = [chunk.text for chunk in metadata]
        else:
            chunker = semantic_chunk_text if self.use_semantic_chunking else chunk_text
            chunks = chunker(text, chunk_size=self.chunk_size, overlap=self.overlap)
        if not chunks:
            raise ValueError("Document contains no extractable chunks")
        embeddings = self.embedding_model.encode(chunks)
        store = VectorStore(dimension=embeddings.shape[1])
        store.add(embeddings, chunks)
        retriever = Retriever(self.embedding_model, store)
        self.chunk_metadata = metadata
        self.document_text = text
        self.document_pages = document.pages
        self.extraction_diagnostics = document.diagnostics
        self.document_path = str(file_path)
        self.vector_store = store
        self.retriever = retriever

    def ask(self, question: str) -> dict:
        """Original four-field response. Use ask_detailed for evidence and uncertainty."""
        detailed = self.ask_detailed(question)
        return {key: detailed[key] for key in
                ("question", "answer", "source", "retrieval_score")}

    def ask_detailed(self, question: str) -> dict:
        """Evaluate all retrieved evidence; expose source coordinates and ranking.

        `candidate` explicitly means unverified extractive output, not calibrated
        confidence. SQuAD-v1 cannot certify semantic absence of an answer.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        if self.retriever is None:
            raise RuntimeError("Load a document before asking questions")
        hits = self.retriever.retrieve(question, top_k=self.top_k)
        response = dict(question=question, answer="", source="", retrieval_score=0.0,
                        status="no_evidence", candidates=[], selection=self.answer_selection,
                        no_answer_supported=False, document_path=self.document_path,
                        extraction_diagnostics=list(self.extraction_diagnostics))
        if not hits:
            return response
        if self.qa_model is None:
            self.qa_model = QAModel()
        candidates = []
        for rank, hit in enumerate(hits,1):
            chunk = hit["chunk"]
            if not chunk.strip():
                continue
            index = hit.get("index",rank-1)
            if callable(getattr(self.qa_model, "candidates", None)):
                extracted = self.qa_model.candidates(question,chunk,top_n=1)
            else:
                # Compatible with string-returning custom QA backends; reject fabricated spans.
                answer = self.qa_model.answer(question,chunk).strip()
                a = chunk.find(answer) if answer else -1
                extracted = [dict(answer=answer,start=a,end=a+len(answer),log_score=None)] if a>=0 else []
            for span in extracted:
                a,b = span['start'],span['end']
                metadata = self.chunk_metadata[index] if "index" in hit and 0 <= index < len(self.chunk_metadata) else None
                candidates.append(dict(span,source=chunk,chunk_index=index,retrieval_rank=rank,
                    retrieval_score=float(hit['score']),
                    document_start=metadata.start+a if metadata else None,
                    document_end=metadata.start+b if metadata else None))
        ranked = rank_candidates(candidates,self.answer_selection)
        response['candidates'] = ranked
        if not ranked:
            response['status'] = 'no_admissible_span'
            return response
        for candidate in ranked:
            a,b = candidate['document_start'],candidate['document_end']
            candidate['pages'] = [p.number for p in self.document_pages
                                  if p.number is not None and a is not None and p.start < b and p.end > a]
        selected = ranked[0]
        response.update(answer=selected['answer'],source=selected['source'],
                        retrieval_score=selected['retrieval_score'],status='candidate',
                        document_start=selected['document_start'],document_end=selected['document_end'],
                        pages=selected['pages'])
        return response
