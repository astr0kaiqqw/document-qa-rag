"""Pipeline tests use real loading, chunking and FAISS with mocked models."""

from unittest.mock import Mock

import numpy as np
import pytest

import rag.pipeline as pipeline_module
from rag.pipeline import RAGPipeline


@pytest.fixture
def pipeline(monkeypatch):
    embeddings = Mock()
    embeddings.encode.side_effect = lambda texts: np.tile(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1)
    )
    qa = Mock()
    qa.answer.return_value = "Python"
    monkeypatch.setattr(pipeline_module, "EmbeddingModel", Mock(return_value=embeddings))
    monkeypatch.setattr(pipeline_module, "QAModel", Mock(return_value=qa))
    return RAGPipeline()


@pytest.fixture
def document(tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("Python is a programming language.", encoding="utf-8")
    return str(path)


def test_load_document(pipeline, document):
    pipeline.load_document(document)
    assert pipeline.vector_store.index.ntotal == 1
    assert pipeline.vector_store.chunks == ["Python is a programming language."]
    assert pipeline.retriever.vector_store is pipeline.vector_store
    pipeline.embedding_model.encode.assert_called_once_with(pipeline.vector_store.chunks)


def test_ask(pipeline, document):
    pipeline.load_document(document)
    assert pipeline.ask("What is a programming language?") == {
        "question": "What is a programming language?",
        "answer": "Python",
        "source": "Python is a programming language.",
        "retrieval_score": 1.0,
    }


@pytest.mark.parametrize("question", ["", "   ", None, 12])
def test_empty_or_invalid_question(pipeline, question):
    with pytest.raises(ValueError, match="non-empty string"):
        pipeline.ask(question)


def test_empty_document(pipeline, tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no extractable text"):
        pipeline.load_document(str(path))
    assert pipeline.retriever is None


def test_unsupported_document(pipeline, tmp_path):
    path = tmp_path / "document.csv"
    path.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        pipeline.load_document(str(path))


def test_ask_before_load(pipeline):
    with pytest.raises(RuntimeError, match="Load a document"):
        pipeline.ask("What is Python?")


def test_empty_retrieval(pipeline, document):
    pipeline.load_document(document)
    pipeline.retriever = Mock()
    pipeline.retriever.retrieve.return_value = []
    assert pipeline.ask("Question?") == {
        "question": "Question?", "answer": "", "source": "", "retrieval_score": 0.0
    }
    assert pipeline.qa_model is None


@pytest.mark.parametrize("answer", ["", "   "])
def test_empty_qa_answer(pipeline, document, answer):
    pipeline.load_document(document)
    pipeline.qa_model = Mock()
    pipeline.qa_model.answer.return_value = answer
    assert pipeline.ask("Question?") == {
        "question": "Question?", "answer": "", "source": "", "retrieval_score": 0.0
    }


def test_answer_ranking_and_all_chunks_processed(pipeline, document):
    pipeline.load_document(document)
    pipeline.top_k = 4
    pipeline.retriever = Mock()
    pipeline.retriever.retrieve.return_value = [
        {"chunk": "empty", "score": 0.9},
        {"chunk": "lower", "score": -0.5},
        {"chunk": "best", "score": -0.1},
        {"chunk": "last", "score": -0.8},
    ]
    pipeline.qa_model = Mock()
    pipeline.qa_model.answer.side_effect = ["", "lower answer", " best answer ", "last answer"]
    result = pipeline.ask("Question?")
    assert result["answer"] == "best answer"
    assert result["source"] == "best"
    assert result["retrieval_score"] == -0.1
    assert pipeline.qa_model.answer.call_count == 4
    pipeline.retriever.retrieve.assert_called_once_with("Question?", top_k=4)


@pytest.mark.parametrize("semantic", [True, False])
def test_switch_chunkers(pipeline, document, monkeypatch, semantic):
    semantic_chunker = Mock(wraps=pipeline_module.semantic_chunk_text)
    legacy_chunker = Mock(wraps=pipeline_module.chunk_text)
    monkeypatch.setattr(pipeline_module, "semantic_chunk_text", semantic_chunker)
    monkeypatch.setattr(pipeline_module, "chunk_text", legacy_chunker)
    assert pipeline.use_semantic_chunking is True
    pipeline.use_semantic_chunking = semantic
    pipeline.load_document(document)
    selected, unused = (semantic_chunker, legacy_chunker) if semantic else (legacy_chunker, semantic_chunker)
    selected.assert_called_once_with(
        "Python is a programming language.", chunk_size=180, overlap=36
    )
    unused.assert_not_called()


def test_reload_replaces_index(pipeline, document, tmp_path):
    pipeline.load_document(document)
    path = tmp_path / "second.txt"
    path.write_text("Another document.", encoding="utf-8")
    pipeline.load_document(str(path))
    assert pipeline.vector_store.chunks == ["Another document."]
    assert pipeline.vector_store.index.ntotal == 1


def test_failed_reload_preserves_document(pipeline, document, tmp_path):
    pipeline.load_document(document)
    previous = pipeline.retriever
    with pytest.raises(FileNotFoundError):
        pipeline.load_document(str(tmp_path / "missing.txt"))
    assert pipeline.retriever is previous


def test_no_chunks(pipeline, document, monkeypatch):
    monkeypatch.setattr(pipeline_module, "semantic_chunk_text", Mock(return_value=[]))
    with pytest.raises(ValueError, match="no extractable chunks"):
        pipeline.load_document(document)


@pytest.mark.parametrize("kwargs", [
    {"chunk_size": 0}, {"overlap": -1}, {"overlap": 180}, {"top_k": 0}
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RAGPipeline(**kwargs)
