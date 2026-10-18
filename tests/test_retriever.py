"""Tests for the Retriever component."""

import numpy as np
import pytest
from unittest.mock import Mock

from rag.retriever import Retriever
from rag.vector_store import VectorStore


@pytest.fixture
def retriever():
    """Create a retriever with a mock embedding model."""

    store = VectorStore(dimension=3)

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    chunks = [
        "Python is a programming language.",
        "Pizza is an Italian dish.",
        "Machine learning is part of AI.",
    ]

    store.add(embeddings, chunks)

    model = Mock()
    model.encode.return_value = np.array(
        [[1.0, 0.0, 0.0]], dtype=np.float32
    )

    return Retriever(model, store)


def test_retrieve_returns_results(retriever):
    results = retriever.retrieve("What is Python?")

    assert len(results) == 3


def test_retrieve_correct_chunk(retriever):
    results = retriever.retrieve("What is Python?", top_k=1)

    assert results[0]["chunk"] == "Python is a programming language."


def test_retrieve_top_k(retriever):
    results = retriever.retrieve("What is Python?", top_k=2)

    assert len(results) == 2


def test_empty_question(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_whitespace_question(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_invalid_top_k(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("What is Python?", top_k=0)