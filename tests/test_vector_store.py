"""Tests for FAISS vector storage."""

import numpy as np
import pytest

from rag.vector_store import VectorStore


def test_add_vectors():
    store = VectorStore(dimension=3)

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    chunks = ["First document", "Second document"]

    store.add(embeddings, chunks)

    assert store.index.ntotal == 2
    assert store.chunks == chunks


def test_search():
    store = VectorStore(dimension=3)

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    store.add(embeddings, ["Python", "Pizza"])

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = store.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk"] == "Python"
    assert np.isclose(results[0]["score"], 1.0)


def test_empty_store():
    store = VectorStore(dimension=3)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert store.search(query) == []


def test_invalid_dimensions():
    store = VectorStore(dimension=3)

    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add(embeddings, ["Invalid"])


def test_mismatched_chunks():
    store = VectorStore(dimension=3)

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add(embeddings, ["Only one chunk"])


def test_save_and_load(tmp_path):
    store = VectorStore(dimension=3)

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    store.add(embeddings, ["Python", "Pizza"])

    save_path = tmp_path / "faiss_index"

    store.save(str(save_path))

    loaded_store = VectorStore.load(str(save_path))

    assert loaded_store.index.ntotal == 2
    assert loaded_store.chunks == ["Python", "Pizza"]

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = loaded_store.search(query, top_k=1)

    assert results[0]["chunk"] == "Python"


def test_invalid_top_k():
    store = VectorStore(dimension=3)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        store.search(query, top_k=0)


def test_zero_vector():
    store = VectorStore(dimension=3)

    embeddings = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add(embeddings, ["Invalid"])