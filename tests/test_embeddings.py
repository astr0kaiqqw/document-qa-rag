"""Tests for the embedding model."""

import numpy as np
import pytest

from rag.embeddings import EmbeddingModel


@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests."""
    return EmbeddingModel()


def test_embedding_shape(model):
    texts = [
        "Machine learning is interesting.",
        "Artificial intelligence is developing."
    ]

    embeddings = model.encode(texts)

    assert embeddings.shape == (2, 384)


def test_embedding_normalization(model):
    embeddings = model.encode(["Natural language processing"])

    norm = np.linalg.norm(embeddings[0])

    assert np.isclose(norm, 1.0, atol=1e-5)


def test_multilingual_similarity(model):
    texts = [
        "Machine learning is a branch of artificial intelligence.",
        "Машинное обучение является частью искусственного интеллекта.",
        "I like eating pizza."
    ]

    embeddings = model.encode(texts)

    similar = np.dot(embeddings[0], embeddings[1])
    different = np.dot(embeddings[0], embeddings[2])

    assert similar > different


def test_empty_input(model):
    embeddings = model.encode([])

    assert embeddings.shape == (0, 384)


def test_embedding_dtype(model):
    embeddings = model.encode(["Hello world"])

    assert embeddings.dtype == np.float32