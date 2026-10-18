import pytest

from rag.chunker import chunk_text


def test_empty_text():
    """Empty text should produce no chunks."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text():
    """Text shorter than chunk_size should remain one chunk."""
    text = "Machine learning is interesting"

    result = chunk_text(text, chunk_size=10, overlap=2)

    assert result == [text]


def test_chunk_size():
    """Each chunk must contain no more than chunk_size words."""
    text = " ".join(f"word{i}" for i in range(1000))

    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert all(len(chunk.split()) <= 300 for chunk in chunks)


def test_overlap():
    """Consecutive chunks should share overlapping words."""
    text = " ".join(f"word{i}" for i in range(20))

    chunks = chunk_text(text, chunk_size=10, overlap=3)

    first_chunk = chunks[0].split()
    second_chunk = chunks[1].split()

    assert first_chunk[-3:] == second_chunk[:3]


def test_all_words_preserved():
    """Chunking must not lose any words."""
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=20, overlap=5)

    reconstructed = chunks[0].split()

    for chunk in chunks[1:]:
        reconstructed.extend(chunk.split()[5:])

    assert reconstructed == words


def test_invalid_chunk_size():
    """Chunk size must be positive."""
    with pytest.raises(ValueError):
        chunk_text("Hello world", chunk_size=0)


def test_invalid_overlap():
    """Overlap must be smaller than chunk_size."""
    with pytest.raises(ValueError):
        chunk_text("Hello world", chunk_size=10, overlap=10)

    with pytest.raises(ValueError):
        chunk_text("Hello world", chunk_size=10, overlap=-1)