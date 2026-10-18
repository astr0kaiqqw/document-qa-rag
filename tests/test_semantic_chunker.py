import pytest

from rag.semantic_chunker import semantic_chunk_text


def test_empty_input():
    """Empty text should produce no chunks."""
    assert semantic_chunk_text("") == []
    assert semantic_chunk_text("   \n\t  ") == []


def test_short_input():
    """Very short text should remain a single chunk."""
    text = "Machine learning is interesting."
    result = semantic_chunk_text(text, chunk_size=180, overlap=36)

    assert result == [text]


def test_paragraph_boundaries_are_respected_when_possible():
    """Paragraph boundaries should remain intact when they fit within the target."""
    text = (
        "First paragraph about embeddings.\n\n"
        "Second paragraph about Word2Vec.\n\n"
        "Third paragraph about negative sampling."
    )

    result = semantic_chunk_text(text, chunk_size=180, overlap=36)

    assert result
    assert all(chunk.strip() for chunk in result)
    assert len(result) >= 1
    assert any("First paragraph" in chunk for chunk in result)
    assert any("Second paragraph" in chunk for chunk in result)
    assert any("Third paragraph" in chunk for chunk in result)


def test_sentences_are_not_split_across_chunk_boundaries():
    """A sentence should not be broken across chunks unless it itself is too long."""
    text = (
        "Sentence one is short. Sentence two is also short. "
        "Sentence three is short too. Sentence four is short as well."
    )

    chunks = semantic_chunk_text(text, chunk_size=20, overlap=4)

    assert chunks
    for chunk in chunks:
        for sentence in ["Sentence one is short.", "Sentence two is also short."]:
            if sentence in chunk:
                assert chunk.count(sentence) == 1


def test_long_paragraphs_are_segmented_without_losing_content():
    """Long paragraphs should split by sentence or word count without dropping content."""
    text = " ".join([f"Sentence {i} about embeddings and machine learning." for i in range(1, 80)])

    result = semantic_chunk_text(text, chunk_size=30, overlap=6)

    assert result
    assert all(chunk.strip() for chunk in result)
    reconstructed = " ".join(result)
    for token in ["Sentence 1 about embeddings and machine learning.", "Sentence 79 about embeddings and machine learning."]:
        assert token in reconstructed


def test_overlap_does_not_remove_content():
    """Overlap should preserve content without dropping sentences or words."""
    text = " ".join(f"word{i}" for i in range(1, 101))

    chunks = semantic_chunk_text(text, chunk_size=20, overlap=4)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    merged = " ".join(chunks)
    assert "word1" in merged
    assert "word100" in merged
    assert len(merged.split()) >= len(text.split())


def test_deterministic_output():
    """Repeated calls with the same input and parameters should produce identical chunks."""
    text = "Paragraph one.\n\nParagraph two with more detail. Paragraph three."

    first = semantic_chunk_text(text, chunk_size=30, overlap=6)
    second = semantic_chunk_text(text, chunk_size=30, overlap=6)

    assert first == second


def test_not_equivalent_to_fixed_word_chunking():
    """A deliberately multi-paragraph example should not reproduce legacy fixed-window chunking."""
    text = (
        "Alpha paragraph with a short first sentence. Beta paragraph with another sentence.\n\n"
        "Gamma paragraph with a distinct sentence. Delta paragraph with a final sentence."
    )

    result = semantic_chunk_text(text, chunk_size=18, overlap=4)

    assert result
    assert len(result) >= 2
    assert any("Alpha paragraph" in chunk for chunk in result)
    assert any("Gamma paragraph" in chunk for chunk in result)
    assert not all(chunk.startswith("Alpha") or chunk.startswith("Beta") for chunk in result)


def test_long_sentence_is_processed_without_loss():
    """A single very long sentence should be segmented by word count and not dropped."""
    long_sentence = " ".join([f"word{i}" for i in range(1, 200)])

    result = semantic_chunk_text(long_sentence, chunk_size=20, overlap=4)

    assert result
    assert all(chunk.strip() for chunk in result)
    reconstructed = " ".join(result)
    assert "word1" in reconstructed
    assert "word199" in reconstructed


def test_invalid_chunk_size():
    """Chunk size must be positive."""
    with pytest.raises(ValueError):
        semantic_chunk_text("Hello world", chunk_size=0)


def test_invalid_overlap():
    """Overlap must be smaller than chunk_size."""
    with pytest.raises(ValueError):
        semantic_chunk_text("Hello world", chunk_size=10, overlap=10)

    with pytest.raises(ValueError):
        semantic_chunk_text("Hello world", chunk_size=10, overlap=-1)
