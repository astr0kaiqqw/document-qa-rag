"""Hybrid paragraph-aware and sentence-aware chunking for semantic retrieval."""

import re


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving sentence punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines while preserving paragraph order."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph and paragraph.strip()
    ]
    return paragraphs or ([text.strip()] if text.strip() else [])


def _split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentence-like units while keeping punctuation."""
    normalized = _normalize_whitespace(paragraph)
    if not normalized:
        return []

    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = [
        sentence.strip()
        for sentence in sentence_pattern.split(normalized)
        if sentence and sentence.strip()
    ]
    return sentences or [normalized]


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Split a sentence by word count only when it exceeds the target chunk size."""
    words = sentence.split()
    if len(words) <= chunk_size:
        return [sentence.strip()] if sentence.strip() else []

    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), chunk_size)
        if words[start:start + chunk_size]
    ]


def _build_semantic_units(text: str, chunk_size: int) -> list[str]:
    """Create semantic units in original order without flattening the document."""
    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        for sentence in _split_sentences(paragraph):
            units.extend(_split_long_sentence(sentence, chunk_size))
    return units


def _last_units_for_overlap(units: list[str], overlap: int) -> list[str]:
    """Take the trailing semantic units needed for overlap."""
    if overlap <= 0 or not units:
        return []

    overlap_words = 0
    overlap_units: list[str] = []
    for unit in reversed(units):
        unit_word_count = len(unit.split())
        if overlap_words >= overlap:
            break
        overlap_units.append(unit)
        overlap_words += unit_word_count

    return list(reversed(overlap_units))


def semantic_chunk_text(
    text: str,
    chunk_size: int = 180,
    overlap: int = 36,
) -> list[str]:
    """Preserve semantic boundaries while assembling chunks.

    Chunks are assembled greedily from paragraph/sentence-level units. A single sentence
    is only split by word count when its own length exceeds the chunk size. Overlap is
    implemented by reusing complete trailing semantic units from the previous chunk when
    possible, rather than flattening the text into one global word stream and re-splitting.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    if not text or not text.strip():
        return []

    units = _build_semantic_units(text, chunk_size)
    if not units:
        return []

    chunk_units: list[list[str]] = []
    current: list[str] = []
    current_word_count = 0

    for unit in units:
        unit_word_count = len(unit.split())

        if current and current_word_count + unit_word_count > chunk_size:
            chunk_units.append(current)
            current = []
            current_word_count = 0

        current.append(unit)
        current_word_count += unit_word_count

    if current:
        chunk_units.append(current)

    if not chunk_units:
        return []

    if overlap == 0:
        return [
            _normalize_whitespace(" ".join(chunk))
            for chunk in chunk_units
            if _normalize_whitespace(" ".join(chunk))
        ]

    chunks: list[str] = []
    for index, current_chunk_units in enumerate(chunk_units):
        prev_overlap = [] if index == 0 else _last_units_for_overlap(chunk_units[index - 1], overlap)
        merged_units = prev_overlap + current_chunk_units
        merged_text = _normalize_whitespace(" ".join(merged_units))
        if merged_text:
            chunks.append(merged_text)

    return chunks
