"""Split extracted document text into overlapping word chunks."""


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """Split text into chunks with a configurable word overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class TextChunk:
    """Coordinates refer to the exact extracted document, before chunking."""
    text: str
    start: int
    end: int
    token_count: int


def _inside_word(text: str, position: int) -> bool:
    def word(c):
        return c.isalnum() or c == '_' or unicodedata.category(c).startswith('M')
    return 0 < position < len(text) and word(text[position - 1]) and word(text[position])


def token_chunk_text(text, tokenizer, *, max_tokens=128, overlap_tokens=24,
                     sentence_boundaries=True) -> list[TextChunk]:
    """Partition exact source substrings within the *encoded* limit including specials.

    Overlap is a target in content tokens. Sentence ends are preferred in the
    latter half of a window. A word larger than the entire budget raises instead
    of silently losing its suffix. Every non-whitespace character remains covered.
    """
    special = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = max_tokens - special
    if capacity <= 0 or not 0 <= overlap_tokens < capacity:
        raise ValueError('Invalid embedding token budget or overlap')
    if not text or not text.strip():
        return []
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Token chunking requires a fast tokenizer with offsets')
    encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, verbose=False)
    offsets = [(a,b) for a,b in encoded['offset_mapping'] if b>a]
    if not offsets:
        raise ValueError('Document has no tokens with source offsets')
    boundaries = {match.end() for match in re.finditer(r'[.!?。！？](?:[\"\'”’)]*)?(?=\s|$)|\n\s*\n',text)}
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + capacity, len(offsets))
        # Never cut the middle of a regular word at the right boundary.
        while end > start and _inside_word(text, offsets[end - 1][1]):
            end -= 1
        if end == start:
            raise ValueError('A single word exceeds the embedding token budget')
        if sentence_boundaries and end < len(offsets):
            candidates = [i for i in range(start + max(1,(end-start)//2),end+1)
                          if offsets[i-1][1] in boundaries]
            if candidates:
                end = candidates[-1]
        a = 0 if start == 0 else offsets[start][0]
        b = len(text) if end == len(offsets) else offsets[end-1][1]
        # Re-tokenization of a substring can differ at boundaries. Verify, never assume.
        while end > start:
            count = len(tokenizer(text[a:b], add_special_tokens=True, truncation=False,
                                  verbose=False)['input_ids'])
            if count <= max_tokens:
                break
            end -= 1
            while end > start and _inside_word(text,offsets[end-1][1]):
                end -= 1
            if end > start:
                b=offsets[end-1][1]
        if end <= start:
            raise ValueError('A single word exceeds the embedding token budget')
        chunks.append(TextChunk(text[a:b],a,b,count))
        if end == len(offsets):
            break
        next_start = max(start+1,end-overlap_tokens)
        while next_start < end and _inside_word(text,offsets[next_start][0]):
            next_start += 1
        if next_start == end and _inside_word(text,offsets[next_start][0]):
            raise ValueError('Cannot start an embedding chunk inside a word')
        start = next_start
    return chunks
