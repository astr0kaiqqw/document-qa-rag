import pytest
from transformers import AutoTokenizer
from rag.chunker import token_chunk_text
from rag.embeddings import MODEL_NAME, EmbeddingModel


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)


@pytest.mark.parametrize('text',[
    'A sentence about embeddings. Another sentence about retrieval. '*100,
    'Қазақстанның астанасы — Астана. Машинное обучение работает локально. '*50,
    'x=3.14159; ∂L/∂w; AB-123/X. '*60,
    'one\n\nTwo  words.\nThree words. '*70,
])
def test_budget_coverage_and_source_coordinates(tokenizer,text):
    chunks=token_chunk_text(text,tokenizer,max_tokens=64,overlap_tokens=12)
    covered=set()
    for chunk in chunks:
        assert chunk.text==text[chunk.start:chunk.end]
        assert len(tokenizer(chunk.text)['input_ids'])==chunk.token_count<=64
        covered.update(range(chunk.start,chunk.end))
    assert all(i in covered for i,c in enumerate(text) if not c.isspace())
    assert all(a.start<b.start and a.end<b.end for a,b in zip(chunks,chunks[1:]))
    assert any(a.end>b.start for a,b in zip(chunks,chunks[1:]))


def test_sentence_boundary_preference(tokenizer):
    text='Alpha beta gamma. Delta epsilon zeta. One two three four five six seven eight nine ten.'
    chunks=token_chunk_text(text,tokenizer,max_tokens=16,overlap_tokens=2)
    assert chunks[0].text.endswith('.')


def test_overlong_word_rejected(tokenizer):
    with pytest.raises(ValueError,match='word'):
        token_chunk_text('а'*300,tokenizer,max_tokens=8,overlap_tokens=0)


@pytest.mark.parametrize('size,overlap',[(2,0),(16,14),(16,-1)])
def test_invalid_budgets(tokenizer,size,overlap):
    with pytest.raises(ValueError):token_chunk_text('text',tokenizer,max_tokens=size,overlap_tokens=overlap)


def test_empty(tokenizer):
    assert token_chunk_text('  ',tokenizer)==[]


def test_embedding_rejects_silent_document_and_query_truncation():
    model=EmbeddingModel()
    with pytest.raises(ValueError,match='exceeds'):
        model.encode(['content '*200])


def test_no_overlap_keeps_ignored_format_characters_between_tokens(tokenizer):
    text='alpha\u200b beta\u200b gamma\u200b delta\u200b epsilon\u200b zeta'
    chunks=token_chunk_text(text,tokenizer,max_tokens=6,overlap_tokens=0,sentence_boundaries=False)
    covered={i for c in chunks for i in range(c.start,c.end)}
    assert all(i in covered for i,char in enumerate(text) if not char.isspace())
