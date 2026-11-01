"""Real local-weight regressions. No alternate model or network fallback."""
from pathlib import Path
import pytest
import torch
from rag.qa_model import QAModel, DEFAULT_MODEL_PATH, valid_boundary
from rag.pipeline import RAGPipeline
from rag.embeddings import EmbeddingModel


@pytest.fixture(scope='module')
def qa():
    if not (DEFAULT_MODEL_PATH/'model.safetensors').exists():
        pytest.skip('Local fine-tuned checkpoint unavailable; real-weight quality checks not run')
    torch.set_num_threads(4)
    return QAModel()


@pytest.mark.parametrize('q,c,gold',[
    ('What carries oxygen in the blood?','Hemoglobin carries oxygen in the blood. White blood cells help fight infections.','Hemoglobin'),
    ('Как называется столица Казахстана?','Столица Казахстана называется Астана. Алматы является крупнейшим городом страны.','Астана'),
    ('Қазақстанның астанасы қай қала?','Қазақстанның астанасы — Астана. Алматы — үлкен қала.','Астана'),
])
def test_real_multilingual_source_spans(qa,q,c,gold):
    result=qa.candidates(q,c,top_n=1)[0]
    assert result['answer']==c[result['start']:result['end']]
    assert gold in result['answer']
    assert valid_boundary(c,result['start']) and valid_boundary(c,result['end'])


def test_real_checkpoint_and_tokenizer_compatible(qa):
    assert len(qa.tokenizer)==qa.model.config.vocab_size==119547
    assert qa.tokenizer.tokenize('What is the capital?')!=['[UNK]']*6


def test_real_offline_end_to_end_and_no_answer_status(qa,tmp_path):
    text='Hemoglobin carries oxygen in the blood. White blood cells help fight infections.'
    path=tmp_path/'medical.txt';path.write_text(text)
    pipe=RAGPipeline();pipe.qa_model=qa;pipe.embedding_model=EmbeddingModel()
    pipe.load_document(str(path))
    answer=pipe.ask_detailed('What carries oxygen in the blood?')
    assert answer['answer']=='Hemoglobin'
    assert pipe.document_text[answer['document_start']:answer['document_end']]=='Hemoglobin'
    assert answer['status']=='candidate'
    missing=pipe.ask_detailed('Who invented this telescope?')
    assert missing['no_answer_supported'] is False
    assert missing['status'] in {'candidate','no_admissible_span'}
