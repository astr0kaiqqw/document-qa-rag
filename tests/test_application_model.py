"""One chosen application model, essential local smoke only (not a benchmark)."""
import pytest
import torch
from rag.qa_model import ExtractiveQAModel, APPLICATION_MODEL_PATH


def test_application_checkpoint_answer_and_null():
    if not (APPLICATION_MODEL_PATH/'model.safetensors').exists():
        pytest.skip('Run scripts/setup_models.py for application model smoke test')
    torch.set_num_threads(4)
    qa=ExtractiveQAModel()
    context='The capital of Kazakhstan is Astana. Almaty is its largest city.'
    candidate=qa.candidates('What is the capital of Kazakhstan?',context,top_n=1)[0]
    assert candidate['answer']=='Astana'
    assert context[candidate['start']:candidate['end']]=='Astana'
    assert candidate['null_margin']>0
    assert qa.answer('Who invented the telescope?',context)==''
