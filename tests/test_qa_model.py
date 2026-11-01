"""Decoder regression tests with real local tokenizer and controlled logits."""
from types import SimpleNamespace
import pytest
import torch
from transformers import AutoTokenizer
from rag.qa_model import QAModel, DEFAULT_MODEL_PATH, valid_boundary


class LogitModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.scores = scores

    def forward(self, input_ids, **kwargs):
        start = torch.full(input_ids.shape, -20.)
        end = start.clone()
        start[:, 0] = end[:, 0] = 100  # Special token must never become an answer.
        for token, (s, e) in self.scores.items():
            start[input_ids == token] = s
            end[input_ids == token] = e
        return SimpleNamespace(start_logits=start, end_logits=end)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_PATH), local_files_only=True)


def decoder(tokenizer, start_text, end_text=None, **options):
    qa = object.__new__(QAModel)
    qa.tokenizer = tokenizer
    first = tokenizer(start_text, add_special_tokens=False)['input_ids'][0]
    last = tokenizer(end_text or start_text, add_special_tokens=False)['input_ids'][-1]
    qa.model = LogitModel({first: (20., -15.), last: (-15., 20.)} if first != last else {first: (20.,20.)})
    qa.max_length = options.get('max_length', 64)
    qa.max_answer_tokens = options.get('max_answer_tokens',30)
    qa.stride = options.get('stride',16)
    qa.whole_words = True
    return qa


@pytest.mark.parametrize('answer', ['Guido van Rossum','Астана','Қазақстан','Zeta-Q7','AB-123/X'])
def test_offsets_preserve_multilingual_and_identifiers(tokenizer, answer):
    qa = decoder(tokenizer, answer, answer)
    context = f'The value is {answer}. End.'
    result = qa.candidates('Which value?', context, top_n=1)[0]
    assert result['answer'] == answer
    assert context[result['start']:result['end']] == answer


def test_partial_wordpiece_is_disallowed(tokenizer):
    qa = decoder(tokenizer, 'Hemoglobin')
    subword = tokenizer('Hemoglobin', add_special_tokens=False)['input_ids'][-1]
    qa.model = LogitModel({subword: (30.,30.)})
    assert qa.answer('What carries oxygen?', 'Hemoglobin carries oxygen.') == 'Hemoglobin'


def test_long_context_and_deduplication(tokenizer):
    qa = decoder(tokenizer, 'violet', 'lantern')
    context = 'Routine weather measurements. ' * 100 + 'The code is violet lantern.'
    candidates = qa.candidates('Which code?', context)
    assert candidates[0]['answer'] == 'violet lantern'
    assert candidates[0]['window'] > 0
    assert len({(c['start'],c['end']) for c in candidates}) == len(candidates)


def test_long_answer_budget_is_configurable(tokenizer):
    answer = 'alpha ' + 'routine '*35 + 'omega'
    qa = decoder(tokenizer, 'alpha', 'omega',max_answer_tokens=64,max_length=128)
    assert qa.answer('Which sequence?', answer) == answer


@pytest.mark.parametrize('text,cut', [('Hemoglobin',6),('Қазақстан',4),('русский',3),('Zeta-Q7',5),('AB-123/X',7),('e\u0301clair',1)])
def test_unicode_boundaries(text,cut):
    assert not valid_boundary(text,cut)


@pytest.mark.parametrize('question,context', [('', 'text'),('q',''),('q','   '),(None,'text')])
def test_invalid_inputs(tokenizer,question,context):
    with pytest.raises(ValueError):decoder(tokenizer,'value').answer(question,context)


def test_question_cannot_consume_all_context_budget(tokenizer):
    with pytest.raises(ValueError,match='budget'):
        decoder(tokenizer,'value',max_length=8).answer('many words '*20,'value')
