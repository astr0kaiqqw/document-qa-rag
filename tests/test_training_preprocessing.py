import pytest
from transformers import AutoTokenizer
from rag.qa_model import DEFAULT_MODEL_PATH
from training.preprocessing import prepare_qa_features


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_PATH),local_files_only=True)


def example(context,answer):
    return dict(id=['e1'],question=['Which answer?'],context=[context],
                answers=[dict(text=[answer] if answer else [],answer_start=[context.index(answer)] if answer else [])])


@pytest.mark.parametrize('answer',['Saint Bernadette Soubirous','Қазақстан','Астана','AB-123/X'])
def test_gold_alignment_no_previous_token(tokenizer,answer):
    context='The answer is '+answer+'.'
    f,p=prepare_qa_features(example(context,answer),tokenizer,max_length=64,stride=16)
    s,e=f['start_positions'][0],f['end_positions'][0]
    assert context[p[0]['context_offsets'][s][0]:p[0]['context_offsets'][e][1]]==answer


def test_overflow_recovers_answer_near_end(tokenizer):
    text='routine weather measurements. '*100+'The answer is violet lantern.'
    f,p=prepare_qa_features(example(text,'violet lantern'),tokenizer,max_length=64,stride=16)
    assert len(p)>1 and p[0]['cls_label']
    assert any(r['gold_text']=='violet lantern' for r in p)
    assert all(len(ids)==64 for ids in f['input_ids'])


def test_empty_answers_and_invalid_annotation(tokenizer):
    f,p=prepare_qa_features(example('Evidence is absent.',''),tokenizer,max_length=64,stride=16)
    assert p[0]['cls_label'] and f['start_positions'][0]==0
    bad=example('Evidence is absent.','absent');bad['answers'][0]['answer_start']=[0]
    with pytest.raises(ValueError,match='Gold'):
        prepare_qa_features(bad,tokenizer,max_length=64,stride=16)


def test_multiple_references_choose_visible_one(tokenizer):
    row=example('alpha '+('weather '*80)+'omega','alpha')
    row['answers']=[dict(text=['alpha','omega'],answer_start=[0,row['context'][0].index('omega')])]
    _,p=prepare_qa_features(row,tokenizer,max_length=32,stride=8)
    assert p[0]['gold_text']=='alpha'
    assert p[-1]['gold_text']=='omega'
