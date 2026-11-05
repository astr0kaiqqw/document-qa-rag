import json
from pathlib import Path
import pytest
from quality.evaluate import score, aggregate


def test_squad_style_metrics_and_empty_gold():
    assert score('The Paris!','Paris','The Paris!')['em']==1
    assert score('one two','two three','one two')['f1']==.5
    assert score('','', '')['f1']==1
    assert score('in 1980','','in 1980')['em']==0
    assert score('##bin','Hemoglobin','Hemoglobin')['invalid'] is True


def test_false_answers_and_refusals_are_separate():
    rows=[dict(reference='Paris',**score('','Paris','Paris')),
          dict(reference='Astana',**score('Astana','Astana','Astana')),
          dict(reference='',**score('unrelated','','unrelated')),
          dict(reference='',**score('','',''))]
    result=aggregate(rows)
    assert result['false_refusal_rate']==.5
    assert result['false_answer_rate']==.5
    assert result['n']==result['no_answer_n']==2


@pytest.mark.parametrize('filename',['audit_questions.json','independent_questions.json'])
def test_evaluation_fixtures_have_verbatim_gold_and_unique_ids(filename):
    data=json.loads((Path(__file__).resolve().parents[1]/'quality/data'/filename).read_text())
    assert len({r['id'] for r in data})==len(data)
    assert all(not r['answer'] or r['answer'] in r['context'] for r in data)
