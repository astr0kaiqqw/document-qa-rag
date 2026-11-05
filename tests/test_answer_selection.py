from unittest.mock import Mock
import pytest
from rag.answer_selection import rank_candidates
from rag.pipeline import RAGPipeline


def candidate(answer,rank,log_score,start=0):
    return dict(answer=answer,source=answer,start=0,end=len(answer),chunk_index=rank,
                document_start=start,document_end=start+len(answer),retrieval_rank=rank,
                retrieval_score=1/rank,log_score=log_score)


def test_selection_can_choose_lower_retrieval_evidence():
    rows=[candidate('Paris',1,-6),candidate('Астана',2,-0.5,100)]
    assert rank_candidates(rows,'retrieval')[0]['answer']=='Paris'
    assert rank_candidates(rows,'qa')[0]['answer']=='Астана'
    assert all('qa_rank' in r and 'selection_score' in r for r in rank_candidates(rows,'rrf'))


def test_deduplicates_coordinates_not_repeated_answers():
    rows=[candidate('Paris',2,-3),candidate('Paris',1,-2),candidate('Paris',3,-1,100)]
    result=rank_candidates(rows)
    assert len(result)==2
    assert result[0]['retrieval_rank']==1


def test_fabricated_and_nan_candidates_rejected():
    valid=candidate('evidence',1,-1)
    invalid=dict(valid,answer='fabricated')
    assert rank_candidates([invalid])==[]
    assert rank_candidates([dict(valid,log_score=float('nan'))])==[]


def test_unknown_strategy():
    with pytest.raises(ValueError):rank_candidates([],'random')


def test_empty_retrieval_is_transparent_no_evidence():
    pipe=RAGPipeline();pipe.retriever=Mock();pipe.retriever.retrieve.return_value=[]
    result=pipe.ask_detailed('Is there evidence?')
    assert result['answer']=='' and result['status']=='no_evidence'
    assert result['no_answer_supported'] is False


def test_pipeline_rejects_non_source_answer():
    pipe=RAGPipeline();pipe.retriever=Mock();pipe.retriever.retrieve.return_value=[dict(chunk='The source is silent.',score=.8)]
    pipe.qa_model=Mock(spec=['answer']);pipe.qa_model.answer.return_value='invented claim'
    assert pipe.ask_detailed('What happened?')['status']=='no_admissible_span'
