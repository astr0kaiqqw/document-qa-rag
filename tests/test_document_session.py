from io import BytesIO
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import numpy as np
import pytest
from docx import Document
from transformers import AutoTokenizer
from document_app.service import DocumentSession, friendly_error
from rag.embeddings import MODEL_NAME


class FakeQA:
    supports_no_answer=True
    def candidates(self,question,context,top_n=1):
        answer=next((name for name in ['Aurora','Meridian'] if name in context),'') if question!='Unknown?' else ''
        if not answer:return []
        start=context.index(answer)
        return [dict(answer=answer,start=start,end=start+len(answer),log_score=-.1)]


@pytest.fixture
def session():
    tok=AutoTokenizer.from_pretrained(MODEL_NAME,local_files_only=True)
    embedding=SimpleNamespace(model=SimpleNamespace(tokenizer=tok,max_seq_length=128),
        encode=lambda texts:np.tile(np.array([[1.,0.,0.]],dtype=np.float32),(len(texts),1)))
    models=SimpleNamespace(lock=RLock(),embedding=lambda:embedding,qa=lambda:FakeQA())
    session=DocumentSession(models)
    yield session
    session.close()


def test_multiple_documents_history_and_exact_citations(session):
    a=session.add_upload('station.txt',b'The station is named Aurora.')
    b=session.add_upload('lab.txt',b'The laboratory is named Meridian.')
    assert session.ask(a.key,'Which name?')['answer']=='Aurora'
    result=session.ask(b.key,'Which name?')
    assert result['answer']=='Meridian' and result['filename']=='lab.txt'
    assert result['answer'] in result['supporting_paragraph']
    assert len(a.history)==len(b.history)==1
    assert session.ask(a.key,'Unknown?')['display_status']=='no_reliable_answer'
    assert session.add_upload('station.txt',b'The station is named Aurora.') is a
    assert len(session.documents)==2


def test_docx_upload_and_safe_filename(session):
    doc=Document();doc.add_paragraph('The laboratory is named Meridian.')
    data=BytesIO();doc.save(data)
    record=session.add_upload('../../laboratory.docx',data.getvalue())
    assert record.name=='laboratory.docx'
    assert session.ask(record.key,'Which name?')['answer']=='Meridian'


@pytest.mark.parametrize('name,data',[('x.exe',b'file'),('x.txt',b'')])
def test_bad_uploads_rejected(session,name,data):
    with pytest.raises(ValueError):session.add_upload(name,data)
    assert not session.documents


def test_invalid_utf8_does_not_poison_session(session):
    with pytest.raises(UnicodeDecodeError):session.add_upload('bad.txt',b'\xff')
    assert not session.documents
    assert 'UTF-8' in friendly_error(UnicodeDecodeError('utf8',b'\xff',0,1,'invalid'))


def test_close_deletes_only_session_files(session):
    session.add_upload('a.txt',b'Aurora')
    directory=Path(session._temporary.name)
    assert directory.exists()
    session.close()
    assert not directory.exists() and not session.documents
