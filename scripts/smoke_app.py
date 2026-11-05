"""Essential Streamlit acceptance smoke: actual upload widgets and real local models."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from io import BytesIO
import json
from pathlib import Path
import sys
from docx import Document
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def button(at,label):
    return next(b for b in at.button if b.label==label)


def main():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180).run()
    assert not at.exception,at.exception
    pdf=ROOT/'uploads/embedding.pdf'
    if not pdf.exists():raise FileNotFoundError('Real acceptance PDF missing: uploads/embedding.pdf')
    at.file_uploader[0].set_value([('embedding.pdf',pdf.read_bytes(),'application/pdf')]).run()
    button(at,'Обработать файлы').click().run()
    assert not at.exception,at.exception
    session=at.session_state['documents'];pdf_key=next(iter(session.documents))
    at.text_input(key='question').set_value('Which classifier is used to distinguish positive and negative examples?')
    button(at,'Найти ответ').click().run()
    assert not at.exception,at.exception
    result=session.documents[pdf_key].history[-1]
    assert result['answer'],result
    assert 'logistic regression' in result['answer'].lower(),result['answer']
    assert result['filename']=='embedding.pdf' and result['pages'],result
    pipeline=session.documents[pdf_key].pipeline
    assert pipeline.document_text[result['document_start']:result['document_end']]==result['answer']
    assert result['answer'] in result['supporting_paragraph']
    assert any(result['answer'] in pipeline.document_text[p.start:p.end] for p in pipeline.document_pages if p.number in result['pages'])
    acceptance={'pdf_answer':{k:result[k] for k in ['question','answer','filename','pages','display_status']}}
    at.text_input(key='question').set_value('What is the password for my email account?')
    button(at,'Найти ответ').click().run()
    missing=session.documents[pdf_key].history[-1]
    assert missing['display_status'] in {'uncertain_candidate','no_reliable_answer'}
    acceptance['unanswerable']={k:missing[k] for k in ['answer','status','display_status','no_answer_supported']}
    data=b'The research station is named Aurora. The access code is violet lantern.'
    at.file_uploader[0].upload('station.txt',data,'text/plain').run()
    button(at,'Обработать файлы').click().run()
    assert len(session.documents)==2
    txt_key=next(k for k,d in session.documents.items() if d.name=='station.txt')
    assert at.selectbox[0].value==txt_key
    at.text_input(key='question').set_value('What is the station named?')
    button(at,'Найти ответ').click().run()
    txt_result=session.documents[txt_key].history[-1]
    assert 'Aurora' in txt_result['answer'],txt_result
    acceptance['second_document']={k:txt_result[k] for k in ['answer','filename','display_status']}
    at.selectbox[0].select(pdf_key).run()
    assert at.selectbox[0].value==pdf_key and len(session.documents[pdf_key].history)==2
    doc=Document();doc.add_paragraph('The laboratory is named Meridian. It opened in 2020.')
    stream=BytesIO();doc.save(stream)
    at.file_uploader[0].upload('laboratory.docx',stream.getvalue(),'application/vnd.openxmlformats-officedocument.wordprocessingml.document').run()
    button(at,'Обработать файлы').click().run()
    assert len(session.documents)==3
    at.text_input(key='question').set_value('What is the laboratory named?')
    button(at,'Найти ответ').click().run()
    docx_key=at.selectbox[0].value;docx_result=session.documents[docx_key].history[-1]
    assert 'Meridian' in docx_result['answer'],docx_result
    acceptance['docx_document']={k:docx_result[k] for k in ['answer','filename','display_status']}
    assert not at.exception,at.exception
    button(at,'Сбросить сессию').click().run()
    assert not at.session_state['documents'].documents
    assert not at.exception,at.exception
    acceptance.update(upload_types=['PDF','TXT','DOCX'],switch_document=True,multiple_questions=True,reset=True,streamlit_exceptions=0)
    output=ROOT/'reports/local_results/application_smoke.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(acceptance,ensure_ascii=False,indent=2))
    print(json.dumps(acceptance,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
