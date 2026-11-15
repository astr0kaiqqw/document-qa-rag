from io import BytesIO
from types import SimpleNamespace
from threading import RLock
from pathlib import Path
import base64
import json
from urllib.error import HTTPError,URLError
import pytest
from PIL import Image
from pypdf import PdfWriter,PdfReader
from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
from document_app.vision import LocalVision,VisionError,encode_views,NoRedirect
from document_app.sources import inspect_source,render_source,choose_pages
from document_app.service import DocumentSession


def text_pdf(path,text='The station is named Aurora and is located in the northern valley.'):
    w=PdfWriter();p=w.add_blank_page(width=600,height=800)
    p[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})})})
    stream=DecodedStreamObject();stream.set_data(f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET'.encode())
    p[NameObject('/Contents')]=stream;w.write(path)


@pytest.mark.parametrize('url',['https://ollama.com','http://example.com','http://127.0.0.1.evil.org','http://user:pass@localhost:11434','http://localhost:11434/path','http://localhost:11434?redirect=x'])
def test_only_loopback(url):
    with pytest.raises(VisionError):LocalVision(base_url=url)


def test_no_cloud_or_redirect():
    with pytest.raises(VisionError):LocalVision(model='qwen3-vl:235b-cloud')
    with pytest.raises(VisionError):NoRedirect().redirect_request(None,None,None,None,None,None)


def test_remote_alias_rejected(monkeypatch):
    client=LocalVision();responses=iter([{'models':[{'name':client.model}]},{'capabilities':['vision'],'remote_host':'https://ollama.com'}])
    monkeypatch.setattr(client,'_request',lambda *a,**k:next(responses))
    with pytest.raises(VisionError,match='Удалённая'):client.check()


def test_model_missing_and_not_vision(monkeypatch):
    client=LocalVision();monkeypatch.setattr(client,'_request',lambda *a,**k:{'models':[]})
    with pytest.raises(VisionError,match='отсутствует'):client.check()
    monkeypatch.setattr(client,'_request',lambda path,*a,**k:{'models':[{'name':client.model}]} if path=='/api/tags' else {'capabilities':['completion']})
    with pytest.raises(VisionError,match='не поддерживает'):client.check()


@pytest.mark.parametrize('error,match',[(URLError('refused'),'не запущена'),(TimeoutError(),'время'),(HTTPError('local',500,'bad',{},BytesIO(b'out of memory')),'памяти')])
def test_service_errors(monkeypatch,error,match):
    client=LocalVision()
    def fail(*a,**k):raise error
    monkeypatch.setattr(client.http,'open',fail)
    with pytest.raises(VisionError,match=match):client._request('/api/chat',{})


def test_views_keep_overview_and_details():
    image=Image.new('RGB',(3800,1600),'white')
    encoded=encode_views(image)
    assert len(encoded)==3 and image.size==(3800,1600)
    for value in encoded:
        with Image.open(BytesIO(base64.b64decode(value))) as prepared:assert max(prepared.size)<=1280


def test_missing_fact_and_unfinished_response(monkeypatch):
    client=LocalVision();monkeypatch.setattr(client,'check',lambda:'ready')
    monkeypatch.setattr(client,'_request',lambda *a,**k:{'done':True,'message':{'content':json.dumps({'answer':'Not visible','not_found':True})}})
    result=client.answer('Absent?',Image.new('RGB',(100,100)))
    assert result['answer']=='' and result['not_found']
    monkeypatch.setattr(client,'_request',lambda *a,**k:{'done':False})
    with pytest.raises(VisionError,match='не завершён'):client.answer('q',Image.new('RGB',(100,100)))


def test_page_level_classification_and_render(tmp_path):
    native=tmp_path/'native.pdf';text_pdf(native)
    scan=tmp_path/'scan.pdf';Image.new('RGB',(500,700),'white').save(scan,'PDF')
    mixed=tmp_path/'mixed.pdf';writer=PdfWriter();writer.append(PdfReader(native));writer.append(PdfReader(scan));writer.write(mixed)
    routes=inspect_source(mixed)
    assert [p.visual for p in routes]==[False,True]
    assert render_source(mixed,2).width>0
    with pytest.raises(ValueError,match='вне'):render_source(mixed,3)


def test_page_selection_no_ocr_requires_choice():
    from document_app.sources import PageRoute
    routes=[PageRoute(n,True,'scan') for n in range(1,10)]
    p=SimpleNamespace(retriever=None)
    with pytest.raises(ValueError,match='Укажите'):choose_pages(p,routes,'q')
    assert choose_pages(p,routes,'q',8)==[8]


def test_image_route_does_not_need_ocr_or_qa(tmp_path,monkeypatch):
    import document_app.service as svc
    def no_load(*args,**kwargs):raise ValueError('Document contains no extractable text')
    monkeypatch.setattr(svc.RAGPipeline,'load_document',no_load)
    vision=SimpleNamespace(model='test-local',answer=lambda q,image:dict(answer='A blue square',note='',not_found=False))
    models=SimpleNamespace(lock=RLock(),embedding=lambda:None,vision=lambda:vision,qa=lambda:pytest.fail('QA called on image'))
    session=DocumentSession(models)
    output=BytesIO();Image.new('RGB',(100,100),'blue').save(output,'PNG')
    try:
        doc=session.add_upload('drawing.png',output.getvalue())
        result=session.ask(doc.key,'What is visible?')
        assert result['answer_method']=='vision' and result['pages']==[1]
        assert Path(doc.source_path).read_bytes()==output.getvalue()
        assert result['source']=='' and 'document_start' not in result
        with pytest.raises(ValueError):session.ask(doc.key,'q',mode='text')
    finally:session.close()


def test_corrupt_image_not_ready():
    session=DocumentSession(SimpleNamespace(lock=RLock()))
    try:
        with pytest.raises(Exception):session.add_upload('broken.png',b'not an image')
        assert not session.documents
    finally:session.close()


def test_explicit_text_page_not_replaced_by_retrieval_hit():
    from rag.pipeline import RAGPipeline
    from rag.document_loader import SourcePage
    p=RAGPipeline(context_expansion=True)
    p.document_text='Native answer.\n\nUnrelated visual OCR.'
    p.document_pages=(SourcePage(1,0,14,'Native answer.'),SourcePage(2,16,len(p.document_text),'Unrelated visual OCR.'))
    p.retriever=SimpleNamespace(retrieve=lambda *a,**k:[dict(chunk='Unrelated visual OCR.',index=-1,score=.99)])
    seen=[]
    def candidates(q,text,**kwargs):
        seen.append(text);return [dict(answer='Native',start=0,end=6,log_score=-1)]
    p.qa_model=SimpleNamespace(candidates=candidates,supports_no_answer=True)
    result=p.ask_detailed('question',allowed_pages=[1],source_pages=[1])
    assert seen==['Native answer.'] and result['pages']==[1]


def test_visual_result_in_real_streamlit_widgets(monkeypatch):
    import document_app.service as svc
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(LocalVision,'check',lambda self:'local test ready')
    def no_text(*args,**kwargs):raise ValueError('Document contains no extractable text')
    monkeypatch.setattr(svc.RAGPipeline,'load_document',no_text)
    vision=SimpleNamespace(model='local-test',answer=lambda q,image:dict(answer='Blue square',note='Blue square',not_found=False))
    models=SimpleNamespace(lock=RLock(),embedding=lambda:None,vision=lambda:vision,qa=lambda:pytest.fail('not QA'))
    s=DocumentSession(models);output=BytesIO();Image.new('RGB',(100,100),'blue').save(output,'PNG')
    try:
        doc=s.add_upload('scene.png',output.getvalue())
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=120)
        at.session_state['documents']=s;at.session_state['selected_document']=doc.key;at.run()
        at.text_input[0].set_value('What is visible?')
        next(b for b in at.button if b.label=='Найти ответ').click().run()
        assert not at.exception
        assert s.documents[doc.key].history[-1]['answer_method']=='vision'
        assert any(c.value=='Локальная модель по изображению' for c in at.caption)
        assert any('Сгенерированный' in c.value for c in at.caption)
    finally:s.close()


def test_partial_text_layer_is_supplemented_by_ocr(tmp_path,monkeypatch):
    import rag.document_loader as loader
    path=tmp_path/'partial.pdf';path.write_bytes(b'fixture')
    monkeypatch.setattr(loader,'PdfReader',lambda _:SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda:'Small footer')]))
    calls=[]
    def recognize(path,index):calls.append(index);return 'Visible diagram labels in the image'
    monkeypatch.setattr(loader,'pdf_page_text',recognize)
    doc=loader.load_document_details(path,ocr_pages={1},emit_warnings=False)
    assert calls==[0] and 'Small footer' in doc.text and 'Visible diagram' in doc.text
