"""Regression coverage for weekday/subject/time relationships, not one schedule."""
import os
from pathlib import Path
import pytest
from PIL import Image, ImageDraw
from rag.timetable import format_records, read_timetable, time_answer

MON1='Расписание | Понедельник | Основы Кибербезопасности Л 367 (12:00-13:00)'
MON2='Расписание | Понедельник | Основы Кибербезопасности Л 367 (13:00-14:00)'
SUN='Расписание | Воскресенье | Основы Кибербезопасности (09:00-10:00)'
TEXT='Header\n\n'+MON1+'\n\n'+MON2+'\n\nРасписание | Вторник | Базы данных (11:00-12:00)\n\n'+SUN


@pytest.mark.parametrize('question',[
    'во сколько у меня основы кибербезопасности в понедельник?',
    'во сколько у меня основы кб в понедельник?',
    'когда основы кибербезопасности в пн?',
])
def test_all_monday_slots_without_sunday(question):
    spans=time_answer(question,TEXT)
    assert [TEXT[a:b] for a,b in spans]==[MON1,MON2]


def test_without_day_returns_all_explicit_days():
    assert [TEXT[a:b] for a,b in time_answer('Когда основы кб?',TEXT)]==[MON1,MON2,SUN]


@pytest.mark.parametrize('question, expected', [
    ('В каком кабинете у меня урок в субботу и кто его ведёт?', [1, 2]),
    ('Кто ведёт базы данных во вторник?', [0]),
    ('В какой аудитории базы данных в субботу?', [1, 2]),
    ('Кто ведёт математику в субботу?', []),
    ('В каком кабинете урок в четверг?', []),
])
def test_class_details_stay_in_the_requested_day(question, expected):
    records = [
        'Расписание | Вторник | Базы данных Иванова А. А. П 105 (10:00-11:00)',
        'Расписание | Суббота | Базы данных Садыков Б. Б. Л 208 (10:00-11:00)',
        'Расписание | Суббота | Базы данных Садыков Б. Б. Л 208 (11:00-12:00)',
    ]
    text = '\n\n'.join(records)
    assert [text[a:b] for a, b in time_answer(question, text)] == [records[i] for i in expected]


def test_kazakh_subject_includes_both_slots():
    question = 'Дүйсенбі күні менің киберқауіпсіздік сабағы сағат нешеде?'
    text = TEXT.replace('Основы Кибербезопасности', 'Кибер қауіпсіздігі негіздері')
    records = [text[a:b] for a, b in time_answer(question, text)]
    assert len(records) == 2
    assert format_records(records, question) == (
        'Дүйсенбі · Кибер қауіпсіздігі негіздері Л 367 (12:00–14:00)'
    )


def test_list_keeps_distinct_subjects_and_breaks():
    records = [MON1, MON2, 'Расписание | Понедельник | Обработка языка (14:00-15:00)',
               'Расписание | Понедельник | Основы Кибербезопасности Л 367 (16:00-17:00)']
    text = '\n\n'.join([*records, SUN])
    question = 'Какие у меня уроки в понедельник?'
    matches = [text[a:b] for a, b in time_answer(question, text)]
    assert matches == records
    assert format_records(matches, question).split('\n\n') == [
        'Понедельник · Основы Кибербезопасности Л 367 (12:00–14:00)',
        'Понедельник · Обработка языка (14:00–15:00)',
        'Понедельник · Основы Кибербезопасности Л 367 (16:00–17:00)',
    ]


@pytest.mark.parametrize('question',[
    'во сколько основы кб в четверг?',
    'во сколько математика в понедельник?',
    'во сколько базы данных в понедельник?',
    'во сколько в понедельник?',
    'во сколько основы кибербезопасностика в понедельник?',
    'В каком кабинете основы кб?',
])
def test_absent_or_unsupported_facts_not_invented(question):
    assert time_answer(question,TEXT)==[]


def test_ocr_bad_time_is_not_used():
    for value in ['29:00-30:00','13:00-12:00','12:99-13:00']:
        assert not time_answer('когда основы кб в пн?',MON1.replace('12:00-13:00',value))


def grid():
    image=Image.new('RGB',(1400,600),'white')
    draw=ImageDraw.Draw(image)
    for x in (100,700,1300):draw.line((x,20,x,500),fill='#bbbbbb',width=2)
    for y in (100,200,300,400,500):draw.line((100,y,1300,y),fill='#bbbbbb',width=2)
    draw.text((120,130),'A sample lesson',fill='black')
    draw.text((720,230),'Another lesson',fill='black')
    return image


def test_grid_reads_actual_headers_without_assuming_seven_days():
    readings=iter(['Пн','Пт','Основы Кибербезопасности (12:00-13:00)','Базы данных (09:00-10:00)'])
    result=read_timetable(grid(),lambda crop,psm:next(readings))
    assert 'Понедельник | Основы Кибербезопасности' in result
    assert 'Пятница | Базы данных' in result
    assert 'Вторник' not in result


def test_grid_unreadable_headers_do_not_invent_days():
    assert read_timetable(grid(),lambda crop,psm:'?') is None


def test_normal_image_does_not_trigger_grid_ocr():
    assert read_timetable(Image.new('RGB',(600,400),'white'),lambda *args:pytest.fail('unexpected OCR')) is None


def test_pipeline_timetable_does_not_use_prose_qa(monkeypatch):
    from rag.pipeline import RAGPipeline
    from rag.document_loader import SourcePage
    from types import SimpleNamespace
    pipeline=RAGPipeline()
    pipeline.document_text=TEXT
    pipeline.document_pages=(SourcePage(1,0,len(TEXT),TEXT),)
    pipeline.retriever=SimpleNamespace(retrieve=lambda *a,**kw:[])
    pipeline.qa_model=SimpleNamespace(candidates=lambda *a,**kw:pytest.fail('prose QA must not guess table times'))
    result=pipeline.ask_detailed('Во сколько основы кб в понедельник?')
    assert result['answer']==MON1+'\n\n'+MON2
    assert result['pages']==[1]
    assert all(TEXT[s['start']:s['end']]==s['text'] for s in result['evidence_spans'])
    assert pipeline.ask_detailed('Во сколько основы кб в четверг?')['answer']==''


def test_actual_user_pdf_ocr_before_and_after(monkeypatch):
    """Opt-in private OCR regression; visual answers now have their own route."""
    path=os.environ.get('DOCUMENT_QA_SCHEDULE_PDF')
    if not path:pytest.skip('Set DOCUMENT_QA_SCHEDULE_PDF for private OCR acceptance')
    import rag.ocr as ocr
    from rag.document_loader import load_document_details
    from document_app.sources import inspect_source
    with monkeypatch.context() as patch:
        patch.setattr(ocr,'read_timetable',lambda *args, **kwargs:None)
        before=load_document_details(path,emit_warnings=False)
        assert 'кибербезопасности' not in before.text.lower()
    after=load_document_details(path,emit_warnings=False)
    spans=time_answer('во сколько основы кб в понедельник?',after.text)
    answer=' '.join(after.text[a:b] for a,b in spans)
    assert '12:00-13:00' in answer and '13:00-14:00' in answer
    assert '09:00-10:00' not in answer
    assert inspect_source(path)[0].visual
