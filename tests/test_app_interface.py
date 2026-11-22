"""Essential UI states through Streamlit's actual upload widget API."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
from document_app import speech
from document_app.service import DocumentSession
from document_app.speech import LocalSpeech


def test_empty_app_upload_error_and_reset():
    root=Path(__file__).resolve().parents[1]
    at=AppTest.from_file(str(root/'app.py'),default_timeout=120).run()
    assert not at.exception
    assert len(at.file_uploader)==1
    process=next(b for b in at.button if b.label=='Обработать файлы')
    assert process.disabled
    at.file_uploader[0].set_value([('bad.txt',b'\xff','text/plain')]).run()
    next(b for b in at.button if b.label=='Обработать файлы').click().run()
    assert any('UTF-8' in error.value for error in at.error)
    assert not at.session_state['documents'].documents
    assert not at.exception
    next(b for b in at.button if b.label=='Сбросить сессию').click().run()
    assert not at.session_state['documents'].documents
    assert not at.exception


def test_transcribed_question_is_submitted_without_typing(monkeypatch):
    transcript = 'Какие у меня уроки в понедельник?'
    submitted = []

    def transcribe(self, data, language):
        assert language == 'ru'
        return transcript

    monkeypatch.setattr(LocalSpeech, 'transcribe', transcribe)
    monkeypatch.setattr(DocumentSession, 'ask',
                        lambda self, key, question, **options: submitted.append(question))

    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root/'app.py'), default_timeout=120).run()
    at.file_uploader[0].set_value([('schedule.txt', b'Cybersecurity on Monday.', 'text/plain')]).run()
    next(b for b in at.button if b.label == 'Обработать файлы').click().run()
    old_widget = at.text_input[0].id
    next(menu for menu in at.selectbox if menu.label == 'Язык записи').select('Русский').run()
    at.audio_input[0].set_value(('question.wav', b'RIFF', 'audio/wav')).run()
    next(b for b in at.button if b.label == 'Распознать запись').click().run()

    assert at.text_input[0].id != old_widget
    assert at.text_input[0].value == transcript
    old_audio = at.audio_input[0].id
    next(b for b in at.button if b.label == 'Сбросить запись').click().run()
    assert at.audio_input[0].id != old_audio
    assert at.audio_input[0].value is None
    assert next(b for b in at.button if b.label == 'Распознать запись').disabled
    assert at.text_input[0].value == transcript
    next(b for b in at.button if b.label == 'Найти ответ').click().run()
    assert submitted == [transcript]

    edited = 'Во сколько начинаются уроки?'
    at.text_input[0].set_value(edited)
    next(b for b in at.button if b.label == 'Найти ответ').click().run()
    assert submitted == [transcript, edited]
    at.audio_input[0].set_value(('new-question.wav', b'RIFF', 'audio/wav')).run()
    assert not next(b for b in at.button if b.label == 'Распознать запись').disabled

    class ReloadedSpeech(LocalSpeech):
        def transcribe(self, data, language):
            assert language == 'ru'
            return 'В какой аудитории занятие?'

    monkeypatch.setattr(speech, 'LocalSpeech', ReloadedSpeech)
    next(b for b in at.button if b.label == 'Распознать запись').click().run()
    assert at.text_input[0].value == 'В какой аудитории занятие?'
    assert not at.error
    assert not at.exception
