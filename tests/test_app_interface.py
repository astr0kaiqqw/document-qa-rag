"""Essential UI states through Streamlit's actual upload widget API."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


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
