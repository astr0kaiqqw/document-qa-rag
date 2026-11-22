"""OCR routing/provenance and bounded, local voice workflow."""
from io import BytesIO
from types import SimpleNamespace
import wave
import pytest
from PIL import Image, ImageDraw, ImageFont
from rag import document_loader as loader
from document_app.speech import LocalSpeech, SpeechError


def wav(seconds=1):
    output=BytesIO()
    with wave.open(output,'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
        audio.writeframes(b'\0\0' * int(seconds*16000))
    return output.getvalue()


def test_mixed_pdf_routes_only_scanned_page(tmp_path, monkeypatch):
    path=tmp_path/'mixed.pdf';path.write_bytes(b'fixture')
    monkeypatch.setattr(loader,'PdfReader',lambda _:SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda:'First text page'),SimpleNamespace(extract_text=lambda:'')]))
    called=[]
    def ocr(path,index):
        called.append(index);return 'Базы данных — кабинет 402'
    monkeypatch.setattr(loader,'pdf_page_text',ocr)
    result=loader.load_document_details(path,emit_warnings=False)
    assert called==[1]
    page=result.pages[1]
    assert page.number==2 and result.text[page.start:page.end]=='Базы данных — кабинет 402'


def test_page_limit_before_ocr(tmp_path,monkeypatch):
    path=tmp_path/'many.pdf';path.write_bytes(b'fixture')
    monkeypatch.setattr(loader,'PdfReader',lambda _:SimpleNamespace(pages=[None]*101))
    with pytest.raises(ValueError,match='100 страниц'):loader.load_document_details(path)


@pytest.mark.parametrize('data',[b'',b'broken',wav(61),wav(0)])
def test_audio_limits(data,tmp_path):
    with pytest.raises(SpeechError):LocalSpeech(tmp_path).transcribe(data)


@pytest.mark.parametrize('language', ['ru', 'kk'])
def test_voice_result_and_silence(tmp_path, language):
    speech=LocalSpeech(tmp_path)
    def transcribe(audio, **options):
        assert options['language'] == language
        return [SimpleNamespace(text=' В каком кабинете? ')], None
    speech.model=SimpleNamespace(transcribe=transcribe)
    assert speech.transcribe(wav(),language)=='В каком кабинете?'
    speech.model=SimpleNamespace(transcribe=lambda *a,**k:([],None))
    with pytest.raises(SpeechError,match='Речь не обнаружена'):speech.transcribe(wav())


def test_missing_speech_model(tmp_path):
    with pytest.raises(SpeechError,match='setup_speech'):LocalSpeech(tmp_path).transcribe(wav())


def test_real_image_and_scan_ocr(tmp_path):
    import shutil
    if not shutil.which('tesseract'):pytest.skip('Tesseract not installed')
    image=Image.new('RGB',(1700,400),'white')
    draw=ImageDraw.Draw(image)
    from pathlib import Path
    fonts=['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    font_path=next((p for p in fonts if Path(p).exists()),None)
    if not font_path:pytest.skip('No Cyrillic test font installed')
    font=ImageFont.truetype(font_path,48)
    draw.text((70,90),'Базы данных — среда, 15:00, кабинет 402',font=font,fill='black')
    draw.text((70,180),'Database class — room 402',font=font,fill='black')
    png=tmp_path/'schedule.png';image.save(png)
    pdf=tmp_path/'scan.pdf';image.save(pdf,'PDF',resolution=150)
    jpg=tmp_path/'schedule.jpg';image.save(jpg)
    webp=tmp_path/'schedule.webp';image.save(webp)
    for path in [png,jpg,webp,pdf]:
        result=loader.load_document_details(path,emit_warnings=False)
        assert '402' in result.text and 'Базы данных' in result.text
        assert result.pages[0].number==1
