"""Tests using temporary, programmatically generated documents."""

from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from rag.document_loader import load_document


def test_txt_loading(tmp_path: Path) -> None:
    """UTF-8 text and internal paragraph breaks are preserved."""
    path = tmp_path / "example.txt"
    path.write_text("  Hello, әлем!\n\nSecond paragraph.\n ", encoding="utf-8")
    assert load_document(str(path)) == "Hello, әлем!\n\nSecond paragraph."


def test_docx_loading(tmp_path: Path) -> None:
    """Empty paragraphs are skipped and other paragraphs are trimmed."""
    path = tmp_path / "example.docx"
    document = Document()
    for paragraph in ["  First paragraph.  ", "", "   ", "Second paragraph."]:
        document.add_paragraph(paragraph)
    document.save(path)
    assert load_document(str(path)) == "First paragraph.\n\nSecond paragraph."


def test_pdf_loading(tmp_path: Path) -> None:
    """Real PDF text streams on multiple pages are extracted in order."""
    path = tmp_path / "example.pdf"
    writer = PdfWriter()
    for text in ["First page.", "Second page."]:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    writer.write(path)
    assert load_document(str(path)) == "First page.\n\nSecond page."


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_document(str(tmp_path / "missing.txt"))


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "example.csv"
    path.write_text("some text", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported document extension"):
        load_document(str(path))


@pytest.mark.parametrize("content", ["", " \n\t "])
def test_empty_txt(tmp_path: Path, content: str) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="no extractable text"):
        load_document(str(path))


def test_uppercase_extension(tmp_path: Path) -> None:
    path = tmp_path / "example.TXT"
    path.write_text("Text", encoding="utf-8")
    assert load_document(str(path)) == "Text"


def test_invalid_utf8_propagates(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        load_document(str(path))


def test_corrupt_docx_propagates(tmp_path: Path) -> None:
    from zipfile import BadZipFile

    path = tmp_path / "corrupt.docx"
    path.write_bytes(b"Not a DOCX archive")
    with pytest.raises(BadZipFile):
        load_document(str(path))


def test_pdf_normalization_does_not_invent_word_spaces():
    from rag.document_loader import normalize_pdf_text
    text = 'co\u00adoperate\x00\r\n\r\nE=mc²; AB-123/X; asfeaturesin; maygeneralizebetter'
    assert normalize_pdf_text(text) == 'cooperate\n\nE=mc²; AB-123/X; asfeaturesin; maygeneralizebetter'


def test_pdf_page_metadata_and_missing_text_warning(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import rag.document_loader as loader
    path = tmp_path / 'pages.pdf';path.write_bytes(b'fixture')
    raw = ['  First\u00ad page.  ', '', 'Formula \ue000 and text.']
    monkeypatch.setattr(loader,'PdfReader',lambda _:SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda s=s:s) for s in raw]))
    with pytest.warns(loader.ExtractionWarning,match='no extractable text'):
        document=loader.load_document_details(str(path))
    assert document.text == 'First page.\n\nFormula \ue000 and text.'
    assert document.pages[0].raw_text == raw[0]
    assert document.pages[1].start==document.pages[1].end
    third=document.pages[2]
    assert third.number==3
    assert document.text[third.start:third.end]=='Formula \ue000 and text.'
    assert len(document.diagnostics)==3


def test_details_keep_txt_whitespace_and_raw_text(tmp_path):
    from rag.document_loader import load_document_details
    path=tmp_path/'source.txt';path.write_text('  A  word.\n\nБір сөз.  ',encoding='utf-8')
    doc=load_document_details(str(path))
    assert doc.text=='A  word.\n\nБір сөз.'
    assert doc.pages[0].raw_text.startswith('  ')
    assert doc.pages[0].number is None
