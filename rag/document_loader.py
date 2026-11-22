"""Conservative extraction with page provenance and explicit PDF diagnostics."""
from dataclasses import dataclass
from pathlib import Path
import unicodedata
import warnings

from docx import Document
from pypdf import PdfReader
from rag.ocr import image_text, pdf_page_text, MAX_PAGES, OCRUnavailable


class ExtractionWarning(UserWarning):
    """Extraction succeeded but some source content may require manual inspection."""


@dataclass(frozen=True)
class SourcePage:
    number: int | None
    start: int
    end: int
    raw_text: str


@dataclass(frozen=True)
class DocumentText:
    text: str
    path: str
    pages: tuple[SourcePage, ...]
    diagnostics: tuple[str, ...]


def normalize_pdf_text(text: str) -> str:
    """Remove only NUL/discretionary soft hyphens; preserve formulas and layout.

    Never guess spaces in merged words or join ambiguous hard-hyphen line ends.
    Coordinates after normalization refer to DocumentText.text, not PDF glyphs.
    """
    return text.replace('\r\n','\n').replace('\r','\n').replace('\x00','').replace('\u00ad','')


def _load_txt(file_path: Path) -> str:
    return file_path.read_text(encoding='utf-8')


def _load_docx(file_path: Path) -> str:
    document = Document(file_path)
    return '\n\n'.join(p.text.strip() for p in document.paragraphs if p.text.strip())


def load_document_details(file_path: str, *, emit_warnings=True, ocr_pages=None) -> DocumentText:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'Document does not exist: {path}')
    extension = path.suffix.lower()
    if path.stat().st_size > 25 * 1024 * 1024:
        raise ValueError('Максимальный размер файла — 25 МБ.')
    pages=[];diagnostics=[]
    if extension == '.pdf':
        texts=[];cursor=0
        pdf_pages = PdfReader(path).pages
        if len(pdf_pages) > MAX_PAGES:
            raise ValueError('Максимальный объём PDF — 100 страниц.')
        for number,page in enumerate(pdf_pages,1):
            raw=page.extract_text() or ''
            normalized=normalize_pdf_text(raw).strip()
            usable = any(c.isalnum() and unicodedata.category(c) != 'Co' for c in normalized)
            if not usable or number in (ocr_pages or ()):
                if not usable:normalized = ''
                try:
                    recognized = normalize_pdf_text(pdf_page_text(path, number - 1)).strip()
                    if recognized:normalized = normalized + '\n\n' + recognized if normalized else recognized
                    diagnostics.append(f'Page {number}: OCR completed' if recognized else f'Page {number}: OCR could not read text')
                except (OCRUnavailable, RuntimeError) as error:
                    diagnostics.append(f'Page {number}: {error}')
            if not normalized:
                diagnostics.append(f'Page {number}: no extractable text; scanned content may require OCR')
                pages.append(SourcePage(number,cursor,cursor,raw))
                continue
            if any(unicodedata.category(c)=='Co' or c=='\ufffd' for c in normalized):
                diagnostics.append(f'Page {number}: unmapped/private-use glyphs; formulas may be damaged')
            if '\x00' in raw or '\u00ad' in raw:
                diagnostics.append(f'Page {number}: removed NUL or discretionary soft hyphen; raw text retained')
            if texts:
                cursor+=2
            start=cursor;cursor+=len(normalized)
            pages.append(SourcePage(number,start,cursor,raw))
            texts.append(normalized)
        text='\n\n'.join(texts)
    elif extension in {'.png', '.jpg', '.jpeg', '.webp'}:
        text = image_text(path)
        pages = [SourcePage(1, 0, len(text), text)]
        diagnostics.append('Page 1: OCR completed')
    elif extension in {'.txt','.docx'}:
        raw=(_load_txt if extension=='.txt' else _load_docx)(path)
        text=raw.strip();pages=[SourcePage(None,0,len(text),raw)]
    else:
        raise ValueError(f'Unsupported document extension {extension!r}. Supported extensions: .pdf, .txt, .docx, .png, .jpg, .jpeg, .webp.')
    if not text:
        raise ValueError('Document contains no extractable text; OCR could not read text. ' + '; '.join(diagnostics))
    if diagnostics and emit_warnings:
        warnings.warn(f'{path.name}: '+ '; '.join(diagnostics),ExtractionWarning,stacklevel=2)
    return DocumentText(text,str(path),tuple(pages),tuple(diagnostics))


def _load_pdf(file_path: Path) -> str:
    return load_document_details(str(file_path)).text


def load_document(file_path: str) -> str:
    """Compatibility API returning trimmed text; use details for page/raw provenance."""
    return load_document_details(file_path).text
