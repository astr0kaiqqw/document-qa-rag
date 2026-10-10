"""Extract document text without performing chunking or model inference."""

from pathlib import Path

from docx import Document
from pypdf import PdfReader


def _load_txt(file_path: Path) -> str:
    """Read a text document as UTF-8, preserving internal newlines."""
    return file_path.read_text(encoding="utf-8")


def _load_pdf(file_path: Path) -> str:
    """Extract all PDF pages, separating non-empty pages by blank lines."""
    reader = PdfReader(file_path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page)


def _load_docx(file_path: Path) -> str:
    """Extract non-empty body paragraphs, preserving paragraph separation."""
    document = Document(file_path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def load_document(file_path: str) -> str:
    """Load a PDF, UTF-8 TXT, or DOCX file and return trimmed extracted text.

    Extensions are case-insensitive. Raises FileNotFoundError for missing
    files and ValueError for unsupported formats or empty extracted text.
    Decoding, filesystem, and document parsing errors propagate to the caller.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document does not exist: {path}")

    loaders = {".txt": _load_txt, ".pdf": _load_pdf, ".docx": _load_docx}
    extension = path.suffix.lower()
    if extension not in loaders:
        raise ValueError(
            f"Unsupported document extension {extension!r}. "
            "Supported extensions: .pdf, .txt, .docx."
        )

    text = loaders[extension](path).strip()
    if not text:
        raise ValueError(f"Document contains no extractable text: {path}")
    return text
