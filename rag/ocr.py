"""Bounded local OCR. No document is sent to a provider."""
from rag.timetable import DAYS, read_timetable
from time import monotonic

MAX_PAGES = 100
MAX_PIXELS = 20_000_000


class OCRUnavailable(RuntimeError):
    pass


def recognize_image(image):
    import pytesseract
    from PIL import ImageOps
    if image.width * image.height > MAX_PIXELS:
        raise ValueError('Изображение превышает лимит 20 миллионов пикселей.')
    try:
        languages = set(pytesseract.get_languages(config=''))
        if not {'rus', 'eng'} <= languages:
            raise OCRUnavailable('OCR: установите языки Tesseract rus и eng (brew install tesseract-lang).')
        image = ImageOps.exif_transpose(image).convert('RGB')
        deadline = monotonic() + 45
        def read(crop, psm, lang='rus+eng'):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError('OCR: превышено время распознавания страницы (45 секунд).')
            return pytesseract.image_to_string(crop, lang=lang, config=f'--psm {psm}', timeout=remaining).strip()
        ordinary = read(image, 3)
        def read_grid(crop, psm):
            text = read(crop, psm)
            return text if psm != 7 or text.lower() in DAYS else read(crop, psm, 'eng')
        table = read_timetable(image, read_grid)
        return ordinary + '\n\n' + table if table else ordinary
    except pytesseract.TesseractNotFoundError as error:
        raise OCRUnavailable('OCR: установите Tesseract (brew install tesseract tesseract-lang).') from error


def image_text(path):
    try:
        from PIL import Image
        with Image.open(path) as image:
            return recognize_image(image)
    except ImportError as error:
        raise OCRUnavailable('OCR: выполните pip install -r requirements.txt.') from error


def pdf_page_text(path, index):
    try:
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(str(path)) as pdf:
            page = pdf[index]
            try:
                if page.get_width() * page.get_height() * 9 > MAX_PIXELS:
                    raise ValueError('Страница PDF превышает лимит OCR 20 миллионов пикселей.')
                bitmap = page.render(scale=3)
                try:
                    return recognize_image(bitmap.to_pil())
                finally:
                    bitmap.close()
            finally:
                page.close()
    except ImportError as error:
        raise OCRUnavailable('OCR: выполните pip install -r requirements.txt.') from error
