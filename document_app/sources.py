"""Page-level source inspection, local rendering and evidence routing."""
from dataclasses import dataclass
from pathlib import Path
import math
import re
from PIL import Image, ImageOps
from pypdf import PdfReader
from rag.ocr import MAX_PIXELS, MAX_PAGES

IMAGES={'.png','.jpg','.jpeg','.webp'}


@dataclass(frozen=True)
class PageRoute:
    number: int
    visual: bool
    reason: str


def inspect_source(path):
    path=Path(path)
    if path.suffix.lower() in IMAGES:
        with Image.open(path) as image:
            if image.width*image.height>MAX_PIXELS:raise ValueError('Изображение превышает лимит 20 миллионов пикселей.')
            image.verify()
        return [PageRoute(1,True,'Загружено исходное изображение')]
    if path.suffix.lower()!='.pdf':return []
    reader=PdfReader(path)
    if reader.is_encrypted:raise ValueError('PDF защищён паролем. Загрузите незашифрованную копию.')
    if len(reader.pages)>MAX_PAGES:raise ValueError('Максимальный объём PDF — 100 страниц.')
    import pypdfium2 as pdfium
    result=[]
    with pdfium.PdfDocument(str(path)) as pdf:
        for index,page in enumerate(reader.pages):
            text=page.extract_text() or ''
            letters=sum(c.isalnum() for c in text)
            rendered=pdf[index]
            try:
                area=rendered.get_width()*rendered.get_height()
                image_area=0
                for obj in rendered.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
                    x0,y0,x1,y1=obj.get_bounds()
                    image_area+=max(0,x1-x0)*max(0,y1-y0)
                coverage=min(1,image_area/max(1,area))
            finally:rendered.close()
            visual=letters<40 or coverage>=.35
            reason='Скан / недостаточный текстовый слой' if letters<40 else 'Крупное изображение, текстовый слой может быть неполным' if visual else 'Пригодный текстовый слой PDF'
            result.append(PageRoute(index+1,visual,reason))
    return result


def render_source(path,page=1):
    path=Path(path)
    if path.suffix.lower() in IMAGES:
        if page!=1:raise ValueError('У изображения только одна страница.')
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert('RGB')
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(str(path)) as pdf:
        if not 1<=page<=len(pdf):raise ValueError('Номер страницы вне документа.')
        source=pdf[page-1]
        try:
            area=source.get_width()*source.get_height()
            scale=min(3,math.sqrt(MAX_PIXELS/max(1,area)))
            bitmap=source.render(scale=scale)
            try:return bitmap.to_pil().convert('RGB').copy()
            finally:bitmap.close()
        finally:source.close()


def choose_pages(pipeline,routes,question,explicit=None):
    numbers={p.number for p in routes}
    if explicit is not None:
        if explicit not in numbers:raise ValueError('Номер страницы вне документа.')
        return [explicit]
    if len(routes)<=2:return sorted(numbers)
    if pipeline.retriever is None:
        raise ValueError('Для выбора страниц нет поискового текста. Укажите страницу PDF вручную.')
    ranked=[]
    for hit in pipeline.retriever.retrieve(question,top_k=5):
        index=hit['index']
        metadata=pipeline.chunk_metadata[index]
        overlaps=[p for p in pipeline.document_pages if p.number in numbers and p.start<metadata.end and p.end>metadata.start]
        overlaps.sort(key=lambda p:-max(0,min(p.end,metadata.end)-max(p.start,metadata.start)))
        for page in overlaps:
            if page.number not in ranked:ranked.append(page.number)
    if not ranked:raise ValueError('Не удалось выбрать страницу. Укажите её вручную.')
    return ranked[:2]


def needs_visual(question):
    return bool(re.search(r'таблиц|формул|схем|рисунк|картин|diagram|formula|table|visually|picture',question,re.I))
