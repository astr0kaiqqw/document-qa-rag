"""Session-isolated documents; shared local model weights never retain documents."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
import warnings

import torch
from rag.embeddings import EmbeddingModel, MODEL_NAME
from rag.qa_model import ExtractiveQAModel, APPLICATION_MODEL_PATH
from rag.pipeline import RAGPipeline
from rag.document_loader import SourcePage
from rag.ocr import OCRUnavailable
from rag.timetable import PREFIX, format_records, is_timetable_question
from document_app.vision import LocalVision, VisionError
from document_app.sources import inspect_source, render_source, choose_pages, needs_visual

ROOT=Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES=25*1024*1024
MAX_DOCUMENTS=8
PROCESSING_VERSION=4


class LocalModels:
    """Serialize shared inference; all assets are local, no runtime downloading."""
    def __init__(self):
        self.lock=RLock()
        self._embedding=None
        self._qa=None

    def embedding(self):
        with self.lock:
            if self._embedding is None:
                torch.set_num_threads(4)
                folder=ROOT/'models/embedding_model'
                self._embedding=EmbeddingModel(model_path=folder if (folder/'modules.json').exists() else MODEL_NAME)
            return self._embedding

    def vision(self):
        with self.lock:
            configured=LocalVision()
            current=getattr(self, '_vision', None)
            if not isinstance(current, LocalVision) or (current.base_url,current.model,current.timeout)!=(configured.base_url,configured.model,configured.timeout):
                self._vision=configured
            return self._vision

    def qa(self):
        with self.lock:
            if self._qa is None:
                torch.set_num_threads(4)
                self._qa=ExtractiveQAModel()
            return self._qa


def upgrade_cached_models(models):
    """Upgrade pre-vision resource objects on *every cache hit*.

    Preserve loaded weights and locks. Streamlit can return an older resource
    that is different from the one held by the current document session.
    """
    if models.__class__.__name__ == 'LocalModels' and not isinstance(models, LocalModels):
        models.__class__ = LocalModels
    return models


@dataclass
class SessionDocument:
    key: str
    name: str
    pipeline: RAGPipeline
    history: list[dict]=field(default_factory=list)
    source_path: str=""
    page_routes: tuple=()


class DocumentSession:
    """Only files created by this session are deleted on reset/close."""
    def __init__(self, models=None):
        self.models=models or LocalModels()
        self.documents={}
        self._temporary=TemporaryDirectory(prefix='local-document-qa-')

    def add_upload(self, name: str, data: bytes) -> SessionDocument:
        name=Path(name.replace('\\','/')).name.strip()
        extension=Path(name).suffix.lower()
        if extension not in {'.pdf','.docx','.txt','.png','.jpg','.jpeg','.webp'}:
            raise ValueError('Поддерживаются PDF, DOCX, TXT, PNG, JPG и WEBP.')
        if not data:
            raise ValueError('Файл пуст. Выберите документ с текстом.')
        if len(data)>MAX_UPLOAD_BYTES:
            raise ValueError('Максимальный размер файла — 25 МБ.')
        key=hashlib.sha256(name.encode('utf-8')+b'\0'+data).hexdigest()
        if key in self.documents and getattr(self.documents[key].pipeline, 'processing_version', 1) == PROCESSING_VERSION:
            return self.documents[key]
        if key not in self.documents and len(self.documents)>=MAX_DOCUMENTS:
            raise ValueError('В одной сессии доступно до 8 документов. Сбросьте сессию для новых файлов.')
        path=Path(self._temporary.name)/(key+extension)
        source_existed=path.exists()
        path.write_bytes(data)
        try:
            routes=inspect_source(path)
            pipeline=RAGPipeline(top_k=3,context_expansion=True)
            with self.models.lock:
                pipeline.embedding_model=self.models.embedding()
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('always')
                    try:
                        visual_pages={p.number for p in routes if p.visual}
                        if visual_pages:pipeline.load_document(str(path),ocr_pages=visual_pages)
                        else:pipeline.load_document(str(path))
                    except (ValueError, OCRUnavailable, RuntimeError) as error:
                        # An intact image must remain usable when OCR finds no
                        # words (games, formulas, diagrams) or is unavailable.
                        if not routes or not all(p.visual for p in routes) or not (
                            'no extractable text' in str(error) or isinstance(error, (OCRUnavailable, RuntimeError))):
                            raise
                        pipeline.document_pages=tuple(SourcePage(p.number,0,0,'') for p in routes)
                        pipeline.extraction_diagnostics=(f'Поисковый индекс / OCR недоступен: {error}. Источник сохранён для визуальных вопросов.',)

            # Internal temporary names never appear as source citations.
            pipeline.document_path=name
            pipeline.processing_version=PROCESSING_VERSION
            document=SessionDocument(key,name,pipeline,source_path=str(path),page_routes=tuple(routes))
            self.documents[key]=document
            return document
        except Exception:
            if not source_existed:path.unlink(missing_ok=True)
            raise

    def ask(self, key: str, question: str, *, mode='auto', page=None, progress=None) -> dict:
        if key not in self.documents:
            raise ValueError('Сначала выберите загруженный документ.')
        question=question.strip()
        if not question:
            raise ValueError('Введите вопрос.')
        if len(question)>2000:raise ValueError('Максимальный размер вопроса — 2000 символов.')
        if mode not in {'auto','text','visual'}:raise ValueError('Неизвестный режим обработки.')
        document=self.documents[key]
        if getattr(document.pipeline,'processing_version',1)!=PROCESSING_VERSION:
            original=Path(getattr(document,'source_path','') or (Path(self._temporary.name)/(key+Path(document.name).suffix.lower())))
            if not original.is_file():raise ValueError('Сначала загрузите исходный файл повторно для новой обработки.')
            document=self.add_upload(document.name,original.read_bytes())
        routes=getattr(document,'page_routes',())
        with self.models.lock:
            if progress:progress('Выбираем подходящие страницы…')
            selected=choose_pages(document.pipeline,routes,question,page) if routes else []
            visual=bool(routes) and (mode=='visual' or any(p.visual for p in routes if p.number in selected)
                                    or (mode=='auto' and needs_visual(question)))
            if mode=='text' and any(p.visual for p in routes if p.number in selected):
                raise ValueError('У выбранного источника неполный текстовый слой. Используйте визуальный режим.')
            if mode=='text':visual=False
            if mode=='visual' and not routes:raise ValueError('Визуальный режим доступен для PDF и изображений.')
            timetable = mode == 'auto' and visual and is_timetable_question(question) and any(
                p.number in selected and PREFIX in document.pipeline.document_text[p.start:p.end]
                for p in document.pipeline.document_pages
            )
            if timetable:
                if progress:progress('Сверяем строки распознанного расписания…')
                result = document.pipeline.ask_detailed(question, source_pages=selected)
                records = [item['text'] for item in result['evidence_spans']]
                result['answer'] = format_records(records, question)
                result['path_label'] = 'OCR расписания'
                result['preview_pages'] = result.get('pages', selected)
            elif visual:
                answers=[];notes=[]
                for number in selected:
                    if progress:progress(f'Подготавливаем изображение страницы {number}…')
                    image=render_source(document.source_path,number)
                    try:
                        if progress:progress(f'Локальная модель формирует ответ по странице {number}…')
                        reply=self.models.vision().answer(question,image)
                    finally:image.close()
                    notes.append(reply['note'])
                    if reply['answer']:answers.append(f'Страница {number}: '+reply['answer'] if len(selected)>1 else reply['answer'])
                result=dict(question=question,answer='\n\n'.join(answers),source='',pages=selected,
                            retrieval_score=None,status='generated' if answers else 'no_answer',
                            path_label='Локальная модель по изображению',answer_method='vision',
                            model=self.models.vision().model,note='\n\n'.join(notes),
                            route_reason='; '.join(p.reason for p in routes if p.number in selected),
                            preview_pages=selected,extraction_diagnostics=list(document.pipeline.extraction_diagnostics))
            else:
                if progress:progress('Ищем текстовый контекст и извлекаем ответ QA…')
                document.pipeline.qa_model=self.models.qa()
                allowed=[p.number for p in routes if not p.visual] if routes else None
                result=document.pipeline.ask_detailed(question,allowed_pages=([page] if page is not None else allowed),source_pages=([page] if page is not None else None))
                result['path_label']='QA по тексту'
                result['preview_pages']=result.get('pages',[]) if routes else []
        result['filename']=document.name
        result['supporting_paragraph']=supporting_paragraph(document.pipeline,result)
        # Never label an extracted span as a verified fact or a confidence percentage.
        result['display_status']=('generated_unverified' if result.get('answer_method')=='vision' else 'uncertain_candidate') if result['answer'] else 'no_reliable_answer'
        document.history.append(result)
        return result

    def close(self):
        self.documents.clear()
        self._temporary.cleanup()


def supporting_paragraph(pipeline, result, limit=2400):
    if result.get('answer_method') == 'timetable':
        return result.get('source', '')
    start,end=result.get('document_start'),result.get('document_end')
    if start is None or end is None:
        return result.get('source','')
    text=pipeline.document_text
    left=text.rfind('\n\n',0,start)+2
    if left==1:left=0
    right=text.find('\n\n',end)
    if right<0:right=len(text)
    left=max(left,start-limit//2)
    right=min(right,max(end,left+limit))
    return text[left:right]


def friendly_error(error):
    message=str(error)
    if isinstance(error,VisionError):return message
    if 'OCR:' in message or 'лимит' in message or 'Максимальный объём' in message:
        return message
    if isinstance(error,FileNotFoundError) and ('model' in message.lower() or 'QA' in message):
        return 'QA-модель ещё не подготовлена. Выполните: .venv/bin/python scripts/setup_models.py'
    if isinstance(error,UnicodeDecodeError):
        return 'TXT должен быть сохранён в UTF-8. Пересохраните файл и загрузите снова.'
    if 'no extractable text' in message:
        return 'Текст прочитать не удалось. Проверьте качество скана и установку Tesseract с rus/eng. ' + message
    if 'Embedding input exceeds' in message or 'budget' in message:
        return 'Текст или вопрос не помещается в допустимое окно. Сократите вопрос; для необычно длинных слов попробуйте другой документ.'
    if isinstance(error,ValueError) and any(word in message for word in ['Поддерживаются','Введите','Максимальный','Файл пуст','В одной','Сначала','Номер страницы','Для выбора','Не удалось выбрать','У выбранного','Визуальный режим','PDF защищён']):
        return message
    if isinstance(error,OSError) and ('huggingface' in message.lower() or 'cached' in message.lower()):
        return 'Не найдена локальная модель. Один раз выполните: .venv/bin/python scripts/setup_models.py'
    return 'Не удалось обработать источник или вопрос. Проверьте формат, целостность файла и доступность локальных моделей.'
