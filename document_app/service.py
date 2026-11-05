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

ROOT=Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES=25*1024*1024
MAX_DOCUMENTS=8


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

    def qa(self):
        with self.lock:
            if self._qa is None:
                torch.set_num_threads(4)
                self._qa=ExtractiveQAModel()
            return self._qa


@dataclass
class SessionDocument:
    key: str
    name: str
    pipeline: RAGPipeline
    history: list[dict]=field(default_factory=list)


class DocumentSession:
    """Only files created by this session are deleted on reset/close."""
    def __init__(self, models=None):
        self.models=models or LocalModels()
        self.documents={}
        self._temporary=TemporaryDirectory(prefix='local-document-qa-')

    def add_upload(self, name: str, data: bytes) -> SessionDocument:
        name=Path(name.replace('\\','/')).name.strip()
        extension=Path(name).suffix.lower()
        if extension not in {'.pdf','.docx','.txt'}:
            raise ValueError('Поддерживаются только PDF, DOCX и TXT.')
        if not data:
            raise ValueError('Файл пуст. Выберите документ с текстом.')
        if len(data)>MAX_UPLOAD_BYTES:
            raise ValueError('Максимальный размер файла — 25 МБ.')
        key=hashlib.sha256(name.encode('utf-8')+b'\0'+data).hexdigest()
        if key in self.documents:
            return self.documents[key]
        if len(self.documents)>=MAX_DOCUMENTS:
            raise ValueError('В одной сессии доступно до 8 документов. Сбросьте сессию для новых файлов.')
        path=Path(self._temporary.name)/(key+extension)
        path.write_bytes(data)
        try:
            pipeline=RAGPipeline(top_k=3,context_expansion=True)
            with self.models.lock:
                pipeline.embedding_model=self.models.embedding()
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('always')
                    pipeline.load_document(str(path))
            # Internal temporary names never appear as source citations.
            pipeline.document_path=name
            document=SessionDocument(key,name,pipeline)
            self.documents[key]=document
            return document
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def ask(self, key: str, question: str) -> dict:
        if key not in self.documents:
            raise ValueError('Сначала выберите загруженный документ.')
        question=question.strip()
        if not question:
            raise ValueError('Введите вопрос.')
        document=self.documents[key]
        with self.models.lock:
            document.pipeline.qa_model=self.models.qa()
            result=document.pipeline.ask_detailed(question)
        result['filename']=document.name
        result['supporting_paragraph']=supporting_paragraph(document.pipeline,result)
        # Never label an extracted span as a verified fact or a confidence percentage.
        result['display_status']='uncertain_candidate' if result['answer'] else 'no_reliable_answer'
        document.history.append(result)
        return result

    def close(self):
        self.documents.clear()
        self._temporary.cleanup()


def supporting_paragraph(pipeline, result, limit=2400):
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
    if isinstance(error,FileNotFoundError) and ('model' in message.lower() or 'QA' in message):
        return 'QA-модель ещё не подготовлена. Выполните: .venv/bin/python scripts/setup_models.py'
    if isinstance(error,UnicodeDecodeError):
        return 'TXT должен быть сохранён в UTF-8. Пересохраните файл и загрузите снова.'
    if 'no extractable text' in message:
        return 'В документе нет извлекаемого текста. Для сканированного PDF сначала выполните OCR.'
    if 'Embedding input exceeds' in message or 'budget' in message:
        return 'Текст или вопрос не помещается в допустимое окно. Сократите вопрос; для необычно длинных слов попробуйте другой документ.'
    if isinstance(error,ValueError) and any(word in message for word in ['Поддерживаются','Введите','Максимальный','Файл пуст','В одной','Сначала']):
        return message
    if isinstance(error,OSError) and ('huggingface' in message.lower() or 'cached' in message.lower()):
        return 'Не найдена локальная модель. Один раз выполните: .venv/bin/python scripts/setup_models.py'
    return 'Не удалось обработать документ или вопрос. Проверьте, что файл открывается, содержит текст и не защищён паролем.'
