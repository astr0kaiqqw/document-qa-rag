"""Local speech transcription, explicitly prepared model, bounded WAV input."""
from io import BytesIO
from pathlib import Path
from threading import RLock
import wave

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/whisper-base'
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 60


class SpeechError(ValueError):
    pass


class LocalSpeech:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.lock = RLock()

    def transcribe(self, data, language=None):
        if not data or len(data) > MAX_AUDIO_BYTES:
            raise SpeechError('Пустая или слишком большая запись. Запишите вопрос до 60 секунд.')
        if language not in (None, 'ru', 'en', 'kk'):
            raise SpeechError('Выберите русский, казахский, английский или автоматическое определение языка.')
        try:
            with wave.open(BytesIO(data)) as audio:
                duration = audio.getnframes() / audio.getframerate()
                if duration <= 0 or duration > MAX_AUDIO_SECONDS:
                    raise SpeechError('Запись должна длиться от 0 до 60 секунд.')
        except (wave.Error, EOFError, ZeroDivisionError) as error:
            raise SpeechError('Запись повреждена. Повторите запись микрофоном.') from error
        with self.lock:
            if self.model is None:
                if not (self.model_path / 'model.bin').exists():
                    raise SpeechError('Модель речи не установлена. Выполните: .venv/bin/python scripts/setup_speech.py')
                try:
                    from faster_whisper import WhisperModel
                except ImportError as error:
                    raise SpeechError('Установите зависимости: pip install -r requirements.txt') from error
                self.model = WhisperModel(str(self.model_path), device='cpu', compute_type='int8', cpu_threads=4, local_files_only=True)
            try:
                segments, _ = self.model.transcribe(BytesIO(data), language=language, vad_filter=True, beam_size=3)
                text = ' '.join(segment.text.strip() for segment in segments).strip()
            except Exception as error:
                raise SpeechError('Не удалось распознать запись. Проверьте микрофон и повторите попытку.') from error
        if not text:
            raise SpeechError('Речь не обнаружена. Говорите ближе к микрофону и повторите запись.')
        return text[:2000]
