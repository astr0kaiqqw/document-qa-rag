# Локальная Document QA / RAG

Система извлекает ответы из PDF, DOCX и TXT без внешних LLM API. Использует локальный fine-tuned mBERT, multilingual MiniLM и FAISS. Это extractive QA: ответ является подстрокой источника, а не сгенерированным объяснением.

## Состояние

Исправлены восстановление текста ответа, границы слов, длинные QA inputs и silent truncation embeddings. 128 тестов проходят offline. На ранее просмотренном audit eval F1 полного pipeline вырос с 41.02% до 48.59%; **на дополнительной документации Python/NumPy упал с 72.80% до 56.55%**. Система готова к дальнейшей проверке, но не к использованию как источник гарантированно правильных ответов. SQuAD-v1 checkpoint не умеет надёжно отказываться при отсутствии ответа.

Подробности: [реализация](reports/RAG_IMPLEMENTATION_REPORT.md), [сравнение](reports/RAG_BEFORE_AFTER_COMPARISON.md), [ограничения](reports/REMAINING_LIMITATIONS.md), [обучение](reports/RETRAINING_RECOMMENDATION.md), [Git-журнал](reports/DEVELOPMENT_CHANGELOG.md).

## Архитектура

`load_document_details` → точный извлечённый текст и страницы → tokenizer-aware chunks → normalized embeddings → FAISS `IndexFlatIP` → top-3 evidence → QA offsets/whole-word spans → проверка и выбор кандидата.

- QA: `models/final_qa_model`, multilingual BERT, 512 positions, vocab 119547. Original weights/tokenizer и `models/notebooks/rag_qa_training.ipynb` сохранены.
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, dimension 384, effective limit 128 tokens.
- Default indexing: максимум **96 токенов вместе со special tokens**, overlap target 24; sentence boundaries предпочитаются. Значение выбрано по 6 dev вопросам, универсальным optimum не является.
- QA: окна до 512 токенов вместе с вопросом, overlap до 128; max answer 30 wordpieces, настраивается в `QAModel`. Никакого length bonus.
- FAISS normalization не менялась. Candidate selection default — retrieval similarity; opt-in `qa`/`rrf` экспериментальны и не дают калиброванной уверенности.

## Окружение и модели

Проверено в существующем Python 3.14.7/macOS ARM64: torch 2.14.1, transformers 5.20.1, sentence-transformers 6.1.0, faiss-cpu 1.15.1. Прямые зависимости закреплены в `requirements.txt`; полный снимок установленного окружения — `requirements-lock.txt`. Другие платформы/версии не проверялись. В ходе реализации ничего не устанавливалось и не обновлялось.

В новом окружении зависимости устанавливаются пользователем:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Для полного повторения проверенного окружения можно использовать `requirements-lock.txt` на совместимой платформе. Файлы обученной модели не входят в Git: нужны `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`. Embedding model должна уже находиться в локальном Hugging Face cache. `EmbeddingModel` по умолчанию использует `local_files_only=True` и не скачивает отсутствующие weights. Получение модели для нового компьютера — отдельное действие пользователя, не часть inference.

## Использование

Из корня репозитория:

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load_document("uploads/embedding.pdf")
result = pipeline.ask_detailed("What is Word2vec?")
print(result["answer"])
print(result["status"], result["pages"] if "pages" in result else [])
print(result["candidates"])
```

Для строго offline запуска:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python -m pytest -q -p no:cacheprovider
```

`ask()` сохраняет прежние четыре поля: question/answer/source/retrieval_score. Для новых интерфейсов используйте `ask_detailed()`: он возвращает source coordinates в извлечённом документе, страницы, diagnostics и кандидатов. `status="candidate"` означает непроверенный QA-span. `no_answer_supported=False` сообщает об отсутствии надёжного semantic no-answer. Retrieval similarity, log_score и null_margin не являются вероятностями правильности.

`embedding_token_limit`, `token_overlap`, `top_k` и `answer_selection` настраиваются в RAGPipeline. `chunk_size`/`overlap` в словах применяются только при `token_aware_chunking=False`; даже тогда embedding guard не позволяет молча обрезать chunk. `use_semantic_chunking=False` в token-aware режиме отключает предпочтение sentence boundaries. Запрос длиннее embedding budget явно отклоняется — его нужно сократить; suffix запроса не теряется молча.

PDF warnings предупреждают об отсутствующем тексте/неотображаемых glyphs. OCR не включён. Source coordinates относятся к нормализованному извлечённому тексту, не к PDF bounding boxes; raw page strings сохранены.

## Воспроизведение качества

```bash
.venv/bin/python -m quality.evaluate --revision f69c57f --output reports/local_results/baseline_comparable.json
.venv/bin/python -m quality.evaluate --output reports/local_results/final_verified.json
.venv/bin/python -m quality.evaluate --revision f69c57f --dataset quality/data/independent_questions.json --independent --output reports/local_results/independent_baseline.json
.venv/bin/python -m quality.evaluate --dataset quality/data/independent_questions.json --independent --output reports/local_results/independent_final_bounded.json
.venv/bin/python -m quality.check_no_answer --output reports/local_results/no_answer.json
.venv/bin/python -m quality.summarize
```

Эти модули сами включают offline flags. Baseline загружает исходный Python-код из Git, использует тот же неизменённый checkpoint и старую chunking/selection логику. Векторный store не менялся. `--revision` исполняет доверенный код из указанного локального revision.

Audit dataset уже просмотрен ранее; он не является новым blind test. Дополнительный набор был зафиксирован до первого inference и не участвовал в выборе параметров; после оценки он тоже просмотрен. Фиксированные gold заданы вручную, не моделью. Raw runs хранятся в ignored `reports/local_results/`, компактные реальные metrics — `reports/quality_metrics.json`. Исходный диагностический каталог сохранён локально без перезаписи; он описывает состояние до реализации.

Возможное обучение не запускается автоматически. `training.prepare` только проверяет annotations и создаёт новые features; порядок действий и необходимые согласования описаны в retraining report.
