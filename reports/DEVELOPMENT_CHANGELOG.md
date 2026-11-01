# Журнал реализации

Начало: `f69c57fa056947849ce0a032902aab592a635dc3`, ветка `fix/rag-quality-improvements`.
Исходные untracked `evaluation/` и `experiments/` сохранены; модель и notebook не изменяются.
Коммиты остаются локальными.

## Шаг 1 — QA decoder

Проблема: decode искажал текст, допускались части слов, хвост длинного context обрезался.
Изменены: `rag/qa_model.py`, `tests/test_qa_model.py`; добавлен воспроизводимый `quality/evaluate.py` с фиксированными вопросами аудита.
Исправление: offsets, Unicode boundaries для слов/идентификаторов, overlapping windows, уникальные spans с координатами; `answer()` по-прежнему возвращает строку. Между окнами используется сумма log-softmax по context tokens, это не вероятность правильности ответа. Бонус за длину отсутствует.
Проверки: 19 decoder regression tests passed; полный suite 88 passed. Oracle на ранее просмотренных 20 eval: EM 15%, F1 73.54%, invalid 0%; baseline EM 10%, F1 54.71%.
Ограничение: SQuAD v1 не обеспечивает семантический отказ; правильные символы не гарантируют правильный смысл.
Commit: `b9b617f` (проверено `git log`).

## Шаг 2 — Token-aware indexing

Проблема: 18/18 исходных чанков выходили за 128-token budget embeddings.
Файлы: `rag/chunker.py`, `rag/embeddings.py`, `rag/pipeline.py`, `tests/test_token_chunker.py`, `tests/test_pipeline.py`.
Исправление: точные source substrings с координатами; special tokens входят в лимит; sentence-aware token windows с overlap 24; guard запрещает silent truncation документов и запросов. Embeddings по умолчанию загружаются только локально. Legacy word chunkers доступны через `token_aware_chunking=False`, но guard по-прежнему действует.
Проверены dev budgets 96/112/128 tokens: F1 pipeline 69.26/54.97/47.35%, Recall@3 100/83.33/83.33%; выбран 96, только по dev (6 answerable, малая выборка). На обоих корпусах 0 overlimit chunks при всех трёх budgets. Нет подбора по eval.
Проверки: 38 affected tests passed; полный suite 99 passed, затем ещё 23 pipeline tests passed с новым тестом координат.
Ограничение: token budget гарантирует покрытие, но не правильность semantic retrieval; слишком длинное единичное слово отклоняется явно.
Commit: фактический hash будет внесён после создания в следующем шаге.
