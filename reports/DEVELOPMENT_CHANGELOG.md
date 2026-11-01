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
Commit: будет указан фактическим hash в следующей записи журнала.
