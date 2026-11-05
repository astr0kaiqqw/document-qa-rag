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
Commit: `0a3b60c` (проверено `git log`).

## Шаг 3 — Проверяемый выбор evidence и ограничения no-answer

Проблема: нельзя было проследить кандидаты и координаты; сторонний QA backend мог вернуть строку вне источника.
Файлы: `rag/answer_selection.py`, `rag/pipeline.py`, `quality/check_no_answer.py`, тесты selection/pipeline.
Исправление: `ask_detailed()` раскрывает кандидаты, координаты документа, источник, retrieval/QA ranks, диагностические scores; дубликаты одного source span исключаются, fabricated spans отвергаются. Старый `ask()` сохраняет четыре поля. Статус `candidate` не означает доказанную правильность; пустой evidence/невалидный span дают прозрачный отказ.
Dev сравнение retrieval/conditional QA log-softmax/RRF: все F1 69.26%, EM 0%, Recall@3 100% (n=6). Поэтому retrieval остаётся default; альтернативы явно opt-in, raw logits как вероятности не используются. Бонус длины не включён.
No-answer: 6 answerable и 7 unanswerable dev probes. log_score диапазоны пересекаются. CLS margin разделяет именно этот маленький набор, но этого недостаточно для надёжного порога; без слепой калибровки threshold не включается. `no_answer_supported=False`.
Проверки: 29 affected tests passed, затем полный suite 107 passed.
Commit: `e6e8ab9` (проверено `git log`).

## Шаг 4 — PDF normalization и page provenance

Проблема: отсутствовали diagnostics о пустых/неотображаемых glyphs и координаты страниц.
Файлы: `rag/document_loader.py`, `rag/pipeline.py`, `tests/test_document_loader.py`.
Исправление: удаляются только NUL/soft hyphen, нормализуются CR/LF. Сохраняются абзацы, формулы, исходные raw page strings и координаты normalized document. Пустые страницы и private-use/replacement glyphs дают ExtractionWarning и diagnostics; ask_detailed возвращает номера страниц. Неоднозначные слитные слова не изменяются.
Проверки: 37 loader/pipeline tests passed; dev end-to-end повторён, качество сравнено с token-aware dev baseline (без заявления semantic improvement от normalization).
Ограничение: raw text — результат pypdf, а не OCR и не исходные glyph coordinates. Нельзя автоматически восстановить испорченные формулы.
Commit: `1808731` (проверено `git log`).

## Шаг 5 — Проверенная подготовка данных для возможного обучения

Проблема: исторические labels не сохранены, notebook source и output расходятся; truncation терял answer supervision.
Файлы: `training/preprocessing.py`, `training/prepare.py`, `tests/test_training_preprocessing.py`.
Исправление: проверить каждую аннотацию как точный source span до tokenization; sliding windows, несколько gold variants, CLS только для окна без полного ответа, source offsets и stable example ids отдельно от model inputs. CLI создаёт новый output с hashes и отказывается перезаписывать существующий. Обучение не запускается.
Проверки: 7 preprocessing regression tests passed; CLI обработал все 28 audit examples с long-context overflow.
Ограничение: это исправленная подготовка features, не доказательство, какие labels фактически использовались в Colab. Training runner/GPU run и fine-tuning потребуют отдельного согласования и зависимостей.
Commit: `5315874` (проверено `git log`).

## Шаг 6 — Ограничение памяти decoder и настоящие offline regression tests

Проблема: на длинном context декодер накапливал все admissible spans для всех окон.
Файлы: `rag/qa_model.py`, `tests/test_qa_model.py`, `tests/test_local_quality.py`.
Исправление: bounded per-window top-N heap и глобальные уникальные top-N; CPU tensors преобразуются в списки перед поиском spans. Число сохраняемых кандидатов не растёт пропорционально числу окон. Tokenization всего context по-прежнему требует линейной памяти.
Проверки: 20 decoder tests passed; 25 decoder + real local-weight tests passed. Все 56 audit oracle/pipeline predictions совпали до/после bounded decoder; полного speed benchmark не проводилось. Настоящие tests покрывают EN/RU/KK spans, checkpoint/tokenizer и offline pipeline.
Commit: `60485eb` (проверено `git log`).

## Шаг 7 — Независимая оценка и честная фиксация регрессий

Файлы: `quality/evaluate.py`, `quality/summarize.py`, `quality/data/independent_questions.json`, `tests/test_quality_metrics.py`, `reports/quality_metrics.json`.
Baseline повторён из кода Git `f69c57f` на тех же локальных весах; предыдущий аудит не перезаписан. Дополнительно 8 answerable + 4 unanswerable вопроса на выдержках docstrings Python/NumPy, с дословными gold и provenance, зафиксированными до inference. Этот набор не использовался для подбора; после оценки он считается просмотренным. Raw predictions/hits сохранены локально и исключены из Git; компактный snapshot и fixtures включены.
Проверки: 4 metric/fixture tests passed; полный suite **127 passed, 0 failed**. Пакетные зависимости: `pip check` — No broken requirements found.
Результат старого eval n=20: pipeline EM 10→15%, F1 41.02→48.59%, Recall@3 70→90%, invalid 15→0%. Oracle F1 54.71→73.54%. No-boundaries ablation oracle F1 63.71%, то есть whole-word ограничения дают измеримый вклад.
Регрессия нового n=8: pipeline F1 72.80→56.55%, EM 37.5→25%; Recall@1 100→75%, Recall@3 100→100%. Параметры после этого не подбирались. RU/KK в общем audit corpus тоже регрессируют (по одному вопросу). Надёжный no-answer не достигнут.
Оригинальные модель/tokenizer/notebook/PDF/evaluation: 8 файлов проверены hashes, изменений нет. Только ожидаемые исходные untracked evaluation/experiments остаются вне новых коммитов.
Commit: `e02c24c` (проверено `git log`).

## Шаг 8 — Сохранение скрытых символов на границах чанков

Проблема обнаружена дополнительным regression test: при overlap=0 символ U+200B между token offsets мог выпадать из source chunks.
Файлы: `rag/chunker.py`, `tests/test_token_chunker.py`.
Исправление: включать в предыдущее source slice пропущенные tokenizer format characters между tokens; whitespace можно оставить вне границы, содержательные/форматирующие non-whitespace сохраняются. Проверяется реальный encoded budget полученного slice.
Проверки: сначала воспроизведено 1 failed/11 passed, после исправления 12 passed. Все 56 audit oracle/pipeline predictions совпали с предыдущим final run; качество не объявляется выросшим из-за этого edge-case fix.
Commit: `689a54f` (проверено `git log`).

## Сводка истории реализации

| Шаг | Исправление | Основные проверки | Commit |
|---|---|---|---|
| 1 | QA offsets, boundaries, windows | 19 QA tests; full 88; oracle evaluation | b9b617f |
| 2 | Token budgets/coverage | 38 affected; full 99; 23 pipeline; 3 dev budgets | 0a3b60c |
| 3 | Candidate traceability/no-answer limits | 29 affected; full 107; 3 selection strategies | e6e8ab9 |
| 4 | PDF provenance/diagnostics | 37 affected; dev E2E unchanged | 1808731 |
| 5 | Training features, без обучения | 7 tests; CLI 28 examples/32 windows | 5315874 |
| 6 | Bounded QA memory/local-model regressions | 25 tests; all 56 predictions unchanged | 60485eb |
| 7 | Frozen baseline/new document evaluation | full 127; metric checks; regressions recorded | e02c24c |
| 8 | Source format characters at chunk boundaries | reproduced failure; 12 passed; 56 predictions unchanged | 689a54f |

Документация и зависимости: README/requirements/lock и четыре implementation reports обновлены по реальным результатам, без установки пакетов. Собственный hash documentation commit доступен через `git log -1 -- README.md reports/RAG_IMPLEMENTATION_REPORT.md` и приведён в финальном ответе; он не вписывается в собственное содержимое рекурсивно.

Полный финальный suite после шага 8: **128 passed, 0 failed**, offline. Финальные audit и independent document evaluations повторены на коде после последнего production commit; метрики сохранены в `quality_metrics.json`. Семантические ограничения и регрессии не объявляются исправленными.

## Доставка приложения: `feat/final-document-qa`

Пользователь сменил приоритет на готовый продукт без повторного исследования. Выбрана одна multilingual extractive QA модель `deepset/xlm-roberta-base-squad2`, добавлены native CLS/null, Streamlit UI, изолированные upload sessions, original-page evidence и launcher. Старый BERT/notebook не перезаписывались. Первоначальная загрузка модели и установка Streamlit/sentencepiece были явно разрешены; runtime остаётся offline.

| Этап | Проверки | Commit |
|---|---|---|
| Один локальный XLM-R QA checkpoint + null | 52 профильных tests, real answer/null smoke | b9b2012 |
| Backend sessions + исходные страницы | 31 tests; PDF/TXT/DOCX acceptance | 8410889 |
| Готовый Streamlit UI и запуск | 32 UI/backend tests; полный suite 139 passed | 58023bc |

Реальная приёмка: PDF → `logistic regression`, `embedding.pdf`, стр. 7; absent answer → null refusal; TXT → `Aurora`; DOCX → `Meridian`; selection/history/reset прошли; AppTest exceptions 0. Живой Streamlit server health → `ok`. Визуальный browser host недоступен, visual review не заявляется. Новый benchmark или сравнение нескольких моделей не проводились. Подробности продукта — `APPLICATION_DELIVERY.md`, запуск — README. Final documentation commit доступен в истории файлов и финальном ответе.
