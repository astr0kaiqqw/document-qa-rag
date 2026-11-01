# Готовое приложение Document QA

1 ноября 2026, Asia/Almaty. Это протокол доставки продукта и короткой приёмки, не новый диагностический аудит или benchmark.

## Реализованный пользовательский сценарий

Streamlit UI: upload PDF/DOCX/TXT → обработка → выбор файла → вопрос → извлечённый кандидат или прозрачный отказ → filename/page/supporting paragraph → повторные вопросы и переключение файлов → reset.

Сессии изолированы; до 8 файлов по 25 МБ. Индексы и история принадлежат документу. Имена файлов очищаются от path traversal; временные пути не выдаются за source filenames. Reset удаляет только собственные временные файлы сессии. Данные источника выводятся текстом, не исполняются как HTML.

## Выбранная модель и offline setup

`deepset/xlm-roberta-base-squad2`, revision `a5fab9908c8d856e8c583fd41ba6d92444e46477`, локальный safetensors checkpoint, multilingual XLM-R fine-tuned на SQuAD 2.0, CC BY 4.0, attribution deepset. Model card: https://huggingface.co/deepset/xlm-roberta-base-squad2 . Это единственный новый выбранный QA checkpoint; сравнения нескольких моделей и fine-tuning не проводились.

Старая `models/final_qa_model` не изменена. Новый QA каталог — `models/extractive_qa`, модели/manifest не коммитятся. Установлены Streamlit 1.66.1 и sentencepiece 0.2.2 с обычными зависимостями в существующий `.venv`. Model downloader вызывается только явно при первоначальной подготовке. Runtime использует local-only weights и offline flags.

CLS/null сравнивается с non-null span внутри каждого окна, threshold=0 означает прямое соревнование logits, а не подогнанную probability confidence. Если все evidence windows отказываются, возвращается `no_answer`. При непустом ответе UI всегда сообщает «извлечённый кандидат», без обещания достоверности.

## Интеграционная доработка

Поиск сохраняет token-aware embedding budget. Для QA `context_expansion=True` читает исходные страницы, перекрываемые найденным chunk, **отдельно**, либо окружающий TXT/DOCX paragraph. Это устранило ошибку приёмки, когда короткий chunk/смесь соседних слайдов вызывали отказ на answerable вопросе. Doc offsets и page mapping сохраняются. Изменение покрыто отдельным regression test. Старый pipeline API и default context_expansion=False сохранены для существующих пользователей scripts.

## Фактическая приёмка

`./run_app.sh --server.port 8501` успешно запустил живой сервер; HTTP `http://127.0.0.1:8501/_stcore/health` → `ok`.

`scripts/smoke_app.py` использовал Streamlit AppTest с настоящими file_uploader/form/selectbox widgets и реальными моделями, без mocks:

| Проверка | Фактический результат |
|---|---|
| Реальный PDF upload | `uploads/embedding.pdf` проиндексирован |
| Answerable question | “Which classifier is used to distinguish positive and negative examples?” → `logistic regression` |
| Source | `embedding.pdf`, страница 7; exact document slice и supporting paragraph содержат answer |
| Unanswerable question | Вопрос о пароле почты → empty answer, `status=no_answer`, UI `no_reliable_answer` |
| Другой TXT | `station.txt` → `Aurora`, правильный источник |
| DOCX | `laboratory.docx` → `Meridian`, правильный источник |
| Переключение | Возврат к PDF сохранил его собственную историю |
| Повторные вопросы | История не смешивается между документами |
| Reset | Документы/индексы/история очищены |
| Streamlit exceptions | 0 |

Визуальная browser automation была недоступна (browser access host unavailable); полноценный визуальный browser review не заявляется. Процесс Streamlit, health endpoint и реальные widget interactions проверены описанными способами.

Полный существующий и дополненный pytest suite: **139 passed, 0 failed**, 24.95 s, offline. `pip check`: No broken requirements found. Проверки включают native null, local XLM-R loading, session isolation, PDF source expansion, upload errors, reset и прежние 128 regressions. QA unit tests с контролируемыми logits дополняют, а не заменяют реальный acceptance smoke.

## Запуск

На этом компьютере: `./run_app.sh`, затем http://127.0.0.1:8501 . Модель и зависимости уже подготовлены. Для нового компьютера exact commands находятся в README; компоненты вручную соединять не нужно.

## Ограничения

Нет OCR, равной подтверждённой точности EN/RU/KK, абсолютной гарантии no-answer, автоматического разрешения противоречий или permanent document library. Ответы — кандидаты с проверяемым source, не сгенерированные объяснения. Модель не переобучалась, старый checkpoint/notebook не перезаписывался. Новых EM/F1 сравнений или больших evaluation datasets не создавалось.

## Git

Ветка `feat/final-document-qa`, сохранена прежняя история. Функциональные commits:

- `b9b2012` — локальный XLM-R QA, native CLS/null, setup и tests.
- `8410889` — изоляция uploads, история, source/page evidence.
- `58023bc` — Streamlit интерфейс, launcher и acceptance smoke.

Заключительный documentation commit определяется через `git log -1 -- README.md reports/APPLICATION_DELIVERY.md` и указан в финальном ответе. Push не выполнялся; model weights/caches/venv не коммитились. Исходные untracked `evaluation/` и `experiments/` сохранены.
