# Отчёт о реализации улучшений RAG

Дата: 1 ноября 2026, Asia/Almaty. Начальный commit `f69c57fa056947849ce0a032902aab592a635dc3`. Ветка `fix/rag-quality-improvements`; все коммиты локальные, push не выполнялся.

## Результат и границы готовности

Production-код изменён, протестирован и интегрирован. Полный suite: **128 passed, 0 failed**, 23.89 s, существующий `.venv`, offline. Дополнительный `pip check`: No broken requirements found. Система лучше сохраняет и покрывает источник, но качество ответов зависит от корпуса: на старом eval улучшение, на новом наборе документации регрессия. Нельзя объявить её готовой к автономному предоставлению достоверных ответов.

Первый повтор baseline выполнен **до изменения production** в отдельный `reports/local_results/baseline/`, не перезаписывая диагностический каталог. После реализации baseline дополнительно повторён через `quality.evaluate --revision f69c57f` на тех же весах. EM/F1 совпали с аудитом. Исходные untracked `evaluation/` и `experiments/` сохранены, не включены в новые коммиты. Полные результаты local runs исключены из Git; компактные snapshots и необходимые небольшие QA fixtures включены.

## Что исправлено

| Файлы | Исправление | Что не обещает |
|---|---|---|
| `rag/qa_model.py` | original offsets вместо decode; Unicode word/identifier boundaries; overlapping windows; контекстная маска; проверка budget вопроса; уникальные кандидаты | правильный смысловой ответ и semantic no-answer |
| `rag/chunker.py` | token-aware chunks с точными координатами; specials входят в лимит; sentence preference; overlap без потери non-whitespace | универсально лучшие semantic neighbors |
| `rag/embeddings.py` | offline local-only default; явная ошибка при oversize docs/queries | восстановление информации, которой нет в PDF |
| `rag/pipeline.py` | token-aware default; source/page metadata; `ask_detailed`, structural refusals; сохранён `ask()` | калиброванную уверенность |
| `rag/answer_selection.py` | inspectable retrieval/QA/RRF ranking; проверка source spans; dedup по координатам | доказанного победителя на любых документах |
| `rag/document_loader.py` | консервативная normalization, raw page provenance, ExtractionWarning для пустых/повреждённых страниц | OCR, угадывание склеенных слов, восстановление формул |
| `training/preprocessing.py`, `training/prepare.py` | проверка gold annotations, overflow labels, stable ids/offset provenance, отказ от перезаписи output | установление фактических исторических labels и автоматическое обучение |
| `quality/*` | общий scoring, воспроизводимый Git baseline, fixed fixtures, split/corpus/language metrics, no-answer probes | статистическую значимость на маленьком наборе |

`rag/vector_store.py`, `rag/retriever.py`, старые word/semantic chunker APIs не переписаны. Исправлять проверенную cosine normalization не требовалось. Legacy word chunking доступен явно, однако oversize chunks теперь вызывают исключение, а не silent truncation.

### QA scores и выбор spans

Внутри окна выбирается допустимый span без length bonus. Для сравнения окон используется:

`log_score = log softmax(start_logits[context])[start] + log softmax(end_logits[context])[end]`.

Это концентрация двух отдельных token distributions в данном input, **не вероятность истинности**, не joint posterior по допустимым spans и не валидированная confidence между документами. Window size влияет на нормировку. `score` — исходная сумма logits, `null_margin` — разность этой суммы и CLS суммы, только для диагностики. Ни один из них не применяется как произвольный rejection threshold. По умолчанию между evidence chunks сохраняется retrieval selection; `qa` и `rrf` доступны opt-in. RRF складывает reciprocal retrieval и QA ranks без длинового бонуса.

Token windows обрабатываются моделью по одному, а per-window/global top-N хранят ограниченное число candidates. Общая tokenization по-прежнему линейна по input length. Оптимизация heaps не изменила ни одно из 56 audit predictions; latency улучшением без отдельного benchmark не объявляется.

## Как выбиралась конфигурация

Dev: 6 answerable + 1 unanswerable вопрос. Eval аудита был просмотрен ранее; подбор параметров в текущем этапе выполнялся на dev. Budgets включают special tokens; overlap target 24.

| Budget | Dev pipeline F1 | Recall@1 | Recall@3 | Overlimit chunks |
|---|---:|---:|---:|---:|
| 96 | 69.26% | 83.33% | 100% | 0 |
| 112 | 54.97% | 66.67% | 83.33% | 0 |
| 128 | 47.35% | 66.67% | 83.33% | 0 |

Поэтому default = 96 tokens, а не 48 слов. Малый dev не устанавливает универсальный optimum. На fixed 96 chunks retrieval/QA/RRF дали одинаковые dev F1 69.26%/EM 0%; оснований переключать default на QA score нет. Консервативная PDF normalization не изменила dev predictions. На audit corpora стало 35 synthetic/41 PDF chunks, все не длиннее 96 tokens; baseline имел 18/18 chunks за лимитом модели 128. Одно слово, не помещающееся целиком, вызывает понятную ошибку; оно не отрезается молча.

### Изолированная абляция границ

На ранее просмотренном eval n=20 offsets+windows без whole-word constraints дают oracle F1 63.71%, EM 5%; с constraints — F1 73.54%, EM 15%. Это измеримый вклад на данном наборе; сохранение punctuation и удаление искусственных `##` само по себе не исправляет выученный лишний префикс. Hard cap 30 оставлен настраиваемым: в предыдущем эксперименте увеличение до 90 не помогло. Длинный gold q15 (33 wordpieces) целиком недоступен при default cap, это сохранённое явное ограничение.

## No-answer: реализовано только обоснованное

Пустая retrieval выдача, отсутствие admissible spans и fabricated ответы custom QA отвергаются. Подробный API возвращает `status=no_evidence/no_admissible_span/candidate`, всегда `no_answer_supported=False` для этого checkpoint. `candidate` не выдаётся за подтверждённое знание. Legacy `ask` сохраняет прежнюю форму; для нового UI рекомендуется detailed API.

На 6 answerable и 7 unanswerable dev probes диапазоны log_score пересеклись: answerable [-5.824, -0.006], unanswerable [-2.330, -0.214]. CLS margin разделил только этот маленький dev sample, чего недостаточно для порога с обещанной надёжностью. Threshold не внедрён. На обоих исходных no-answer вопросах и всех 4 новых no-answer вопросах pipeline всё ещё возвращает ложный непустой span. Semantic no-answer **не решён**.

## Тесты и воспроизведение

Сохранены все 69 исходных тестов; добавлено 59. Старые mocked ответы в одном selection test приведены к содержащим их source strings, чтобы проверить прежний ranking контракт при новом обязательном условии extractive source validity; assertions о лучшем score и обработке всех hits сохранены.

Покрыты offsets/EN/RU/KK, special tokens, whole-word и Unicode combining marks, hyphens/identifiers, окна/long answer config, empty inputs, token budgets/overlap, embeddings guard, правильность FAISS, candidate selection/dedup, pipeline source coordinates, extraction warnings, training labels, EM/F1 и отдельные false answers/refusals. Добавлены 5 тестов на настоящих локальных весах/embeddings. Они явно skip при отсутствии fine-tuned weights; в этом запуске **skips не было**.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m pip check
```

Команды общего QA/retrieval/end-to-end сравнения — в README. `quality/evaluate.py` пишет oracle и полный настоящий pipeline, все contexts/hits и scores, split/corpus/language summaries. `quality/summarize.py` делает компактный versioned snapshot; `reports/local_results/` содержит подробности текущего локального запуска. Evaluation scripts не меняют production, notebook или веса.

## Сохранность и Git

Контрольными суммами относительно начала аудита проверены 8 файлов: 4 файла final_qa_model, notebook, PDF, 2 старых evaluation files. Изменений нет. Новые commits не содержат weights, env, caches или больших raw experiments. Исходный диагностический отчёт и его результаты не перезаписаны. Обновлены README, direct requirements и полный environment lock; ничего не устанавливалось.

Фактические hashes и результаты шагов — в `DEVELOPMENT_CHANGELOG.md`. Последний documentation commit можно однозначно получить через `git log -1 -- README.md reports/RAG_IMPLEMENTATION_REPORT.md`; собственный hash не вставляется в его содержимое рекурсивно.

## Проверка acceptance criteria

AC1–AC5, AC8–AC13, AC15 выполнены технически и проверены. AC6: варианты selection оценены, интегрированы traceability/validation/dedup; автоматический переход на QA score не обоснован. AC7 выполнен в разрешённой ограниченной форме: no-answer протестирован, структурные отказы работают, надёжный semantic detector не достигнут. AC14: ограничения checkpoint и новые регрессии явно отмечены. Цель надёжно не выдавать ответ при любом отсутствии evidence в смысле вопроса **не достигнута**. Результат готов к дальнейшей разработке и расширенной проверке, не к обещанию production-grade QA accuracy.
