# Сравнение качества до и после

Все значения измерены на тех же локальных model/tokenizer weights. Baseline — код Git `f69c57f`, final — реализованный decoder + token-aware indexing 96/24 + retrieval selection. Веса/данные не подменялись. Полные predictions, hit texts и scores — ignored `reports/local_results`; компактные метрики и индивидуальные ответы — `quality_metrics.json`.

## Методика

Старый audit: 28 вопросов = 26 answerable + 2 unanswerable; 6 answerable/1 unanswerable dev, 20/1 eval. Eval уже был просмотрен до реализации, поэтому не новый blind test. По dev выбраны budgets 96/112/128; retrieval/QA/RRF не дали различий на выбранном dev. Новые 12 вопросов (8 answerable/4 unanswerable) на 4 реальных excerpts локальных Python/NumPy docstrings зафиксированы отдельно до inference и не использованы для выбора параметров. После этих замеров они тоже просмотрены. Ground truth — ручные точные подстроки, не ответы той же модели. Разметка новых excerpts не гарантирует, что эти тексты никогда не встречались в pretraining mBERT.

EM/F1 — SQuAD-style normalization (lower/ASCII punctuation/articles/whitespace), F1 по multiset слов. `Invalid` — непустой ответ вне точного поданного source; `empty` отдельно. Whole-word validity дополнительно проверяется decoder и тестами. Recall@K здесь означает наличие нормализованного gold span хотя бы в одном из K chunks, не полную human relevance annotation. MRR ограничен top-5. Примеры без полного gold chunk имеют recall=0; короткие повторяющиеся spans могут завышать recall.

False answer rate = непустые ответы на unanswerable; false refusal = пустые ответы на answerable. No-answer не смешивается с answerable EM/F1. За время реализации semantic rejection threshold не включался.

## Основной eval аудита (n=20 answerable)

| Метрика | Baseline | Final | Разница |
|---|---:|---:|---:|
| Oracle EM | 10.00% | 15.00% | +5.00 п.п. |
| Oracle Token F1 | 54.71% | 73.54% | +18.83 п.п. |
| Pipeline EM | 10.00% | 15.00% | +5.00 п.п. |
| Pipeline Token F1 | 41.02% | 48.59% | +7.57 п.п. |
| Recall@1 | 55.00% | 65.00% | +10.00 п.п. |
| Recall@3 | 70.00% | 90.00% | +20.00 п.п. |
| Recall@5 | 80.00% | 95.00% | +15.00 п.п. |
| Pipeline invalid | 15.00% | 0.00% | -15.00 п.п. |
| Pipeline empty | 0.00% | 0.00% | +0.00 п.п. |
| Pipeline F1=0 failures | 40.00% | 35.00% | -5.00 п.п. |
| False refusal | 0.00% | 0.00% | +0.00 п.п. |
| MRR@5 | 0.6500 | 0.7792 | +0.1292 |

На всех 2 исходных unanswerable вопросах (dev+eval) оба pipeline вернули непустые ответы: false answer rate 100%. На eval-only unanswerable всего один вопрос, тоже 100%. Ни одна из этих цифр не свидетельствует о решённом отказе.

## По корпусу и языку: старый eval

| Группа | n | EM до | EM после | F1 до | F1 после | Recall@3 до | Recall@3 после |
|---|---:|---:|---:|---:|---:|---:|---:|
| pdf | 5 | 0.00% | 20.00% | 18.00% | 36.41% | 80.00% | 100.00% |
| synthetic | 15 | 13.33% | 13.33% | 48.70% | 52.66% | 66.67% | 86.67% |
| en | 18 | 5.56% | 16.67% | 36.32% | 53.99% | 66.67% | 94.44% |
| ru | 1 | 0.00% | 0.00% | 66.67% | 0.00% | 100.00% | 100.00% |
| kk | 1 | 100.00% | 0.00% | 100.00% | 0.00% | 100.00% | 0.00% |

`synthetic` здесь — авторский общий TXT corpus аудита; PDF — исходный `uploads/embedding.pdf`. Языковые группы пересекаются с корпусными и не складываются в дополнительные примеры. RU/KK n=1 каждый: нулевой final F1 — реальная регрессия именно этих двух вопросов, а не доказательство полной неспособности модели к языку.

## Новые документы (n=8 answerable, только EN)

Эти excerpts — документация `json`, `csv`, `pathlib.Path`, `numpy.linalg.norm`, поставляемая с локально установленными Python/NumPy; source/version сохранены в fixture. В runner они используют общий TXT route, обозначенный `synthetic`, но тексты не были сочинены моделью. Это новый домен по сравнению с embedding lecture PDF, не репрезентативная production выборка.

| Метрика | Baseline | Final | Разница |
|---|---:|---:|---:|
| Oracle EM | 37.50% | 37.50% | +0.00 п.п. |
| Oracle F1 | 66.25% | 66.25% | +0.00 п.п. |
| Pipeline EM | 37.50% | 25.00% | -12.50 п.п. |
| Pipeline F1 | 72.80% | 56.55% | -16.25 п.п. |
| Recall@1 | 100.00% | 75.00% | -25.00 п.п. |
| Recall@3 | 100.00% | 100.00% | +0.00 п.п. |
| Recall@5 | 100.00% | 100.00% | +0.00 п.п. |
| Pipeline invalid | 25.00% | 0.00% | -25.00 п.п. |
| Pipeline empty | 0.00% | 0.00% | +0.00 п.п. |
| Pipeline F1=0 failures | 12.50% | 25.00% | +12.50 п.п. |
| False answer rate (4 absent) | 100.00% | 100.00% | +0.00 п.п. |
| False refusal | 0.00% | 0.00% | +0.00 п.п. |

**Регрессия не скрывается:** на новом corpus лучший первый chunk стал находиться реже, а QA зависит от соседних слов. Соблюдение embedding budget и сохранность source не гарантируют лучший lexical answer. После просмотра нового test параметры не подбирались, QA/RRF/length bonus не включались для искусственного исправления этого числа. Поэтому универсальный рост качества не заявляется. Нужны более разнообразные dev документы и новое независимое подтверждение перед production rollout.

## Индивидуальные улучшения и регрессии

Ниже все answerable eval вопросы, а не только удачные. Полные контексты зафиксированы в quality fixtures; исходные snippets и hit scores доступны в local results. Разница F1 лексическая, не полный judgement смысла.

| Набор / ID | Эталон | Baseline pipeline | Final pipeline | Δ F1 |
|---|---|---|---|---:|
| exposed_audit/q06 | Guido van Rossum | by Guido van Rossum | by Guido van Rossum | +0.000 |
| exposed_audit/q07 | 1991 | in 1991 | in 1980 | -0.667 |
| exposed_audit/q08 | the process by which plants convert light energy into chemical energy | The experimental enzyme is named Zeta - Q7 | is the process by which plants convert light energy into chemical energy | +0.952 |
| exposed_audit/q09 | Mount Wilson in California | in the laboratory | . ↵  ↵ The museum opens at nine and closes at five | -0.333 |
| exposed_audit/q10 | Hemoglobin | is violet lantern | . ↵  ↵ Hemoglobin | +1.000 |
| exposed_audit/q11 | Jupiter | is Jupiter | is Jupiter | +0.000 |
| exposed_audit/q12 | Астана | называется Астана | major European city | -0.667 |
| exposed_audit/q13 | Астана | Астана | on Mount Wilson | -1.000 |
| exposed_audit/q14 | TCP | . TCP | . ↵  ↵ TCP | +0.000 |
| exposed_audit/q15 | copying all project documents to two independent encrypted drives every Friday and checking that the copied files can be restored before deleting any old backup versions | requires copying all project documents to two independent encrypted drives | requires copying all project documents | -0.248 |
| exposed_audit/q16 | Zeta-Q7 | . Photosynthesis | named Zeta-Q7 | +0.667 |
| exposed_audit/q17 | AB-123/X | is violet lantern | on the label | +0.000 |
| exposed_audit/q18 | letters,  maps, and photographs | This telescope | stores letters,  maps, and photographs | +0.889 |
| exposed_audit/q19 | violet lantern | is violet lantern | is violet lantern | +0.000 |
| exposed_audit/q20 | titanium alloy | of titanium alloy | of titanium alloy | +0.000 |
| exposed_audit/q24 | logistic regression | Use logistic regression | by slowly adjusting all ↵ the embeddings | -0.800 |
| exposed_audit/q25 | dot product | on window size | ◦ Minimizethe similarity of the (w , c neg) pairs drawn from the ↵ negative data | +0.000 |
| exposed_audit/q26 | sigmoid | ##ities Sim ( w, c ) ≈w · c | the sigmoid | +1.000 |
| exposed_audit/q27 | Stochastic gradient descent | ##s as the embeddings | ◦ Stochastic gradient descent! ↵ We’ll adjust | +0.667 |
| exposed_audit/q28 | Target embeddings matrix W ↵ Context embedding matrix C | of positive and negative training instances, and an initial set of embedding vectors | learns two sets of embeddings | +0.054 |
| new_documents/d01 | JavaScript Object Notation | ( JavaScript Object Notation | (JavaScript Object Notation | +0.000 |
| new_documents/d02 | simplejson library | maintained simplejson | :mod:`marshal` and :mod:`pickle | -0.500 |
| new_documents/d03 | PEP 305 | by PEP 305 | by PEP 305 | +0.000 |
| new_documents/d04 | reading, writing, and registration of dialects | : reading, writing, and registration of dialects | :mod:`marshal` and :mod:`pickle` modules | -0.800 |
| new_documents/d05 | PurePath | a filesystem | a filesystem | +0.000 |
| new_documents/d06 | a PosixPath or a WindowsPath | a PosixPath or a WindowsPath | a PosixPath or a WindowsPath | +0.000 |
| new_documents/d07 | Matrix or vector norm | or vector norm | or vector norm | +0.000 |
| new_documents/d08 | ord | the ord parameter | the ``ord`` parameter | +0.000 |

Часть улучшений — восстановление полных слов и источника (`Hemoglobin`, `dot product`), часть — другое evidence после пересегментации. RU/KK ухудшения связаны с retrieval/selection: нужный span не оказывается первым; default не выбирает по некалиброванному QA score. Лишние `is/by/for` в oracle остаются ограничением checkpoint.

Уточнение к старому diagnostic report: объяснение q13 через удаление UNK относится к **oracle offset** сравнению (`Астана` → `— Астана`). В старой combined short-chunk таблице `is Paris` было ошибкой retrieval/selection, а не только UNK. Исходный отчёт сохранён как исторический артефакт; здесь эти причины различаются.

## Проверки, не равные quality improvement

128 tests passed, 0 failed. No missing/mismatched tensors при загрузке checkpoint; vocab/tokenizer 119547. FAISS normalization не менялась и исходные numeric ranking tests проходят. Новые индексируемые chunks не выходят за embedding limit. Original weights/tokenizer/notebook и baseline evaluation сохранены. Эти инварианты улучшены независимо от регрессии lexical F1 на новом домене.
