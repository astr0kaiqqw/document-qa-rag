# Диагностический аудит качества локальной RAG-системы

Дата: 1 ноября 2026, Asia/Almaty. Production-код, веса, notebook, uploads и существующий evaluation не изменены. Все новые файлы находятся в `experiments/rag_quality_audit/`. Никаких внешних LLM API, скачивания моделей, установки зависимостей или переобучения не выполнялось.

## 1. Основной вывод

**Категория F: несколько взаимодействующих проблем.** Сохранённый BERT ошибается даже на идеальном контексте; декодирование дополнительно искажает исходный текст; длинные чанки частично невидимы embedding-модели; интеграция выбирает непустой ответ по сходству документа, не по качеству ответа. Одного изменения chunk size или бонуса за длину недостаточно. Доказательств, что виноват только BERT или только поиск, нет.

**Подтверждено:** неполные/избыточные и иногда внутрисловные ответы BERT при oracle context; повреждение текста через `tokenizer.decode`; отсутствие отказа на вопросах без ответа; потеря хвоста контекста при QA; truncation при embeddings. **Гипотеза:** фактические training labels могли начинаться на один токен раньше; notebook содержит прямое несоответствие сохранённых выводов и текущего кода, но исходные training features отсутствуют.

## 2. Архитектура и обследованный репозиторий

Прочитаны все Python-модули `rag/`, все 7 тестовых модулей, README, requirements, pytest.ini, оба файла evaluation, 73 ячейки notebook и их сохранённые выводы, config/tokenizer и структура локального checkpoint. В `uploads/` один PDF; других исходных документов, отдельного сервера, API и локальных training checkpoints нет. Каталог `evaluation/` уже был untracked до аудита — это не результат нашей работы. README устарел: описывает только загрузчик и 10 тестов. `requirements.txt` содержит лишь pypdf/python-docx/pytest, поэтому не описывает воспроизводимое ML-окружение.

Путь данных: `load_document` → `semantic_chunk_text` (по умолчанию 180 слов + overlap 36) → multilingual MiniLM → нормализованные векторы 384 → `IndexFlatIP` → top-3 → BERT отдельно на каждом чанке → первый непустой ответ с наибольшим retrieval similarity. Альтернатива — фиксированные word windows. «Semantic» chunker использует регулярные выражения и целые предложения; отдельной нейросемантической сегментации нет.

Окружение: существующий `.venv`, Python 3.14.7, torch 2.14.1, transformers 5.20.1, sentence-transformers 6.1.0, faiss-cpu 1.15.1; CPU, 4 потока в диагностике. Полные версии: `environment_freeze.txt`. Обе модели доступны локально, запуск с `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

Локальная QA-модель: `models/final_qa_model`, BertForQuestionAnswering, 12 слоёв, hidden 768, max positions 512, vocab 119547, float32. Fast tokenizer тоже 119547, cased; контрольная строка разбивается нормально. QA head и все 199 tensors загружаются без missing/mismatched weights. Конфигурация и словарь совместимы с mBERT из notebook; **побайтовое совпадение с Colab checkpoint-21900 доказать нельзя**: оригинального checkpoint, контрольных сумм и trainer_state рядом нет. `protected_hashes_before.json` фиксирует текущие артефакты, а не их происхождение.

## 3. Что действительно работает

Существующие тесты: **69 passed**, offline, `pytest -q -p no:cacheprovider`. Это проверка контрактов, не семантического качества. В pipeline-тестах QA и embeddings заменены mock; retriever-тесты используют mock embeddings, отдельные embedding-тесты настоящую модель. Тестов реального QA в исходном наборе нет.

FAISS: документы и запросы нормализуются в `VectorStore.add/search`, embeddings также возвращаются нормализованными. `IndexFlatIP` поэтому соответствует cosine; проверено сравнением настоящего поиска с numpy dot на обоих корпусах. Максимальная ошибка менее 1e-6, norm запроса 1.0. Направление сортировки и query encoding правильные. Все непустые chunks сохраняются, загрузка и переиндексация работают.

## 4. Аудит обучения в Colab

- Ячейки 3, 6, 9: `rajpurkar/squad` (SQuAD v1), `google-bert/bert-base-multilingual-cased`; QA head при первой загрузке ожидаемо инициализирован заново.
- Официальные split: train 87 599, validation 10 570. Отдельного test split нет. Сначала пробное обучение на первых 2000/500; затем в ячейке 26 заново загружен base model, что отделяет пробный этап от финального.
- Финальные параметры (27–29): lr 2e-5, batch 8/8, 2 epochs, weight decay 0.01, fp16, save each epoch, log every 500. Явного seed нет; Trainer использует свои defaults. Финальная loss-кривая в output отсутствует.
- Tokenization (18): question первым, context вторым, only_second, max_length 384, padding max_length, offset_mapping и sequence_ids. Это правильный порядок. Используется первый annotated answer; char end = start + len(text).
- Реальная функция cell 18 в **текущем тексте** корректно находит содержащие ответ start/end tokens на проверенных примерах; offsets удаляются до model.forward. Ответы за пределами первых 384 токенов получают CLS/CLS. Нет overflow/stride, поэтому такие примеры теряют supervision ответа. Это не доказательство неправильного offsets alignment.
- Cell 17 содержит лишнее `start_position -= 1`. В первом примере получает `to Saint Bernadette Soubirous`. В cell 18 такого вычитания нет. Однако сохранённый output cell 19 показывает start=136, а воспроизведение **нынешней** cell 18 с локальным исправным tokenizer даёт start=137/end=141 и точный ответ. Это подтверждённая несогласованность notebook. У ячеек execution_count=null, восстановить фактическую историю запуска невозможно. Нельзя объявлять весь train dataset испорченным только по этому следу.
- Потеря: стандартное среднее двух CrossEntropyLoss для start/end; это подтверждено исходным кодом **установленного** BertForQuestionAnswering. В notebook loss не переопределён; точный исходник старой версии из Colab локально не сохранён.
- Eval dataset передан Trainer, но eval_strategy и load_best_model_at_end не заданы. Последний checkpoint-21900 выбран вручную после ошибки save_model (cell 30), восстановлен (33) и сохранён (34). Сравнения checkpoint-10950 и 21900 по EM/F1 нет.
- Исторический tokenizer checkpoint после загрузки имел vocab=5 (43, 59, 64), обычные слова становились UNK. Исправления 44 и 65–67 загружают базовый tokenizer и сохраняют его на Drive. Локально этот дефект **не воспроизводится**.
- Исторические EM=25.00%, F1=72.37% (53) — первые 100 validation examples после исправления tokenizer, independent argmax start/end, max length 384, decode. Берётся максимум по всем reference answers, нормализация SQuAD-подобная. Это действительно вывод notebook, а не новый замер. Маленький последовательный срез тематически смещён; validation уже использовался для отладки, это не нетронутый финальный test. Прямой утечки train в validation в коде не найдено, но независимость на уровне параграфов/дубликатов по отсутствующим данным не проверена. Английский SQuAD не подтверждает качество QA на русском/казахском.
- Воспроизведение cell 18 на наших 26 answerable examples: два длинных контекста получают CLS из-за truncation, остальные имеют правильные границы по видимому ответу. Различия пробелов/пунктуации после decode не означают ошибку token labels; см. `training_label_checks.json`.

## 5. Протокол экспериментов и ограничения метрик

28 заранее зафиксированных вопросов в `dataset.json`: 26 answerable + 2 unanswerable, 7 dev (6 answerable) + 21 eval (20 answerable). 22 вопроса с авторскими контролируемыми контекстами, 6 с дословными страницами реального PDF. Первые 5 авторских вопросов адаптированы из существующего evaluation; они оставлены в dev. Включены определения, числа, имена, многословные и длинные ответы, начало/конец контекста, ответ после 512 токенов, пунктуация, RU и KK. Для каждого answerable reference проверен как точная подстрока context. PDF-ответы сверены с извлечённой страницей; visual QA ограничен отсутствием Poppler (`pdftoppm/pdfinfo`), новые зависимости не устанавливались.

EM/F1: lower, удалить ASCII punctuation и английские артикли, whitespace collapse, F1 по multiset слов. Это лексические метрики: не измеряют полностью смысл и не требуют полного предложения. `valid_span` = непустая точная подстрока исходного поданного контекста; частичный корень слова тоже формально span, хотя ответ плохой. Invalid rate = 1 − valid_span; отдельно empty rate и F1=0 failure rate. Для no-answer точным ответом считается пустая строка, метрики считаются отдельно.

Retrieval Recall@K в этом отчёте — **доля вопросов с хотя бы одним чанком, содержащим нормализованный gold span**, а не оценка релевантности всех passages. MRR ограничен top-5. Короткие общие gold spans могут завысить recall. Показатель также включает потерю полного span при chunking; если ни один chunk не содержит весь gold, recall=0. `support_chunk_indices`, все тексты hits и scores сохранены. В synthetic corpus 22 вопроса используют общий TXT из уникальных contexts; PDF вопросы ищутся только в настоящем PDF. Это контролируемый стресс-тест, не production benchmark.

Dev/eval назначены до запуска; ни тексты, ни reference, ни параметры по результатам eval не менялись. Фиксированные варианты запускались и на dev, и на eval; дополнительный integration comparison — разведочный, независимым вторым blind test не является. Малый dev и один PDF не позволяют доказать переносимость выигрыша, статистическую значимость или оптимальность параметров.

## 6. BERT отдельно от RAG — первый большой эксперимент

`QAModel.answer` вызван напрямую для всех 28 примеров до экспериментов с retrieval. Ни PDF loader, ни FAISS, ни embeddings в этих предсказаниях не участвуют.

| Oracle QA, отложенные 20 answerable | EM | Token F1 | Invalid span | Пустые |
|---|---:|---:|---:|---:|
| baseline | 10.00% | 54.71% | 35.00% | 0.00% |
| offset | 5.00% | 59.71% | 0.00% | 0.00% |
| window_offset | 5.00% | 63.71% | 0.00% | 0.00% |
| max90 | 5.00% | 59.71% | 0.00% | 0.00% |
| top5 | 5.00% | 59.71% | 0.00% | 0.00% |
| length_alpha2 | 10.00% | 68.15% | 0.00% | 0.00% |
| top5_length_alpha2 | 10.00% | 68.15% | 0.00% | 0.00% |

Уже baseline на идеальных контекстах имеет низкий EM. Для `Who created Python?` ответ `by Guido van Rossum`; `What does NLP stand for?` → `for Natural Language Processing`; Hemoglobin → `##bin`. Это исключает retrieval как единственную причину. Из 20 eval вопросов baseline имеет 5 ответов с F1=0, то есть 25%; EM error rate 90%.

Offsets сохраняют оригинальные символы, убирают искусственные `##` и пробелы, но не исправляют выбранные моделью смысловые границы. Снижение EM после offsets реально: казахское `Астана` превращается в `— Астана`, поскольку baseline пропускает UNK при decode, а offsets честно возвращает символ. Нельзя утверждать, что эта коррекция всегда повышает EM.

Стандартный `transformers.pipeline('question-answering', ...)` попытались запустить на том же объекте модели; установленная версия выдаёт KeyError: task removed. Никаких других моделей или text-generation вместо QA не использовано. Ошибка: `hf_limitation.json`. Наши offsets/valid-span варианты — отдельные прямые реализации, **не успешный запуск стандартного HF pipeline**.

## 7. Аудит QA inference

`rag/qa_model.py:43`: правильный порядок question/context, only_second, 512, attention_mask/token_type_ids передаются через **inputs, eval/no_grad включены. `sequence_ids==1` корректно исключает question и special tokens. Полный поиск допустимых start≤end до 30 tokens корректно максимизирует сумму logits; это не independent argmax и не Top-5. Ускорение через ограниченные top-k кандидаты не является исправлением качества.

Подтверждённые ограничения:

1. `qa_model.py:93`: token decode вместо вырезания оригинальных char offsets. Стирает/добавляет символы и может вернуть `##` в начале. Mask контекста при этом правильный.
2. Нет overflow/windows: gold после 512 токенов не поступает модели. q19 без окна F1=0, с окном `is violet lantern`, F1=0.8. На обычных production чанках в этом корпусе QA window не нужен: они помещаются в 512 с вопросом.
3. Hard cap 30 wordpieces: эталон q15 требует 33 tokens и целиком недоступен span decoder. Однако увеличение до 90 само по себе **не изменило** выбранные ответы: модель предпочитает короткий span.
4. Разрешены границы внутри WordPiece-слова. Даже offsets могут выдать `bin`/`fier`; это корректная подстрока, но плохой ответ.
5. Нет no-answer branch, CLS comparison, threshold или калиброванной confidence. На обоих unanswerable oracle вопросах все варианты возвращают содержательный ложный span; abstention accuracy=0/2. На SQuAD v1 отсутствие support=no-answer ожидаемо, CLS labels из truncation не превращают модель в обученную SQuAD v2 QA.
6. Сумма logits пригодна для сравнения spans внутри одного input, но не является вероятностью правильного ответа; напрямую между чанками она не калибрована.

## 8. Поиск, extraction и chunking

В реальном extracted PDF видны склейки `asfeaturesin`, `maygeneralizebetter`, `carandautomobileare`, `Maximizethe`, служебные номера слайдов, разорванные формулы и символы private-use. `document_loader.py:20` только соединяет результаты pypdf, не исправляет их. Это подтверждённое свойство extracted text; визуальная сверка исходной типографики недоступна. Автоматическое «исправление» всех склеек не выполнялось: без словаря/визуальной проверки оно может портить формулы и термины.

`semantic_chunker.py:128–135` добавляет целые overlap units **после** построения базовых чанков. Поэтому chunk_size — не строгая максимальная длина. У original PDF максимум 232 слова при target 180; у smaller максимум 96 при target 48. Overlap может быть больше 36 слов из-за длинного unit; это объясняет размеры, а не потерю всех слов.

Embedding model `paraphrase-multilingual-MiniLM-L12-v2` имеет `max_seq_length=128`. Все 9 исходных PDF чанков и все 9 synthetic чанков длиннее этого лимита. Получаем вектор префикса, хотя BERT позже получает полный chunk. В 12 из 32 вхождений gold spans в исходных чанках ответ целиком за видимым для embeddings префиксом (`additional_evidence.json`). Это объясняет часть несовпадений поиска и контекста; причинная роль для каждого отдельного вопроса требует более крупной token-aware абляции.

| Retrieval eval, n=20 | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| original | 55.00% | 70.00% | 80.00% | 0.650 |
| smaller_chunks | 75.00% | 90.00% | 90.00% | 0.825 |
| fixed_word_chunks | 50.00% | 70.00% | 75.00% | 0.593 |

У smaller chunks embedding truncation исчезает на synthetic (0/32), но остаётся на PDF (22/34): word count не гарантирует token budget, особенно для формул. Исправление векторной нормализации не нужно. Top-K меняет доступность evidence, но production selection обычно всё равно выбирает top-1: любой доступный контекст порождает непустой QA span.

## 9. End-to-end и абляции

A — вручную проверенный oracle context; B — отдельные retrieved chunks с исходным QA и исходным правилом выбора; C — настоящий `RAGPipeline.load_document/ask` на synthetic TXT и исходном PDF. Везде одинаковые questions, веса, tokenizer и scoring. Для B чанки построены из extracted text, поэтому сравнение B/C проверяет интеграцию; оно не отделяет extraction от chunking как независимую переменную. B и C совпали по **всем 28 строкам**, включая no-answer. Независимой decoder-ошибки между этими путями не найдено.

| Условие, eval n=20 | EM | F1 | Invalid | F1=0 failure |
|---|---:|---:|---:|---:|
| A | 10.00% | 54.71% | 35.00% | 25.00% |
| B | 10.00% | 41.02% | 15.00% | 40.00% |
| C | 10.00% | 41.02% | 15.00% | 40.00% |
| smaller_chunks | 5.00% | 37.38% | 20.00% | 50.00% |
| fixed_word_chunks | 10.00% | 35.97% | 25.00% | 55.00% |
| offset_only | 5.00% | 39.36% | 0.00% | 40.00% |
| window_only | 5.00% | 39.36% | 0.00% | 40.00% |
| qa_score_ranking | 5.00% | 47.13% | 0.00% | 30.00% |
| combined_short_offset_window | 5.00% | 44.05% | 0.00% | 40.00% |

Empty answer rate во всех этих answerable условиях 0%. На двух no-answer full pipeline также возвращает ложные непустые ответы. Отсутствие пустых ответов не означает отсутствие ошибок.

Минимальные варианты: offset_only меняет только извлечение исходного текста; window_only добавляет QA windows; smaller_chunks меняет размер/overlap при прежнем BERT; qa_score_ranking использует offset spans и выбирает по максимальной сумме QA logits вместо similarity (его эффект сравнивать с offset_only); combined_short_offset_window объединяет короткие чанки, offsets и окна без length bonus. Все находятся только в диагностических scripts; исходный класс RAGPipeline не патчится.

### Таблица original vs experimental

Ниже experimental — **combined_short_offset_window**, а не победитель, выбранный по test. Показываем и улучшения, и ухудшения. Все значения для eval n=20.

| Метрика | Original Pipeline | Experimental Pipeline | Разница |
|---|---:|---:|---:|
| Exact Match | 10.00% | 5.00% | -5.00 п.п. |
| Token F1 | 41.02% | 44.05% | +3.02 п.п. |
| Recall@1 | 55.00% | 75.00% | +20.00 п.п. |
| Recall@3 | 70.00% | 90.00% | +20.00 п.п. |
| Invalid answer rate | 15.00% | 0.00% | -15.00 п.п. |

Интерпретация: улучшены покрытие поиска и точность воспроизведения символов, но EM регрессирует. Самые короткие чанки при исходном decode ухудшили F1, хотя recall вырос. qa_score_ranking повышает aggregate F1, но снижает EM и не калиброван; это исследовательский результат, а не готовое правило production.

## 10. Top-5 и length-aware scoring

В `evaluation/evaluate_qa.py` **все** допустимые spans, не Top-5, плюс `alpha*log1p(length)` при alpha 0/2. Набор всего 5 вопросов, нет независимого test. Эти файлы существовали до аудита и не изменены.

Мы отдельно проверили top-5 start × top-5 end и length bonus alpha=2. Top-5 дал те же ответы, что полный поиск на этом наборе. У length bonus есть реальный выигрыш oracle F1, но также усиление лишнего текста: q27 F1 0.667→0.333, no-answer q21 разрастается до почти всего context, q19 выдаёт повторяющийся бессмысленный текст. q02 становится полнее; Hemoglobin восстанавливается целиком. Это не универсальная коррекция; рекомендуется не включать бонус автоматически. Max90 без бонуса ничего не улучшил.

## 11. Конкретные примеры исходного и экспериментального pipeline

В колонке experiment — combined_short_offset_window; выбранные строки включают регрессии. Полные predictions/context/reference/scores: JSON-файлы рядом.

| ID | Вопрос | Эталон | Original C | Experiment | Изменение F1 |
|---|---|---|---|---|---:|
| q02 | What is the goal of learning word vectors? | to adjust those word vectors to maximize the similarity of positive word pairs and minimize the similarity of negative word pairs | is to adjust those word vectors to maximize the similarity of positive word pairs and minimize the similarity of negative word pairs | is to adjust those word vectors | -0.574 |
| q06 | Who created Python? | Guido van Rossum | by Guido van Rossum | by Guido van Rossum | +0.000 |
| q08 | What is photosynthesis? | the process by which plants convert light energy into chemical energy | The experimental enzyme is named Zeta - Q7 | is the process by which plants convert light energy into chemical energy | +0.952 |
| q10 | What carries oxygen in the blood? | Hemoglobin | is violet lantern | bin | +0.000 |
| q13 | Қазақстанның астанасы қай қала? | Астана | Астана | is Paris | -1.000 |
| q16 | What is the name of the experimental enzyme? | Zeta-Q7 | . Photosynthesis | named Zeta-Q7 | +0.667 |
| q19 | What is the access code? | violet lantern | is violet lantern | is violet lantern | +0.000 |
| q25 | How is similarity computed? | dot product | on window size | ed from dot product | +0.667 |
| q27 | How are the word weights learned? | Stochastic gradient descent | ##s as the embeddings | learned classifier weights as the word embeddings | +0.000 |
| q28 | Which two sets of embeddings does SGNS learn? | Target embeddings matrix W ↵ Context embedding matrix C | of positive and negative training instances, and an initial set of embedding vectors | learns two | -0.100 |

Различия возникают потому, что новая сегментация меняет и ранжирование, и соседние токены для BERT. Улучшение символов через offsets не гарантирует правильную смысловую границу; сокращение контекста может убрать отвлекающий текст или полезную связь. q13 — отдельная регрессия из-за удаления UNK в исходном decode, а не смены весов. q19 показывает, что retrieval может найти хвост длинного документа даже когда oracle без windows его отрезает.

## 12. Первопричины и что ещё не доказано

| Наблюдение | Статус | Где | Практический смысл |
|---|---|---|---|
| Ошибки при oracle context | Подтверждено на 28 случаях | saved BERT + decoder | retrieval не объясняет всё |
| Некорректное восстановление исходных символов | Подтверждено | qa_model decode | offsets чинят формат, не знание |
| Однотокенный сдвиг исторической разметки | Сильная гипотеза | notebook 17–20, mismatch source/output | сначала получить реальные train features |
| Мультиязычный tokenizer сломан сейчас | Опровергнуто локальной загрузкой | final_qa_model | не заменять tokenizer вслепую |
| Поломана cosine-нормализация | Не обнаружено, численно проверено | vector_store | не менять FAISS metric |
| Чанки длиннее embedding budget | Подтверждено | semantic_chunker → embeddings | vector описывает только префикс |
| Выбор ответа по retrieval score | Подтверждённое поведение | pipeline.ask | лучший ответ в top-3 может игнорироваться |
| Нужен обязательный length bonus | Не подтверждено как универсальное решение | experimental decoder | есть измеренные регрессии |
| Нужен немедленный retraining | Не доказано | provenance/labels отсутствуют | сначала стабилизировать и расширить проверку |

Можно без переобучения исправить offsets, покрытие длинных inputs, token-aware segmentation, улучшить выбор evidence и добавить честное поведение при отсутствии надёжного ответа после отдельной калибровки. Нельзя обещать, что это устранит неправильные границы, выученные checkpoint. Если реальный training dataset подтвердит shift или обученный checkpoint остаётся слабым на расширенном oracle benchmark, повторное fine-tuning будет обосновано — отдельным согласованным шагом.

## 13. Ограничения и следующие шаги

1. Зафиксировать правильный checkpoint/tokenizer hash, trainer_state и реальные tokenized training features из Colab; проверить 100–500 labels, особенно ответы в начале слова и за лимитом 384. Сравнить с текущим cell 18. Одной notebook-иллюстрации недостаточно для решения о переобучении.
2. Расширить независимый набор: несколько настоящих документов, разные темы, EN/RU/KK, множественные правильные spans, no-answer. Разделять train/dev/test по документам и темам; не считать наш synthetic benchmark оценкой качества всех пользователей.
3. После согласования production-патча: offsets с source coordinates и QA score metadata; затем token-aware chunk budget под embedding tokenizer. Размер 48 слов из эксперимента не гарантирует 128 tokens и не является найденным optimum.
4. Отдельно проверить sentence/whole-word boundary constraints и calibrated answer selection. Не удалять автоматически первые слова ответа: это может скрыть дефект обучения и испортить корректные имена.
5. Сделать отдельную dev/test калибровку отказа от ответа. SQuAD v1 не даёт доказанной no-answer способности; не объявлять CLS score вероятностью. Ни similarity threshold, ни probability threshold здесь не подбирались.
6. Закрепить зависимости после решения о поддерживаемой версии Transformers. Стандартный HF QA pipeline в текущей версии отсутствует; сравнение через другую среду потребует явного решения, среда в этом аудите не заменялась.
7. PDF extraction проверен текстово; визуальный просмотр не выполнен из-за отсутствия Poppler. Шум формул/таблиц и DOCX tables не исправлялись. Нет production latency benchmark, полного SQuAD rerun, доверительных интервалов и доказанной эквивалентности весов Colab.

## 14. Воспроизведение и артефакты

Из корня репозитория, в существующем `.venv`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python experiments/rag_quality_audit/build_dataset.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/rag_quality_audit/run_audit.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/rag_quality_audit/run_integration_experiments.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/rag_quality_audit/check_evidence.py
.venv/bin/python experiments/rag_quality_audit/make_report.py
```

Scripts перезаписывают только свои результаты в experimental directory. Повторный запуск `run_audit.py --oracle-only` пересчитывает только изолированный QA. Сравнение float scores в декодере использует сумму Python float, как production; тексты/параметры dataset после первого запуска не менялись. Незначительные числовые расхождения между версиями backend возможны.

- `dataset.json`: все 28 контекстов, точные gold spans, split, page number.
- `oracle_results.json`: ответы, EM/F1, span validity, start/end chars, logits, CLS score; baseline через неизменённый QAModel, диагностические режимы через тот же checkpoint.
- `retrieval_results.json`, `RETRIEVAL_DETAILS.md`: все retrieved contexts и similarity scores, support indices, recall/MRR.
- `end_to_end_results.json`, `integration_experiments.json`: B/C и отдельные/комбинированные изменения.
- `*_summary.json`: все агрегаты раздельно dev/eval/all; no-answer не смешивается с answerable.
- `chunk_statistics.json`, `additional_evidence.json`, `training_label_checks.json`: budgets, математическая проверка cosine, notebook-label reproduction.
- `notebook_review.txt`: текст всех ячеек/сохранённые текстовые outputs, без выполнения notebook.
- `protected_hashes_before.json`, `integrity.json`: проверка неизменности исходных артефактов.

## 15. Статус простыми словами

**1. Что работает** — загрузка, построение индекса, cosine-поиск и технические контракты; 69 тестов проходят, локальные веса/tokenizer загружаются.

**2. Что работает неправильно** — смысловые границы ответов, восстановление текста, покрытие embeddings, окончательный выбор ответа и отказ при отсутствии ответа.

**3. Главная причина плохих ответов** — сочетание слабых границ у checkpoint и потерь при передаче/выборе контекста. Историческая ошибка обучения вероятна, но не доказана по реальным training features.

**4. Что я проверил экспериментально** — 28 oracle вопросов, отдельный поиск и полный pipeline, исходный PDF, альтернативные чанки, offsets, окна, длину, Top-5, QA ranking и notebook labels.

**5. Какие изменения улучшили результаты** — offsets убрали несуществующие в исходнике строки; короткие чанки подняли recall; окна нашли ответ за пределом 512; некоторые варианты повысили F1, но одновременно снизили EM. Универсального победителя нет.

**6. Нужно ли переобучать BERT** — сейчас не запускать: сначала подтвердить реальную разметку и checkpoint; затем решать по независимому oracle тесту. Реальное переобучение не проводилось.

**7. Что нам делать дальше** — получить training artifacts, расширить benchmark, согласовать минимальные production изменения и проверять каждое отдельно. Экспериментальные исправления в основной код не внесены.
