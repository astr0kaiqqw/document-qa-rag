# Нужно ли переобучать BERT

**Переобучение не выполнялось.** Существующие weights и tokenizer не изменены. Следующий шаг — проверить реальные training artifacts и расширить независимую оценку. Плохие границы у checkpoint остаются ограничением: oracle EM 15%, F1 73.54% на 20 прежних eval вопросах, несмотря на исправленный decoder. Это основание готовить controlled fine-tuning experiment, но не доказательство, что любой retraining поможет.

## Что удалось и не удалось проверить

В `models/notebooks/rag_qa_training.ipynb` cell 17 добавляет предыдущий токен, cell 18 в нынешнем коде — нет. Сохранённый output cell 19 start=136; воспроизведение текущей cell 18 с нормальным tokenizer даёт 137/end=141 (`Saint Bernadette Soubirous`). Словарь исторического checkpoint tokenizer был 5, затем исправлен до 119547; локально tokenizer уже исправен. Execution counts отсутствуют. По этим данным нельзя установить фактический preprocessing финального training run.

Пользователю нужно получить из Colab/Drive: `checkpoint-10950`, `checkpoint-21900` или хотя бы их manifest/config и SHA256, trainer_state/training_args, train/validation tokenized feature sample с input_ids/labels/example_id, Dataset revision/fingerprint и точный исходник preprocess на момент `dataset.map`. Запрашивается перенос существующих артефактов, не повторное обучение.

## Уже подготовлено в репозитории

`training.preprocessing.prepare_qa_features` проверяет каждый annotated char span, поддерживает stable ids, multiple references, only_second, overflow/stride и отдельный provenance map. Нет `start_position -= 1`. CLS/CLS применяется лишь к окнам, где полный gold отсутствует. Labels были проверены regression tests EN/RU/KK, punctuation, long context и multiple gold answers.

`training.prepare` только проверяет/сохраняет features. Формат входного локального JSON:

```json
[{"id":"e1","question":"Which city?","context":"The city is Astana.","answers":{"text":["Astana"],"answer_start":[12]}}]
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m training.prepare --data /path/to/local_qa.json --tokenizer models/final_qa_model --output /path/to/new_features.json --max-length 384 --stride 128
```

Output должен быть новым; существующий файл не перезаписывается. В запуске реализации подготовлено 32 windows из 28 audit examples как smoke check, **не как датасет для fine-tuning**. Benchmark примеры нельзя добавлять в train. CLI сохраняет hashes входных данных и tokenizer. Trainer/GPU runner не запускался и не считается проверенным.

## Воспроизводимый план эксперимента после отдельного одобрения

1. Скопировать approved base/checkpoint в новое read-only input location, зафиксировать SHA256 всех файлов, версии PyTorch/Transformers/Datasets/Accelerate, dataset revision и seed=42. Не перезаписывать `models/final_qa_model` и notebook. Все outputs — новый отдельный каталог.
2. Если actual labels смещены, сначала исправить и вручную проверить случайную выборку 100–500 training features. Если labels правильны, исследовать доменный и языковой разрыв отдельно. Проверить отсутствие train/validation/test пересечения по document id и нормализованному context hash.
3. Использовать полные overflow windows (начальный кандидат max_length=384, stride=128), question-first/only_second, специальные позиции исключать из span evaluation. Подготовить отдельные validation feature maps без потери example_id/offsets. Не оценивать только token classification accuracy.
4. Английский SQuAD v1 не обеспечивает RU/KK и no-answer. Для этих задач нужны отдельные проверенные multilingual и unanswerable annotations с законным доступом/лицензией; большие downloads требуют согласования. Валидировать quality отдельно по языкам, документам и absent-answer cases. Benchmark настоящего аудита оставить вне train/dev.
5. Начальный сравнительный режим: lr=2e-5, batch 8, 2 epochs, weight_decay=0.01; это **предложенные** параметры из исходного notebook, а не подтверждённый optimum. При GPU mixed precision включать только после проверки поддерживаемого оборудования; CPU smoke test без fp16. Не смешивать изменения labels, данных и гиперпараметров в одном необъяснимом опыте.
6. Сравнить сохранённые checkpoints по полноценным example-level EM/F1: объединение window candidates по offsets, максимум по valid gold references, тот же whole-word decoder и без length bonus. Выбирать checkpoint по validation F1 с контролем EM, source validity и no-answer false answers/refusals; не выбирать просто последнюю эпоху. Финальный test использовать один раз после выбора.
7. Для no-answer отдельно обучить/валидировать способность к CLS/null на настоящих отрицательных примерах, а не только truncation negatives. Rejection threshold подбирать на dev, заморозить и проверить на новых документах с целевыми ограничениями false refusals. Не выдавать сырые logits за confidence.
8. Сохранить model + **тот же fast tokenizer**, trainer_state, training_args, dataset/feature manifests, seeds, hashes и evaluation predictions. Проверить повторную загрузку, vocab=119547 и что ordinary text не превращается в UNK. Прогнать полный regression suite и тот же RAG comparison. Новый checkpoint подключать только после review результата.

Для этих шагов потребуются approval на реальное обучение и, если нужны, новые данные/модели. Наличие подготовленной функции preprocessing не означает разрешение или запуск retraining.
