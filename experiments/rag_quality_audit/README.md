# Архив диагностического этапа

Здесь сохранены исходники ранее выполненного аудита и исторический `RAG_QUALITY_DIAGNOSTIC_REPORT.md`. Они относятся к прежнему BERT checkpoint и **не запускаются приложением Streamlit**. Финальная версия приложения описана в корневом README и `reports/APPLICATION_DELIVERY.md`.

Исторические скрипты могут требовать старую локальную модель, training notebook и исходный `uploads/embedding.pdf`. Эти файлы не публикуются вместе с весами; `scripts/setup_models.py` восстанавливает модели готового приложения, а не исторический fine-tuned BERT.

Сгенерированные datasets, извлечённые PDF-тексты, predictions, логи и подробные результаты остаются локальными и игнорируются Git. Нужные небольшие фиксированные QA fixtures текущего evaluation уже находятся в `quality/data/`, компактные versioned metrics — `reports/quality_metrics.json`.

`evaluation/` в корне также сохранён как исторический эксперимент с length-aware span scoring. Он не является production-декодером и не включается в пользовательский запуск.
