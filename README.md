# Credit Card Fraud Detection Service (MLOps)

Промышленный асинхронный ML-сервис для классификации и выявления кредитного мошенничества (фрода) на базе датасета `creditcard.csv`. Проект реализован с разделением на независимые контуры обучения (**Train**) и предсказания (**Inference**).

## Технологический стек
* **Язык:** Python 3.11 (Alpine)
* **Менеджер зависимостей:** `uv` 
* **API Сервер:** FastAPI + Uvicorn + Pydantic v2
* **Контейнеризация:** Docker, Docker Compose (с общими Volumes)
* **ML-модель:** Scikit-Learn Pipeline (StandardScaler + RandomForestClassifier)
* **Тестирование:** Pytest

## Структура проекта
* `data/` — каталог с исходными данными (`creditcard.csv`).
* `models_registry/` — внутренний регистр (хранилище) версий ML-моделей.
* `src/shared/` — общий код: централизованный логгер, конфигурация `.env` и единый класс препроцессинга данных `DataPreprocessor`.
* `src/train/` — short-lived сервис обучения моделей по требованию.
* `src/inference/` — 24/7 сервис инференса (FastAPI) с поддержкой динамической смены моделей (Hot Swap).
* `tests/` — юнит-тесты для проверки препроцессинга.

## Инструкция по запуску

### 1. Подготовка
Поместите ваш датасет в папку `data/creditcard.csv`: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.
## Для создания модели для предсказания запустите следующую команду:
```bash
docker compose run --rm train
```
## Создайте файл `.env` на основе примера:

PROJECT_NAME="Fraud Detection Service"
ENV="development"
DATA_DIR="data"
MODELS_REGISTRY_DIR="models_registry"
CLASSIFICATION_THRESHOLD=0.5


### 2. Запуск Inference-сервера (24/7)
```bash
docker compose up --build -d inference
```
Интерактивная документация Swagger UI доступна по адресу: http://localhost:8000/docs

### 3. Запуск сессии переобучения модели (По требованию)
```bash
docker compose run --rm train
```
Контейнер автоматически обучит модель, завалидирует метрику F1 (порог > 0.75), сохранит новую версию в регистр, обновит рабочую модель в рантайме FastAPI через эндпоинт `/hotswap` и завершит свою работу.

### 4. Запуск юнит-тестов (Опционально)
```bash
uv run pytest
```
