import os
import httpx
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # Для примера альтернативы
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.shared.config import settings
from src.shared.logger import setup_logger
from src.shared.preprocessing import DataPreprocessor

logger = setup_logger("train.pipeline")

def get_next_version(registry_path) -> str:
    """Автоматически определяет следующий номер версии (v1, v2, v3...)"""
    import re
    max_ver = 1
    if not registry_path.exists():
        return "v2"  # Если папки нет

    for file in os.listdir(registry_path):
        match = re.match(r"model_v(\d+)\.joblib", file)
        if match:
            ver = int(match.group(1))
            if ver > max_ver:
                max_ver = ver
    return f"v{max_ver + 1}"

def run_training_pipeline():
    logger.info("=== Запуск Training Pipeline по требованию ===")

    #Загрузка данных
    data_path = settings.data_path / "creditcard.csv"#на реальном проекте sql-запрос
    if not data_path.exists():
        logger.error(f"Файл с данными не найден по пути: {data_path}")
        return

    logger.info(f"Загрузка свежих данных из {data_path}...")
    df = pd.read_csv(data_path)

    #Разделяем на признаки и таргет
    if 'Class' not in df.columns:
        logger.error("В датасете нет целевой колонки 'Class'!")
        return

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Делаем валидационный сплит, чтобы проверить метрики перед деплоем
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #Инициализация общего препроцессора
    # Наш препроцессор переведет Time в Hour и выстроит правильный порядок колонок
    preprocessor = DataPreprocessor()
     
    #Создание нового пайплайна
    logger.info("Инициализация нового пайплайна модели...")
    from sklearn.preprocessing import StandardScaler
    new_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=50,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Прогоняем данные через препроцессор вручную, так как пайплайн ожидает уже очищенные колонки
    logger.info("Применение общего препроцессинга к тренировочным данным...")
    X_train_clean = preprocessor.fit_transform(X_train)
    X_val_clean = preprocessor.transform(X_val)

    # Обучаем модель внутри пайплайна
    logger.info("Старт обучения модели Random Forest...")
    new_pipeline.fit(X_train_clean, y_train)
    logger.info("Обучение успешно завершено.")

    # Валидация метрик перед деплоем
    logger.info("Валидация новой модели на отложенной выборке...")
    preds = new_pipeline.predict(X_val_clean)
    new_f1 = f1_score(y_val, preds)
    logger.info(f"F1-Score новой модели: {new_f1:.4f}")

    # Задаем жесткий порог качества
    THRESHOLD_F1 = 0.7
    if new_f1 < THRESHOLD_F1:
        logger.warning(f"Валидация провалена! F1-score ({new_f1:.4f}) ниже порога ({THRESHOLD_F1}). Модель отклонена.")
        return
    logger.info(f"Валидация успешна! Модель превосходит порог качества {THRESHOLD_F1}.")
    # Сохранени
    next_version = get_next_version(settings.registry_path)
    model_save_path = settings.registry_path / f"model_{next_version}.joblib"

    logger.info(f"Сохранение новой версии модели в регистр: {model_save_path}")
    joblib.dump(new_pipeline, model_save_path)

    # Автоматический Hotswap (Уведомление Inference сервиса)
    # Когда запустим в Docker, адрес инференса будет http://inference:8000
    # Для локального теста используем localhost
    inference_url = os.getenv("INFERENCE_INTERNAL_URL", "http://127.0.0.1:8000")
    hotswap_endpoint = f"{inference_url}/hotswap?version={next_version}"

    logger.info(f"Отправка триггера на горячую смену модели: {hotswap_endpoint}")
    try:
        # Делаем синхронный POST запрос к FastAPI
        with httpx.Client() as client:
            response = client.post(hotswap_endpoint, timeout=10.0)
            if response.status_code == 200:
                logger.info(f"Успех! Сервис Inference успешно переключился на модель {next_version}.")
            else:
                logger.error(f"Inference ответил ошибкой при hotswap: {response.text}")
    except Exception as e:
        logger.error(f"Не удалось связаться с Inference сервисом для hotswap: {str(e)}")
        logger.warning("Модель сохранена в регистр, но не была обновлена в рантайме инференса.")

if __name__ == "__main__":
    run_training_pipeline()