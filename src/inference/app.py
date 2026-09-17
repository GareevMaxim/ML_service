import asyncio
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Query

from src.shared.config import settings
from src.shared.logger import setup_logger
from src.shared.preprocessing import DataPreprocessor
from src.inference.schemas import TransactionInput, PredictionResponce, HotswapResponce

logger = setup_logger("inference.app") #подгружаем логгер из inference

#Контейнер для хранения модели, препроцессора и текущей версии
ML_CONTEXT = {
    "pipeline": None,
    "preprocessor": None,
    "version": "none"
}
# Блокировка для безопасного обновления модели из разных потоков
model_lock = asyncio.Lock()

def load_model_from_registry(version: str) -> bool:
    try:
        model_file = f"model_{version}.joblib"
        model_path = settings.registry_path / model_file

        if not model_path.exists():
            logger.error(f"Файл модели {model_file} не найден в регистре по пути {model_path}")
            return False

        logger.info(f"Загрузка модели {model_file} в память...")
        pipeline = joblib.load(model_path)

        ML_CONTEXT["pipeline"] = pipeline
        ML_CONTEXT["preprocessor"] = DataPreprocessor()
        ML_CONTEXT["version"] = version

        logger.info(f"Модель {version} успешно загружена и готова к рабооте.")
        return True
    except Exception as e:
        logger.error(f"Критическая ошбика при загрузке модели{version}: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan-контекст: выполняется ОДИН раз при запуске сервера
    и инициализирует последнюю доступную модель (v1 по дефолту).
    """
    logger.info("Старт FastAPI приложения. Инициализация ML-контекста...")
    success = load_model_from_registry("v1")
    if not success:
        logger.warning("Не удалось загрузить дефолтную модель v1 при старте! Сервис запущен без модели.")
    yield
    logger.info("Остановка FastAPI приложения. Очистка контекста")
    ML_CONTEXT.clear()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan = lifespan
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Эндпоинт для Docker Compose, проверка работоспособности"""
    if ML_CONTEXT.get("pipeline") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис активен, но ML модель не загружена"
        )
    return {
        "status": "healthy",
        "current_model_version": ML_CONTEXT["version"]
    }

@app.post("/predict", response_model=PredictionResponce, status_code=status.HTTP_200_OK)
async def predict(payload: TransactionInput):
    """Эндпоинт для выдачи предсказания по транзакции (24/7)."""
    if ML_CONTEXT["pipeline"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLEm,
            detail="Модель не доступна. Попробуйте позже."
        )

    try:
        # 1. Переводим Pydantic модель в Pandas DataFrame (как ожидает препроцессор)
        raw_data = pd.DataFrame([payload.model_dump()])

        preprocessed_data = ML_CONTEXT["preprocessor"].transform(raw_data)

        pipeline = ML_CONTEXT["pipeline"]

        preprocessed_data = preprocessed_data[pipeline.feature_names_in_]

        prediction = int(pipeline.predict(preprocessed_data)[0])
        probabilities = pipeline.predict_proba(preprocessed_data)[0]#Вероятности классов. Для фрода индекс 1
        fraud_prob = float(probabilities[1])

        logger.info(f"Успешное предсказание. Класс: {prediction}, Вероятность фрода: {fraud_prob:.4f}")
        return PredictionResponce(
            is_fraud=prediction,
            fraud_probability=fraud_prob,
            model_version=ML_CONTEXT['version']
        )
    except Exception as e:
        logger.error(f"Ошибка во время инференса модели: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка при расчете предсказания: {str(e)}"
        )

@app.post("/hotswap", response_model=HotswapResponce)
async def hotswap_model(version: str = Query(..., description="Версия модели для загрузки, например 'v2'")):
     """
     Hotswap эндпоинт. Позволяет обновить модель в памяти «на лету» без перезагрузки контейнера.
     Использует asyncio.Lock для потокобезопасности.
     """
     logger.info(f"Получен запрос на горячую замену модели на версию: {version}")

     async with model_lock:
         success = load_model_from_registry(version)
         if not success:
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail=f"Не удалось обновить модель на версию '{version}'. Проверьте логи сервера. "
             )
     return HotswapResponce(
        status = "success",
        message = "Модель успешно обновлена в оперативной памяти без остановки сервиса.",
        loaded_version=version
    )