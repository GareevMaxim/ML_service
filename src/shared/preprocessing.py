import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.shared.logger import setup_logger

logger = setup_logger("shared.preprocessing")

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Инициализация препроцессора для антифрод-системы.
        Вся работа со скалированием Amount зашита внутри загружаемого Pipeline,
        поэтому здесь мы только подготавливаем структуру признаков.
        """
        self.feature_columns = None 

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Запуск fit() для DataPreprocessor...")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Трансформирует сырые данные. Выделяет признак Hour и удаляет Time.
        Гарантирует СТРОГИЙ порядок колонок для соответствия сохраненному Pipeline.
        """
        logger.info(f"Трансформация датасета. Размер входных данных: {X.shape}")
        
        X_clean = X.copy()
        
        # Feature Engineering: Переводим секунды в часы
        if 'Time' in X_clean.columns:
            X_clean['Hour'] = (X_clean['Time'] // 3600) % 24
            X_clean = X_clean.drop(columns=['Time'])
            logger.info("Признак 'Time' успешно преобразован в 'Hour', колонка 'Time' удалена.")
        else:
            if 'Hour' not in X_clean.columns:
                raise ValueError("В данных отсутствует обязательная колонка 'Time' или 'Hour'.")

        # Удаляем целевую переменную 'Class', если она случайно пришла
        if 'Class' in X_clean.columns:
            X_clean = X_clean.drop(columns=['Class'])

        # Выстраиваем колонки ровно так, как они шли на вход модели в вашем ноутбуке:
        correct_order = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour']
        
        try:
            X_clean = X_clean[correct_order]
        except KeyError as e:
            logger.error(f"Во входных данных отсутствуют необходимые колонки: {str(e)}")
            raise ValueError(f"Входной JSON должен содержать все поля V1-V28, Amount и Time. Отсутствует: {str(e)}")
        
        logger.info("Трансформация данных успешно завершена. Порядок фичей верифицирован.")
        return X_clean

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
