import pytest
import pandas as pd
import numpy as np
from src.shared.preprocessing import DataPreprocessor

def test_time_to_hour_conversion():
    """Тест: Проверяем, что секунды правильно переводятся в часы и колонка Time удаляется."""
    preprocessor = DataPreprocessor()
    
    # Создаем тестовый словарь
    test_data = {
        'Time': [3600.0, 7200.0],
        'Amount': [100.0, 250.0]
    }
    # Автоматически генерируем V1-V28
    for i in range(1, 29):
        test_data[f'V{i}'] = [0.1 * i, -0.1 * i]
        
    raw_data = pd.DataFrame(test_data)
    
    # Запускаем обработку
    processed_data = preprocessor.fit_transform(raw_data)
    
    # Проверки 
    assert 'Time' not in processed_data.columns, "Колонка Time должна быть удалена"
    assert 'Hour' in processed_data.columns, "Колонка Hour должна присутствовать"
    
    # Проверяем правильность перевода секунд в часы:
    assert processed_data['Hour'].iloc[0] == 1
    assert processed_data['Hour'].iloc[1] == 2

def test_transform_without_fit_does_not_crash():
    """Тест: Проверяем общую устойчивость метода transform."""
    preprocessor = DataPreprocessor()
    
    test_data = {'Time': [0.0], 'Amount': [10.0]}
    for i in range(1, 29):
        test_data[f'V{i}'] = [0.0]
        
    raw_data = pd.DataFrame(test_data)
    
    try:
        preprocessor.transform(raw_data)
        success = True
    except Exception:
        success = False
        
    assert success is True, "Метод transform не должен вызывать критических ошибок структуры"
