import logging
import sys

def setup_logger(name: str = 'ml_service') -> logging.Logger:
    logger = logging.getLogger(name)

    # Избегаем дублирования логов, если логгер уже настроен
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger