"""
Centralized Logger for Mango Leaf Disease Classification

Every module should use this logger instead of print().
"""

import logging
from pathlib import Path

from src.config.config import LOG_DIR


LOG_FILE = LOG_DIR / "training.log"


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger.

    Parameters
    ----------
    name : str
        Name of the module requesting the logger.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Output
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger