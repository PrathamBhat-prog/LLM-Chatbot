import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger with a standard format.
    If called multiple times, it won't add duplicate handlers.
    """
    logger = logging.getLogger(name if name else "llm_chatbot")

    if not logger.handlers:
        # Set level
        logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Log format: [LEVEL] 2025-12-08 20:15:00 - message
        formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        logger.propagate = False

    return logger
