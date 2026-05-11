"""
Logging utility — provides a consistent logger across the project.
"""

import io
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Create a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Use a UTF-8 stream to avoid UnicodeEncodeError on Windows (cp1252)
        stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | "
                "%(name)-24s | "
                "%(levelname)-8s | "
                "%(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
