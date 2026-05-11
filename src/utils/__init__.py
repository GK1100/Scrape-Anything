"""
Logging utility — provides a consistent logger across the project.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Create a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "\033[90m%(asctime)s\033[0m | "
                "\033[1m%(name)-24s\033[0m | "
                "%(levelname)-8s | "
                "%(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
