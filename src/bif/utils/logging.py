"""Logging utilities for BIF."""

from __future__ import annotations

import logging


def get_logger(name: str = "bif", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log(msg: str, logger: logging.Logger | None = None, rank: int = 0) -> None:
    _logger = logger or get_logger()
    _logger.info("[rank=%d] %s", rank, msg)
