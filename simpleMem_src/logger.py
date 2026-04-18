# -*- coding: utf-8 -*-
"""统一日志系统"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_logger: Optional[logging.Logger] = None
_log_file_path: Optional[Path] = None


def _init_logger() -> logging.Logger:
    global _log_file_path

    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _log_file_path = log_dir / f"{timestamp}-{os.getpid()}.log"

    logger = logging.getLogger("simpleMem")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = _init_logger()
    return _logger


def get_log_file_path() -> Optional[Path]:
    return _log_file_path
