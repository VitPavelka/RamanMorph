# ramanmorph3/logging_utils.py
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, List


_DEFAULT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
		*,
		level: Any[int, str] = "INFO",
		log_file: Optional[str, Path] = None,
		fmt: str = _DEFAULT_FMT,
		datefmt: str = _DEFAULT_DATEFMT,
		force: bool = False,
) -> None:
	"""
	Configure application-wide logging.

	:param level: Logging level, e.g. "INFO", "DEBUG", or logging.INFO.
	:param log_file: Optional path to a log file. If provided, a FileHandler is added.
	:param fmt: Log message format.
	:param datefmt: Datetime format for log entries.
	:param force: If True, remove existing handlers and reconfigure logging.
	"""
	if isinstance(level, str):
		level = level.upper()

	handlers: List[logging.Handler] = []

	console = logging.StreamHandler(stream=sys.stderr)
	console.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
	handlers.append(console)

	if log_file is not None:
		log_path = Path(log_file)
		log_path.parent.mkdir(parents=True, exist_ok=True)
		file_handler = logging.FileHandler(log_path, encoding="utf-8")
		file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
		handlers.append(file_handler)

	logging.basicConfig(level=level, handlers=handlers, force=force)


def get_logger(name: str) -> logging.Logger:
	"""
	Get a named logger.

	Note: Use module-level loggers: `logger = get_logger(__name__)`.

	:param name: Name of the logger.
	"""
	return logging.getLogger(name)
