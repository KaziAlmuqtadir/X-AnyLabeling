import logging
import os
import sys
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Callable, Dict

import termcolor

COLORS: Dict[str, str] = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


def singleton(cls):
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color and record.levelname in COLORS:
            record = self._color_record(record)
        record.asctime = self.formatTime(record, self.datefmt)
        return super().format(record)

    def _color_record(self, record: logging.LogRecord) -> logging.LogRecord:
        def colored(text, color):
            return termcolor.colored(text, color=color, attrs={"bold": True})

        record.levelname2 = colored(
            f"{record.levelname:<7}", COLORS[record.levelname]
        )
        record.message2 = colored(record.msg, COLORS[record.levelname])
        record.asctime2 = termcolor.colored(
            self.formatTime(record, self.datefmt), color="green"
        )
        record.module2 = termcolor.colored(record.module, color="cyan")
        record.funcName2 = termcolor.colored(record.funcName, color="cyan")
        record.lineno2 = termcolor.colored(record.lineno, color="cyan")

        return record


LOG_DIR = os.path.join(os.path.expanduser("~"), ".xanylabeling", "logs")
LOG_FILE = os.path.join(LOG_DIR, "xanylabeling.log")


@singleton
class AppLogger:
    def __init__(self, name="X-AnyLabeling"):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        # The logger's own level is a floor for all handlers; keep it at
        # DEBUG so the file handler can capture debug records even when the
        # console handler is set to a higher level (e.g. INFO).
        self.logger.setLevel(logging.DEBUG)
        self._setup_handler()

    def _setup_handler(self):
        stream_handler = logging.StreamHandler(sys.stderr)
        handler_format = ColoredFormatter(
            "%(asctime)s | %(levelname2)s | %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s"
        )
        stream_handler.setFormatter(handler_format)
        # Matches the CLI's default --logger-level until overridden via
        # set_level()/setLevel().
        stream_handler.setLevel(logging.INFO)
        self.stream_handler = stream_handler
        self.logger.addHandler(stream_handler)

        # Always persist DEBUG+ logs to disk (with tracebacks), independent
        # of the console verbosity, so users can retrieve details of a
        # failure (e.g. a custom model that failed to load) after the fact.
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(module)s:%(funcName)s:%(lineno)s - %(message)s"
                )
            )
            self.logger.addHandler(file_handler)
            self.file_handler = file_handler
        except OSError:
            # Fall back to console-only logging if the log directory/file
            # cannot be created (e.g. read-only home directory).
            self.file_handler = None

    def __getattr__(self, name: str) -> Callable:
        return getattr(self.logger, name)

    def set_level(self, level: str):
        # Only the console handler follows the requested verbosity; the
        # file handler always stays at DEBUG (see _setup_handler).
        self.stream_handler.setLevel(level)

    # Alias so existing call sites using the stdlib-style name
    # (`logger.setLevel(...)`) also only affect console verbosity instead
    # of clobbering the logger's DEBUG floor needed by the file handler.
    setLevel = set_level


logger = AppLogger()
