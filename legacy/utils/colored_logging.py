#!/usr/bin/env python3
"""Colored logging utility module.

Provides a unified colored log formatter for console output, plus a few
convenience helpers for emitting log records at each severity level. Optionally
mirrors records to a (plain, uncolored) log file.
"""

import logging
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Logging formatter that colorizes the level name for console output."""

    # ANSI color codes per log level.
    COLORS = {
        'DEBUG': '\033[36m',    # cyan
        'INFO': '\033[34m',     # blue
        'WARNING': '\033[33m',   # yellow
        'ERROR': '\033[31m',    # red
        'CRITICAL': '\033[35m', # magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format a log record, wrapping the level name in ANSI color codes."""
        # Add color to the level name for console output.
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_colored_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_str: str = '%(asctime)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """Configure the colored logging system.

    Args:
        level: Logging level.
        log_file: Optional path to a log file.
        format_str: Log format string.

    Returns:
        The configured logger instance.
    """
    # Create the console handler (with color).
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(format_str)
    console_handler.setFormatter(console_formatter)

    # Assemble the list of handlers.
    handlers = [console_handler]

    # If a log file was provided, add a file handler (without color).
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure the root logger.
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Force a reconfiguration.
    )

    return logging.getLogger(__name__)


def get_colored_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Get a logger that emits colored output.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional path to a log file.

    Returns:
        The configured logger instance.
    """
    # Apply the base configuration.
    setup_colored_logging(level, log_file)

    # Return the logger with the requested name.
    return logging.getLogger(name)


# Convenience helpers
def info(msg: str, logger: Optional[logging.Logger] = None):
    """Emit a colored log record at INFO level."""
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(msg)


def warning(msg: str, logger: Optional[logging.Logger] = None):
    """Emit a colored log record at WARNING level."""
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.warning(msg)


def error(msg: str, logger: Optional[logging.Logger] = None):
    """Emit a colored log record at ERROR level."""
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.error(msg)


def debug(msg: str, logger: Optional[logging.Logger] = None):
    """Emit a colored log record at DEBUG level."""
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.debug(msg)


def critical(msg: str, logger: Optional[logging.Logger] = None):
    """Emit a colored log record at CRITICAL level."""
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.critical(msg)


if __name__ == "__main__":
    # Smoke-test the colored logging functionality.
    logger = setup_colored_logging(logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\nTesting the convenience helpers:")
    info("convenience helper - info")
    warning("convenience helper - warning")
    error("convenience helper - error")
    debug("convenience helper - debug")
    critical("convenience helper - critical")
