import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

from colorlog import ColoredFormatter

ROOT_LOGGER_NAME: str = "od_platform"
DEFAULT_SECTION_WIDTH: int = 60


def format_log_rule(width: int = DEFAULT_SECTION_WIDTH, char: str = "=") -> str:
    """Return a stable terminal/log separator line."""
    return char * max(1, width)


def format_log_section(title: str, width: int = DEFAULT_SECTION_WIDTH, char: str = "=") -> str:
    """Return a centered section title for CLI logs."""
    if width <= len(title) + 2:
        return title
    return f" {title} ".center(width, char)


def format_status_label(status: str) -> str:
    """Normalize short status labels used in CLI logs."""
    labels = {
        "created": "[OK]",
        "existed": "[SKIP]",
        "warning": "[WARN]",
        "error": "[ERROR]",
        "dry_run": "[DRY-RUN]",
    }
    return labels.get(status, f"[{status.upper()}]")


def get_logger(
    base_path: Path,
    log_type: str = "general",
    model_name: str | None = None,
    log_level: int = logging.INFO,
    temp_log: bool = False,
    encoding: str = "utf-8",
    logger_name: str = ROOT_LOGGER_NAME,
) -> logging.Logger:
    """Configure the project root logger with console and file handlers."""
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    log_dir: Path = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:21]
    prefix = "temp" if temp_log else log_type.replace("_", "-")

    filename_parts = [prefix, timestamp]
    if model_name:
        safe_model = "".join(c if c.isalnum() or c in "_-" else "_" for c in model_name)
        filename_parts.append(safe_model)
    log_file: Path = log_dir / ("_".join(filename_parts) + ".log")

    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)-8s - "
        "%(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s%(reset)s "
        "%(level_log_color)s[%(levelname)-8s]%(reset)s "
        "%(thin_white)s%(filename)-24s%(reset)s:"
        "%(blue)s%(lineno)-4d%(reset)s "
        "%(thin_white)s|%(reset)s "
        "%(message_log_color)s%(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "thin_white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red,bg_white",
        },
        secondary_log_colors={
            "level": {
                "DEBUG": "thin_white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red,bg_white",
            },
            "message": {
                "DEBUG": "thin_white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red,bg_white",
            },
        },
        style="%",
    )
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding=encoding, errors="replace")
        except OSError:
            pass
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(format_log_rule())
    logger.info(format_log_section("Logging Ready"))
    logger.info("runtime: %s %s", platform.system(), platform.release())
    logger.info("log type: %s", log_type)
    logger.info("log file: %s", log_file)
    logger.info("log level: %s", logging.getLevelName(log_level))
    logger.info("model name: %s", model_name or "none")
    logger.info(format_log_rule())

    return logger
