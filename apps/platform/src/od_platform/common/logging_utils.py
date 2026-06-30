import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

from colorlog import ColoredFormatter

ROOT_LOGGER_NAME: str = "od_platform"


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
        "%(log_color)s[%(levelname)-8s]%(reset)s "
        "%(cyan)s%(filename)-25s%(reset)s:"
        "%(blue)s%(lineno)-4d%(reset)s "
        "%(log_color)s│ %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red,bg_white",
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

    logger.info("=" * 60)
    logger.info("日志系统初始化完成")
    logger.info("运行环境: %s %s", platform.system(), platform.release())
    logger.info("阶段类型: %s", log_type)
    logger.info("日志文件: %s", log_file)
    logger.info("日志级别: %s", logging.getLevelName(log_level))
    logger.info("模型名称: %s", model_name or "无")
    logger.info("=" * 60)

    return logger
