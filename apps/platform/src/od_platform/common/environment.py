"""Runtime environment checks."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TextIO

EXPECTED_CONDA_ENV = "odplat"


def set_kmp_duplicate_lib_ok() -> None:
    """设置 KMP_DUPLICATE_LIB_OK=TRUE（幂等）。

    torch 与 matplotlib 在同一进程加载多个 libiomp5md.dll 会触发
    OMP Error #15 崩溃；该变量为官方建议的无害 workaround。
    统一在进程早期（CLI 入口/工具执行）调用，避免各端重复 setdefault。
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def is_expected_environment(expected: str = EXPECTED_CONDA_ENV) -> bool:
    """Return whether the current Python process appears to run in the expected env."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    executable = str(Path(sys.executable)).lower().replace("/", "\\")
    expected_lower = expected.lower()
    return conda_env.lower() == expected_lower or f"envs\\{expected_lower}\\" in executable


def warn_if_not_expected_environment(
    logger: logging.Logger,
    *,
    expected: str = EXPECTED_CONDA_ENV,
) -> bool:
    """Log a warning when the current process does not appear to use the expected env.

    The check is intentionally non-blocking: README and AGENTS.md still define odplat as the
    development environment, but CLI tools should explain the problem instead of failing before
    argparse/service logic can report useful errors.
    """
    if is_expected_environment(expected):
        return True

    logger.warning(
        "当前 Python 环境似乎不是 %s: CONDA_DEFAULT_ENV=%r, executable=%s。"
        "建议先执行 `conda activate %s`。",
        expected,
        os.environ.get("CONDA_DEFAULT_ENV"),
        sys.executable,
        expected,
    )
    return False


def warn_cli_if_not_expected_environment(
    *,
    logger: logging.Logger | None = None,
    stream: TextIO | None = None,
    expected: str = EXPECTED_CONDA_ENV,
) -> bool:
    """Warn from a CLI boundary without requiring logging to be configured first."""
    if is_expected_environment(expected):
        return True

    message = (
        f"当前 Python 环境似乎不是 {expected}: "
        f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}, executable={sys.executable}。"
        f"建议先执行 `conda activate {expected}`。"
    )
    if logger is not None and logger.handlers:
        logger.warning(message)
    else:
        print(f"WARNING: {message}", file=stream or sys.stderr)
    return False
