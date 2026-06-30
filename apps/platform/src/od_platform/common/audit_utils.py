"""审计上下文工具模块。

为高风险运维命令生成可机读的执行上下文，便于事后追踪。

@FileName:   audit_utils.py
@Function:   reset_project 审计上下文采集与落盘
"""

import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> str:
    """执行只读 git 命令并返回文本结果。

    Args:
        args: git 子命令参数。
        cwd: git 执行目录。

    Returns:
        命令标准输出；git 不可用或执行失败时返回 ``"unknown"``。
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _audit_context(
    root_dir: Path,
    argv: list[str],
    tool_name: str = "reset_project",
    tool_version: str = "0.1.0",
) -> dict[str, Any]:
    """采集审计上下文。

    Args:
        root_dir: 工作区根目录。
        argv: 当前命令行参数快照。
        tool_name: 工具名称。
        tool_version: 工具版本。

    Returns:
        可 JSON 序列化的审计上下文字典。
    """
    root = root_dir.resolve()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "tool_version": tool_version,
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "root_dir": str(root),
        "argv": argv,
        "os_info": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "git_commit": _run_git(["rev-parse", "--short", "HEAD"], root),
        "git_status": _run_git(["status", "--short"], root),
    }


def write_audit_record(log_dir: Path, context: dict[str, Any]) -> Path:
    """写入独立审计日志文件。

    Args:
        log_dir: 审计日志目录。
        context: 审计上下文字典。

    Returns:
        写入的审计日志路径。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    audit_file = log_dir / f"reset-project_audit_{timestamp}_{os.getpid()}.log"
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    audit_file.write_text(f"[AUDIT] {payload}\n", encoding="utf-8")
    return audit_file


def append_audit_result(audit_file: Path, result: dict[str, Any]) -> None:
    """追加 reset_project 执行结果到审计日志。

    Args:
        audit_file: 审计日志文件。
        result: 可 JSON 序列化的执行结果。

    Returns:
        None。
    """
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
    with audit_file.open("a", encoding="utf-8") as file:
        file.write(f"[RESULT] {payload}\n")
