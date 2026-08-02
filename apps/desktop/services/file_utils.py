"""桌面端通用文件/文本工具（services 层，可单测）。

open_path / open_selected_parent 依赖 Qt（QDesktopServices），
其余为纯 Python。
"""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QListWidget


def read_json(path: Path) -> dict:
    """读取 JSON；失败返回带错误信息的 dict（结果浏览保持 UI 存活）。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 结果浏览不应崩溃
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为行字典；UTF-8 失败回退 GBK。"""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except UnicodeDecodeError:
        with path.open("r", encoding="gbk", newline="") as handle:
            return list(csv.DictReader(handle))


def format_mapping(data: Any, preferred: tuple[str, ...] = ()) -> list[str]:
    """映射 → 文本行（preferred 键优先，dict/list 值 JSON 化）。"""
    if not isinstance(data, dict):
        return [str(data)]
    lines: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if key in data:
            lines.append(f"{key:<28}: {data[key]}")
            seen.add(key)
    for key, value in data.items():
        if key in seen:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key:<28}: {value}")
    return lines


def filter_paths(paths: list[Path], query: str) -> list[Path]:
    """按查询串（大小写不敏感）过滤路径列表。"""
    query = query.strip().lower()
    if not query:
        return paths
    return [path for path in paths if query in str(path).lower()]


def split_extra_args(text: str) -> list[str]:
    """解析追加 CLI 参数；失败抛 ValueError。"""
    if not text:
        return []
    try:
        return shlex.split(text, posix=False)
    except ValueError as exc:
        raise ValueError(f"追加参数解析失败: {exc}") from exc


def open_path(path: Path) -> None:
    """打开路径（文件打开所在目录）。"""
    target = path if path.is_dir() else path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def open_selected_parent(list_widget: QListWidget) -> None:
    """打开列表选中项所在目录。"""
    item = list_widget.currentItem()
    if item is not None:
        open_path(Path(item.data(Qt.ItemDataRole.UserRole)))
