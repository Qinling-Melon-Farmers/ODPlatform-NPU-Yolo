"""结果浏览通用视图：评估/质检列表 + 筛选 + 详情。

基类 ``ResultBrowserView`` 负责 UI 与刷新骨架；子类提供
``_scan_paths``（路径来源）与渲染纯函数（可单测）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from services import paths as desktop_paths
from services.file_utils import filter_paths, format_mapping, open_selected_parent, read_json

# ---- 渲染纯函数（可单测） ----

def format_eval_summary(payload: dict[str, Any]) -> str:
    """评估审计 JSON → 摘要文本。"""
    metrics = payload.get("metrics", {})
    lines = [
        "模型评估摘要",
        "=" * 60,
        f"运行名      : {payload.get('run_name') or payload.get('audit_run_name')}",
        f"模型        : {payload.get('model_ref')}",
        f"数据        : {payload.get('data_yaml') or payload.get('config', {}).get('data')}",
        f"创建时间    : {payload.get('created_at')}",
        f"耗时        : {payload.get('elapsed_seconds')} 秒",
        "",
        "指标",
        "-" * 60,
    ]
    lines.extend(format_mapping(metrics, preferred=("fitness", "map50", "map50_95", "precision", "recall")))
    lines.extend(["", "原始 JSON", "-" * 60, json.dumps(payload, ensure_ascii=False, indent=2)])
    return "\n".join(lines)


def format_validation_summary(payload: dict[str, Any]) -> str:
    """质检报告 JSON → 摘要文本。"""
    counts = payload.get("counts", {})
    summary = payload.get("dataset_summary", {})
    results = payload.get("results", [])
    lines = [
        "数据质检摘要",
        "=" * 60,
        f"运行 ID     : {payload.get('run_id')}",
        f"严重级别    : {payload.get('overall_severity')}",
        f"退出码      : {payload.get('exit_code')}",
        f"数据 YAML   : {payload.get('yaml_path')}",
        f"耗时        : {payload.get('duration_seconds')} 秒",
        "",
        "结果计数",
        "-" * 60,
        *format_mapping(counts),
        "",
        "数据集概览",
        "-" * 60,
        *format_mapping(summary),
        "",
        "检查项",
        "-" * 60,
    ]
    for result in results:
        lines.append(f"[{result.get('severity')}] {result.get('name')}: {result.get('summary')}")
    fix_items = payload.get("fix_items") or []
    if fix_items:
        lines.extend(["", "整改清单", "-" * 60])
        lines.extend(f"- {item}" for item in fix_items[:50])
    return "\n".join(lines)


class ResultBrowserView(QWidget):
    """结果浏览基类（列表/筛选/详情/打开）。

    Args:
        runs_root: 运行目录根（测试可注入临时目录）。
    """

    def __init__(self, *, runs_root: Path | None = None) -> None:
        super().__init__()
        self._runs_root = runs_root or desktop_paths.ROOT_DIR
        self._build_ui()
        self._connect_signals()

    # ---- UI ----

    def _build_ui(self) -> None:
        refresh_button = QPushButton("刷新")
        open_button = QPushButton("打开所在目录")
        refresh_button.clicked.connect(self.refresh)
        open_button.clicked.connect(self._open_selected)

        title_label = QLabel(self.tab_title())
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel(self.tab_hint())
        hint_label.setObjectName("HintText")

        self.filter_edit = QLineEdit("")
        self.filter_edit.setPlaceholderText(self.filter_hint())
        self.list_widget = QListWidget()
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)

        left = QVBoxLayout()
        left.addWidget(title_label)
        left.addWidget(hint_label)
        left.addWidget(self.filter_edit)
        buttons = QHBoxLayout()
        buttons.addWidget(refresh_button)
        buttons.addWidget(open_button)
        buttons.addStretch(1)
        left.addLayout(buttons)
        left.addWidget(self.list_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        list_container = QWidget()
        list_container.setLayout(left)
        splitter.addWidget(list_container)
        splitter.addWidget(self.detail)
        splitter.setSizes([420, 820])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.filter_edit.textChanged.connect(lambda _text: self.refresh())
        self.list_widget.currentItemChanged.connect(self._show_item)

    # ---- 子类接口 ----

    def tab_title(self) -> str:
        raise NotImplementedError

    def tab_hint(self) -> str:
        raise NotImplementedError

    def filter_hint(self) -> str:
        return "筛选"

    def _scan_paths(self) -> list[Path]:
        raise NotImplementedError

    def _render_item(self, path: Path) -> str:
        raise NotImplementedError

    def empty_hint(self) -> str:
        return "暂无结果"

    # ---- 数据 ----

    def refresh(self) -> None:
        paths = filter_paths(self._scan_paths(), self.filter_edit.text())
        self.list_widget.clear()
        for path in paths:
            item_text = str(path.relative_to(self._runs_root)) if desktop_paths.is_relative_to(path, self._runs_root) else str(path)
            self.list_widget.addItem(item_text)
            self.list_widget.item(self.list_widget.count() - 1).setData(Qt.ItemDataRole.UserRole, str(path))
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        else:
            self.detail.setPlainText(self.empty_hint())

    def _show_item(self, item) -> None:
        if item is None:
            return
        self.detail.setPlainText(self._render_item(Path(item.data(Qt.ItemDataRole.UserRole))))

    def _open_selected(self) -> None:
        open_selected_parent(self.list_widget)


class EvaluationResultView(ResultBrowserView):
    """模型评估结果浏览（runs/evaluation/**/odp_audit.json）。"""

    def tab_title(self) -> str:
        return "模型评估"

    def tab_hint(self) -> str:
        return "浏览 odp-val 生成的模型评估审计。"

    def filter_hint(self) -> str:
        return "筛选评估结果"

    def _scan_paths(self) -> list[Path]:
        return sorted((self._runs_root / "runs" / "evaluation").glob("**/odp_audit.json"), reverse=True)

    def _render_item(self, path: Path) -> str:
        return format_eval_summary(read_json(path))

    def empty_hint(self) -> str:
        return "暂无模型评估结果。运行 odp-val 后会在这里显示。"


class ValidationResultView(ResultBrowserView):
    """数据质检结果浏览（runs/data_validation/**/report.json|md）。"""

    def tab_title(self) -> str:
        return "数据质检"

    def tab_hint(self) -> str:
        return "浏览 odp-validate 生成的报告和整改清单。"

    def filter_hint(self) -> str:
        return "筛选质检报告"

    def _scan_paths(self) -> list[Path]:
        paths = sorted((self._runs_root / "runs" / "data_validation").glob("**/report.json"), reverse=True)
        if not paths:
            paths = sorted((self._runs_root / "runs" / "data_validation").glob("**/report.md"), reverse=True)
        return paths

    def _render_item(self, path: Path) -> str:
        if path.suffix.lower() != ".json":
            return path.read_text(encoding="utf-8", errors="replace")
        return format_validation_summary(read_json(path))

    def empty_hint(self) -> str:
        return "暂无数据质检报告。运行 odp-validate 后会在这里显示。"
