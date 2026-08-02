"""训练结果页视图：训练 run 列表 + 指标摘要 + 曲线预览。

渲染文本构建提取为模块级纯函数 ``format_training_summary``（可单测）。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
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
from services.file_utils import filter_paths, format_mapping, open_path, read_csv_rows


def format_training_summary(run_dir: Path) -> tuple[str, list[Path]]:
    """构建训练 run 摘要文本；返回 (文本, 可用 PNG 曲线列表)。"""
    results_csv = run_dir / "results.csv"
    lines = ["训练结果摘要", "=" * 60, f"训练目录: {run_dir}", ""]
    if results_csv.exists():
        rows = read_csv_rows(results_csv)
        lines.append(f"results.csv: {results_csv}")
        lines.append(f"epoch 数   : {len(rows)}")
        if rows:
            lines.extend(["", "最后一轮指标", "-" * 60])
            lines.extend(
                format_mapping(
                    rows[-1],
                    preferred=(
                        "epoch",
                        "train/box_loss",
                        "val/box_loss",
                        "metrics/precision(B)",
                        "metrics/recall(B)",
                        "metrics/mAP50(B)",
                        "metrics/mAP50-95(B)",
                    ),
                )
            )
    else:
        lines.append("未找到 results.csv。")

    weights = sorted((run_dir / "weights").glob("*.pt"))
    if weights:
        lines.extend(["", "权重文件", "-" * 60])
        lines.extend(f"- {path.name}" for path in weights)
    plots = sorted(run_dir.glob("*.png"))[:20]
    if plots:
        lines.extend(["", "可用图表", "-" * 60])
        lines.extend(f"- {path.name}" for path in plots)
    return "\n".join(lines), plots


class TrainingResultView(QWidget):
    """训练结果浏览（列表/摘要/曲线切换）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._training_plots: list[Path] = []
        self._training_plot_index = 0
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        refresh_button = QPushButton("刷新")
        open_button = QPushButton("打开训练目录")
        refresh_button.clicked.connect(self.refresh)
        open_button.clicked.connect(self._open_selected)

        self.prev_plot_button = QPushButton("上一张图")
        self.next_plot_button = QPushButton("下一张图")
        self.prev_plot_button.clicked.connect(self._show_previous_plot)
        self.next_plot_button.clicked.connect(self._show_next_plot)

        title_label = QLabel("训练结果")
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel("浏览 YOLO 训练 run、最后一轮指标、权重和训练曲线。")
        hint_label.setObjectName("HintText")

        self.filter_edit = QLineEdit("")
        self.filter_edit.setPlaceholderText("筛选训练结果")

        buttons = QHBoxLayout()
        buttons.addWidget(refresh_button)
        buttons.addWidget(open_button)
        buttons.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(title_label)
        left.addWidget(hint_label)
        left.addWidget(self.filter_edit)
        left.addLayout(buttons)
        self.result_list = QListWidget()
        left.addWidget(self.result_list)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.plot_label = QLabel("选择训练结果后显示曲线图")
        self.plot_label.setObjectName("PreviewPane")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setMinimumHeight(300)

        plot_buttons = QHBoxLayout()
        plot_buttons.addWidget(self.prev_plot_button)
        plot_buttons.addWidget(self.next_plot_button)
        plot_buttons.addStretch(1)

        right = QVBoxLayout()
        right.addWidget(self.detail, stretch=1)
        right.addWidget(self.plot_label, stretch=1)
        right.addLayout(plot_buttons)

        list_container = QWidget()
        list_container.setLayout(left)
        detail_container = QWidget()
        detail_container.setLayout(right)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(list_container)
        splitter.addWidget(detail_container)
        splitter.setSizes([360, 900])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.filter_edit.textChanged.connect(lambda _text: self.refresh())
        self.result_list.currentItemChanged.connect(self._show_item)

    # ---- 数据 ----

    def refresh(self) -> None:
        runs_root = desktop_paths.ROOT_DIR / "runs"
        result_dirs = sorted({path.parent for path in runs_root.glob("**/results.csv")}, reverse=True)
        result_dirs = filter_paths(result_dirs, self.filter_edit.text())
        self.result_list.clear()
        for path in result_dirs:
            item_text = str(path.relative_to(desktop_paths.ROOT_DIR)) if desktop_paths.is_relative_to(path, desktop_paths.ROOT_DIR) else str(path)
            self.result_list.addItem(item_text)
            self.result_list.item(self.result_list.count() - 1).setData(Qt.ItemDataRole.UserRole, str(path))
        if self.result_list.count():
            self.result_list.setCurrentRow(0)
        else:
            self.detail.setPlainText("暂无训练结果。运行 odp-train 后会在这里显示。")
            self.plot_label.setText("暂无训练曲线")

    def _show_item(self, item) -> None:
        if item is None:
            return
        run_dir = Path(item.data(Qt.ItemDataRole.UserRole))
        summary, plots = format_training_summary(run_dir)
        self._training_plots = plots
        self._training_plot_index = 0
        self.detail.setPlainText(summary)
        self._show_plot()

    # ---- 曲线 ----

    def _show_previous_plot(self) -> None:
        if not self._training_plots:
            return
        self._training_plot_index = (self._training_plot_index - 1) % len(self._training_plots)
        self._show_plot()

    def _show_next_plot(self) -> None:
        if not self._training_plots:
            return
        self._training_plot_index = (self._training_plot_index + 1) % len(self._training_plots)
        self._show_plot()

    def _show_plot(self) -> None:
        if not self._training_plots:
            self.plot_label.setText("当前训练目录没有可预览的 PNG 曲线")
            self.plot_label.setPixmap(QPixmap())
            return
        path = self._training_plots[self._training_plot_index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.plot_label.setText(f"无法加载图表: {path.name}")
            return
        self.plot_label.setPixmap(
            pixmap.scaled(
                self.plot_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.plot_label.setToolTip(str(path))

    # ---- 打开 ----

    def _open_selected(self) -> None:
        item = self.result_list.currentItem()
        if item is not None:
            open_path(Path(item.data(Qt.ItemDataRole.UserRole)))
