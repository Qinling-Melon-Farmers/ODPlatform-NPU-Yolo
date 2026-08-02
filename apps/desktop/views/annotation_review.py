"""标注复核页 UI：审计目录选择 + 样本列表 + 画布 + 框编辑表格。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from od_platform.annotation.canvas import _imread_unicode
from od_platform.annotation.writer import read_yolo_label, write_yolo_label
from od_platform.common import paths
from views.annotation_review_view import (
    boxes_to_rows,
    draw_boxes,
    find_image_path,
    list_audit_runs,
    load_review_queue,
    rows_to_boxes,
)

logger = logging.getLogger(__name__)


class AnnotationReviewView(QWidget):
    """标注复核工作台：人工精修 VLM 预标注。"""

    COLUMNS = ("class_id", "x_center", "y_center", "width", "height")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._audit_dir: Path | None = None
        self._review_rows: list[dict[str, str]] = []
        self._current_label_path: Path | None = None
        self._build_ui()
        self.refresh_audit_runs()

    # ---- UI ----

    def _build_ui(self) -> None:
        title_label = QLabel("标注复核")
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel("人工精修 VLM 预标注：加载审计复核队列，编辑框并保存。")
        hint_label.setObjectName("HintText")

        audit_group = QGroupBox("审计目录")
        audit_layout = QHBoxLayout(audit_group)
        self.audit_combo = QComboBox()
        self.refresh_runs_button = QPushButton("刷新")
        self.reload_button = QPushButton("加载复核队列")
        audit_layout.addWidget(self.audit_combo)
        audit_layout.addWidget(self.refresh_runs_button)
        audit_layout.addWidget(self.reload_button)

        self.sample_list = QListWidget()
        self.sample_list.setMinimumWidth(220)

        self.preview_label = QLabel("选择样本后显示图片")
        self.preview_label.setObjectName("PreviewPane")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.box_table = QTableWidget(0, len(self.COLUMNS))
        self.box_table.setHorizontalHeaderLabels(self.COLUMNS)
        self.box_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.add_row_button = QPushButton("添加空行")
        self.delete_row_button = QPushButton("删除选中行")
        self.save_button = QPushButton("保存标注")
        self.save_button.setProperty("primary", True)
        table_buttons = QHBoxLayout()
        table_buttons.addWidget(self.add_row_button)
        table_buttons.addWidget(self.delete_row_button)
        table_buttons.addWidget(self.save_button)
        table_buttons.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(title_label)
        left.addWidget(hint_label)
        left.addWidget(audit_group)
        left.addWidget(QLabel("待复核样本"))
        left.addWidget(self.sample_list)

        right = QVBoxLayout()
        right.addWidget(self.preview_label)
        right.addWidget(self.box_table)
        right.addLayout(table_buttons)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_container = QWidget()
        left_container.setLayout(left)
        right_container = QWidget()
        right_container.setLayout(right)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([320, 920])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        self.refresh_runs_button.clicked.connect(self.refresh_audit_runs)
        self.reload_button.clicked.connect(self.reload_queue)
        self.sample_list.currentItemChanged.connect(self._on_sample_selected)
        self.add_row_button.clicked.connect(self._add_empty_row)
        self.delete_row_button.clicked.connect(self._delete_selected_row)
        self.save_button.clicked.connect(self.save_current)

    # ---- 数据加载 ----

    def refresh_audit_runs(self) -> None:
        self.audit_combo.clear()
        runs = list_audit_runs(paths.RUNS_DIR)
        for run_dir in runs:
            self.audit_combo.addItem(run_dir.name, run_dir)
        if runs:
            self.reload_queue()

    def reload_queue(self) -> None:
        self._audit_dir = self.audit_combo.currentData()
        if self._audit_dir is None:
            return
        self._review_rows = load_review_queue(self._audit_dir)
        self.sample_list.clear()
        for row in self._review_rows:
            self.sample_list.addItem(f"{row.get('image', '?')}  [{row.get('reason', '?')}]")
        if self._review_rows:
            self.sample_list.setCurrentRow(0)
        else:
            self.preview_label.setText("该审计目录没有待复核样本")

    # ---- 样本展示 ----

    def _on_sample_selected(self, current, _previous) -> None:
        if current is None or not self._review_rows:
            return
        row = self._review_rows[self.sample_list.currentRow()]
        label_path = Path(row.get("label_path", ""))
        if not label_path.exists():
            self.preview_label.setText(f"标注文件不存在: {label_path}")
            return
        image_path = find_image_path(label_path, row.get("image", ""))
        if image_path is None:
            self.preview_label.setText(f"找不到图片: {row.get('image')}")
            return

        boxes = read_yolo_label(label_path)
        # 复用 _imread_unicode：兼容 Windows 非 ASCII（中文）路径
        image = _imread_unicode(image_path)
        if image is None:
            self.preview_label.setText(f"无法读取图片: {image_path}")
            return
        self._current_label_path = label_path
        self._show_preview(image, boxes)
        self._fill_table(boxes)

    def _show_preview(self, image, boxes) -> None:
        frame = draw_boxes(image, boxes)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        q_image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _fill_table(self, boxes) -> None:
        rows = boxes_to_rows(boxes)
        self.box_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for col_index, value in enumerate(values):
                self.box_table.setItem(row_index, col_index, QTableWidgetItem(value))

    # ---- 编辑与保存 ----

    def _add_empty_row(self) -> None:
        self.box_table.insertRow(self.box_table.rowCount())
        self.box_table.setItem(self.box_table.rowCount() - 1, 0, QTableWidgetItem("0"))

    def _delete_selected_row(self) -> None:
        row = self.box_table.currentRow()
        if row >= 0:
            self.box_table.removeRow(row)

    def _collect_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for row_index in range(self.box_table.rowCount()):
            values: list[str] = []
            for col_index in range(len(self.COLUMNS)):
                item = self.box_table.item(row_index, col_index)
                values.append(item.text() if item is not None else "")
            rows.append(values)
        return rows

    def save_current(self) -> None:
        if self._current_label_path is None:
            return
        boxes = rows_to_boxes(self._collect_rows())
        write_yolo_label(self._current_label_path, boxes)
        logger.info("已保存 %d 框到 %s", len(boxes), self._current_label_path)
        self.preview_label.setText(f"已保存 {len(boxes)} 框")
