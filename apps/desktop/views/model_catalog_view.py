"""模型目录页视图：浏览/推荐/筛选，一键应用到训练或推理（信号解耦）。

跨页数据（模型回填、系列下拉、状态栏）经信号交由 MainWindow 仲裁。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
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

from od_platform.model_catalog import get_model_info, list_families, list_models, recommend_model


class ModelCatalogView(QWidget):
    """模型目录工作台。

    Signals:
        applied_to_train:  选中模型名（主窗口回填训练页）。
        applied_to_infer:  选中模型名（主窗口回填推理页）。
        families_changed:  系列列表（主窗口转发给任务页）。
        status_changed:    (状态, 详情) 主窗口状态栏。
    """

    applied_to_train = Signal(str)
    applied_to_infer = Signal(str)
    families_changed = Signal(list)
    status_changed = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()
        self.refresh()

    def _build_ui(self) -> None:
        refresh_button = QPushButton("刷新")
        recommend_row = QWidget()
        recommend_layout = QHBoxLayout(recommend_row)
        recommend_layout.setContentsMargins(0, 0, 0, 0)
        recommend_layout.addWidget(QLabel("推荐"))
        self.recommend_edit = QLineEdit("")
        self.recommend_edit.setPlaceholderText("如：最快 / 最准 / yolov8 最准")
        self.recommend_button = QPushButton("推荐")
        recommend_layout.addWidget(self.recommend_edit)
        recommend_layout.addWidget(self.recommend_button)

        self.apply_train_button = QPushButton("应用到模型训练")
        self.apply_infer_button = QPushButton("应用到模型推理")
        apply_row = QWidget()
        apply_layout = QHBoxLayout(apply_row)
        apply_layout.setContentsMargins(0, 0, 0, 0)
        apply_layout.addWidget(self.apply_train_button)
        apply_layout.addWidget(self.apply_infer_button)
        apply_layout.addStretch(1)

        title_label = QLabel("模型目录")
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel("浏览内置 YOLO 系列模型，支持自然语言推荐，可应用到训练/推理。")
        hint_label.setObjectName("HintText")

        self.filter_edit = QLineEdit("")
        self.filter_edit.setPlaceholderText("按名称/描述筛选模型")
        # 系列下拉由 _sync_families() 从 model_catalog 动态填充
        self.family_combo = QComboBox()
        filter_row = QWidget()
        filter_layout = QHBoxLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QLabel("系列"))
        filter_layout.addWidget(self.family_combo)
        filter_layout.addWidget(self.filter_edit)
        filter_layout.addWidget(refresh_button)

        self.model_list = QListWidget()
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)

        left = QVBoxLayout()
        left.addWidget(title_label)
        left.addWidget(hint_label)
        left.addWidget(recommend_row)
        left.addWidget(filter_row)
        left.addWidget(apply_row)
        left.addWidget(self.model_list)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        list_container = QWidget()
        list_container.setLayout(left)
        splitter.addWidget(list_container)
        splitter.addWidget(self.detail)
        splitter.setSizes([420, 820])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.recommend_button.clicked.connect(self.refresh)
        self.apply_train_button.clicked.connect(self._apply_to_train)
        self.apply_infer_button.clicked.connect(self._apply_to_infer)
        self.filter_edit.textChanged.connect(lambda _text: self.refresh())
        self.family_combo.currentTextChanged.connect(lambda _text: self.refresh())
        self.model_list.currentItemChanged.connect(self._show_item)

    # ---- 数据 ----

    def refresh(self) -> None:
        self._sync_families()

        family = self.family_combo.currentText()
        family = None if family == "全部" else family

        if self.recommend_edit.text().strip():
            models = recommend_model(self.recommend_edit.text().strip(), family=family, limit=20)
        else:
            models = list_models(family=family)

        query = self.filter_edit.text().strip().lower()
        if query:
            models = [info for info in models if query in info.name.lower() or query in info.description.lower()]

        self.model_list.clear()
        for info in models:
            item_text = f"{info.name}  [{info.family} {info.variant} | {info.size_category}]"
            self.model_list.addItem(item_text)
            self.model_list.item(self.model_list.count() - 1).setData(Qt.ItemDataRole.UserRole, info.name)
        if self.model_list.count():
            self.model_list.setCurrentRow(0)
        else:
            self.detail.setPlainText("没有匹配的模型。")
        self.status_changed.emit("就绪", f"模型目录 {len(models)} 个")

    def _sync_families(self) -> None:
        """同步自身系列下拉并广播 families_changed（任务页联动）。"""
        families = list_families()
        combo = self.family_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(["全部", *families])
        combo.setCurrentText(current if current in ("全部", *families) else "全部")
        combo.blockSignals(False)
        self.families_changed.emit(families)

    def _show_item(self, item) -> None:
        if item is None:
            return
        info = get_model_info(item.data(Qt.ItemDataRole.UserRole))
        if info is None:
            self.detail.setPlainText("无法读取模型元数据。")
            return
        metric_lines = "\n".join(f"{key:<12}: {value}" for key, value in info.metrics.items())
        lines = [
            "模型详情",
            "=" * 60,
            f"名称          : {info.name}",
            f"系列          : {info.family}",
            f"变体          : {info.variant}",
            f"任务          : {info.task}",
            f"后端          : {info.backend}",
            f"尺寸档位      : {info.size_category}",
            f"参数量        : {info.params_m} M",
            f"CPU 耗时      : {info.speed_cpu_ms} ms",
            "",
            "指标",
            "-" * 60,
            metric_lines,
            "",
            "描述",
            "-" * 60,
            info.description,
        ]
        self.detail.setPlainText("\n".join(lines))

    # ---- 应用 ----

    def _apply_to_train(self) -> None:
        item = self.model_list.currentItem()
        if item is None:
            self.status_changed.emit("提示", "先在模型目录中选择一个模型")
            return
        self.applied_to_train.emit(item.data(Qt.ItemDataRole.UserRole))

    def _apply_to_infer(self) -> None:
        item = self.model_list.currentItem()
        if item is None:
            self.status_changed.emit("提示", "先在模型目录中选择一个模型")
            return
        self.applied_to_infer.emit(item.data(Qt.ItemDataRole.UserRole))
