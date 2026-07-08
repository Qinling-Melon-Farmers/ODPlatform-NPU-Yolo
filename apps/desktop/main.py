"""PySide6 desktop workbench for SteelDefect Studio."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
PLATFORM_SRC = ROOT_DIR / "apps" / "platform" / "src"
if str(PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(PLATFORM_SRC))

try:
    import cv2
    from PySide6.QtCore import Qt, QThread, QUrl, Slot
    from PySide6.QtGui import QDesktopServices, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - optional desktop dependencies.
    print(
        "桌面端需要 PySide6 和 opencv-python。请在 odplat 环境中安装后再运行：\n"
        "  conda activate odplat\n"
        "  pip install PySide6 opencv-python\n"
        f"当前缺失: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

from infer_worker import InferWorker  # noqa: E402


class MainWindow(QMainWindow):
    """Local desktop workbench for inference and result inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SteelDefect Studio")
        self.resize(1440, 900)
        self._thread: QThread | None = None
        self._worker: InferWorker | None = None
        self._last_output_dir: Path | None = None

        self.model_edit = QLineEdit(str(_default_model()))
        self.source_edit = QLineEdit(str(_default_source()))
        self.runtime_edit = QLineEdit("")
        self.pipeline_edit = QLineEdit(str(ROOT_DIR / "apps" / "platform" / "configs" / "runtime" / "infer_pipeline.yaml"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment"])
        self.conf_edit = QLineEdit("0.25")
        self.iou_edit = QLineEdit("0.70")
        self.device_edit = QLineEdit("0")
        self.name_edit = QLineEdit("desktop-steel-demo")
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 10000)
        self.max_det_spin.setValue(300)
        self.classes_edit = QLineEdit("")
        self.classes_edit.setPlaceholderText("如 0,2,5；空=全部类别")
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 1_000_000)
        self.max_frames_spin.setSpecialValueText("不限")
        self.save_check = QCheckBox("保存推理结果")
        self.threaded_check = QCheckBox("多级流水线")
        self.threaded_check.setChecked(True)

        self.start_button = QPushButton("启动")
        self.start_button.setProperty("primary", True)
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.open_output_button = QPushButton("打开输出目录")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.open_output_button.setEnabled(False)

        self.image_label = QLabel("等待启动推理")
        self.image_label.setObjectName("PreviewPane")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(860, 560)
        self.status_label = QLabel("状态：就绪")
        self.status_label.setObjectName("StatusLabel")
        self.summary_label = QLabel("帧数 0 | 检测 0 | FPS 0.00 | 输出目录 -")
        self.summary_label.setObjectName("SummaryLabel")
        self.infer_log = QTextEdit()
        self.infer_log.setReadOnly(True)
        self.infer_log.setMaximumHeight(160)
        self.infer_log.setPlaceholderText("推理日志会显示在这里")

        self.eval_list = QListWidget()
        self.eval_detail = QTextEdit()
        self.eval_detail.setReadOnly(True)
        self.validation_list = QListWidget()
        self.validation_detail = QTextEdit()
        self.validation_detail.setReadOnly(True)
        self.training_list = QListWidget()
        self.training_detail = QTextEdit()
        self.training_detail.setReadOnly(True)

        self._build_layout()
        self._connect_signals()
        self._refresh_all_result_tabs()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name.
        self._stop_worker()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.addWidget(self._build_header())

        tabs = QTabWidget()
        tabs.addTab(self._build_inference_tab(), "推理")
        tabs.addTab(
            self._build_browser_tab(
                title="模型评估",
                hint="浏览 odp-val 生成的模型评估审计。",
                list_widget=self.eval_list,
                detail_widget=self.eval_detail,
                refresh_callback=self._refresh_evaluation_results,
                open_callback=self._open_selected_eval,
            ),
            "模型评估",
        )
        tabs.addTab(
            self._build_browser_tab(
                title="数据质检",
                hint="浏览 odp-validate 生成的报告和整改清单。",
                list_widget=self.validation_list,
                detail_widget=self.validation_detail,
                refresh_callback=self._refresh_validation_reports,
                open_callback=self._open_selected_validation,
            ),
            "数据质检",
        )
        tabs.addTab(
            self._build_browser_tab(
                title="训练结果",
                hint="浏览 YOLO 训练 run、最后一轮指标和权重摘要。",
                list_widget=self.training_list,
                detail_widget=self.training_detail,
                refresh_callback=self._refresh_training_results,
                open_callback=self._open_selected_training,
            ),
            "训练结果",
        )
        root.addWidget(tabs)
        self.setCentralWidget(central)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("Header")
        layout = QHBoxLayout(header)
        title = QLabel("SteelDefect Studio")
        title.setObjectName("AppTitle")
        subtitle = QLabel("钢材表面缺陷检测工作台 | 推理、评估、质检、训练结果浏览")
        subtitle.setObjectName("AppSubtitle")
        text = QVBoxLayout()
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addLayout(text)
        layout.addStretch(1)
        layout.addWidget(self.status_label)
        return header

    def _build_inference_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_preview_panel())
        splitter.addWidget(self._build_control_panel())
        splitter.setSizes([960, 420])
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.addWidget(splitter)
        return page

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.infer_log)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)
        form_group = QGroupBox("推理参数")
        form = QFormLayout(form_group)
        form.addRow("模型", _with_buttons(self.model_edit, [("选择权重", self._browse_model)]))
        form.addRow(
            "输入源",
            _with_buttons(
                self.source_edit,
                [
                    ("图片/视频", self._browse_media_source),
                    ("文件夹", self._browse_folder_source),
                    ("摄像头 0", self._use_camera),
                ],
            ),
        )
        form.addRow("运行配置", _with_buttons(self.runtime_edit, [("选择 YAML", self._browse_runtime)]))
        form.addRow("Pipeline", _with_buttons(self.pipeline_edit, [("选择 YAML", self._browse_pipeline)]))
        form.addRow("任务", self.task_combo)
        form.addRow("置信度", self.conf_edit)
        form.addRow("IoU", self.iou_edit)
        form.addRow("图像尺寸", self.imgsz_spin)
        form.addRow("最大检测数", self.max_det_spin)
        form.addRow("类别过滤", self.classes_edit)
        form.addRow("设备", self.device_edit)
        form.addRow("运行名", self.name_edit)
        form.addRow("最多帧数", self.max_frames_spin)
        form.addRow("", self.save_check)
        form.addRow("", self.threaded_check)

        controls = QGridLayout()
        controls.addWidget(self.start_button, 0, 0)
        controls.addWidget(self.pause_button, 0, 1)
        controls.addWidget(self.stop_button, 1, 0)
        controls.addWidget(self.open_output_button, 1, 1)

        root.addWidget(form_group)
        root.addLayout(controls)
        root.addStretch(1)
        return panel

    def _build_browser_tab(
        self,
        *,
        title: str,
        hint: str,
        list_widget: QListWidget,
        detail_widget: QTextEdit,
        refresh_callback,
        open_callback,
    ) -> QWidget:
        refresh_button = QPushButton("刷新")
        open_button = QPushButton("打开所在目录")
        refresh_button.clicked.connect(refresh_callback)
        open_button.clicked.connect(open_callback)

        header = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel(hint)
        hint_label.setObjectName("HintText")
        header.addWidget(title_label)
        header.addWidget(hint_label)

        buttons = QHBoxLayout()
        buttons.addWidget(refresh_button)
        buttons.addWidget(open_button)
        buttons.addStretch(1)

        left = QVBoxLayout()
        left.addLayout(header)
        left.addLayout(buttons)
        left.addWidget(list_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        list_container = QWidget()
        list_container.setLayout(left)
        splitter.addWidget(list_container)
        splitter.addWidget(detail_widget)
        splitter.setSizes([420, 820])

        page = QWidget()
        root = QHBoxLayout(page)
        root.addWidget(splitter)
        return page

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start_worker)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop_worker)
        self.open_output_button.clicked.connect(self._open_last_output_dir)
        self.eval_list.currentItemChanged.connect(lambda current, _previous: self._show_eval_item(current))
        self.validation_list.currentItemChanged.connect(lambda current, _previous: self._show_validation_item(current))
        self.training_list.currentItemChanged.connect(lambda current, _previous: self._show_training_item(current))

    @Slot()
    def _start_worker(self) -> None:
        if self._thread is not None:
            return
        try:
            conf = float(self.conf_edit.text().strip())
            iou = float(self.iou_edit.text().strip())
        except ValueError:
            self._set_status("失败", "置信度和 IoU 必须是数字")
            return

        self.infer_log.clear()
        self._append_log("启动推理任务")
        self._append_log(f"模型: {self.model_edit.text().strip()}")
        self._append_log(f"输入源: {self.source_edit.text().strip()}")

        self._thread = QThread(self)
        self._worker = InferWorker(
            model=self.model_edit.text().strip(),
            source=self.source_edit.text().strip(),
            conf=conf,
            iou=iou,
            device=self.device_edit.text().strip() or None,
            name=self.name_edit.text().strip() or "desktop-infer",
            task=self.task_combo.currentText(),
            imgsz=self.imgsz_spin.value(),
            max_det=self.max_det_spin.value(),
            classes=self.classes_edit.text().strip() or None,
            runtime_config=self.runtime_edit.text().strip() or None,
            pipeline_yaml=self.pipeline_edit.text().strip() or None,
            save_outputs=self.save_check.isChecked(),
            threaded=self.threaded_check.isChecked(),
            max_frames=self.max_frames_spin.value() or None,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._show_frame)
        self._worker.progress_ready.connect(self._show_progress)
        self._worker.completed.connect(self._finish_success)
        self._worker.failed.connect(self._finish_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("暂停")
        self.stop_button.setEnabled(True)
        self._set_status("运行中", "推理运行中")
        self._thread.start()

    @Slot()
    def _toggle_pause(self) -> None:
        if self._worker is None:
            return
        paused = self._worker.toggle_pause()
        if paused:
            self.pause_button.setText("继续")
            self._set_status("已暂停", "推理已暂停")
            self._append_log("推理已暂停")
        else:
            self.pause_button.setText("暂停")
            self._set_status("运行中", "推理运行中")
            self._append_log("推理已继续")

    @Slot()
    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._set_status("停止中", "正在停止")
            self._append_log("请求停止推理")

    @Slot(object, object)
    def _show_frame(self, _frame, annotated) -> None:
        pixmap = _to_pixmap(annotated)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @Slot(object)
    def _show_progress(self, event) -> None:
        total = event.total_frames if event.total_frames is not None else "未知"
        self._set_status("运行中", f"帧 {event.frame_index}/{total}，FPS {event.loop_fps:.2f}")
        self.summary_label.setText(
            f"帧数 {event.frame_index} | 检测 {event.detections_total} | FPS {event.loop_fps:.2f} | 输出目录 {self._last_output_dir or '-'}"
        )
        if event.frame_index % 30 == 0:
            self._append_log(f"进度: frame={event.frame_index}, fps={event.loop_fps:.2f}, detections={event.detections_total}")

    @Slot(object)
    def _finish_success(self, result) -> None:
        self._last_output_dir = Path(result.output_dir)
        self.open_output_button.setEnabled(True)
        stats = result.stats or {}
        self.summary_label.setText(
            " | ".join(
                [
                    f"帧数 {stats.get('frames', 0)}",
                    f"检测 {stats.get('detections', 0)}",
                    f"平均 FPS {stats.get('avg_fps', 0)}",
                    f"输出目录 {result.output_dir}",
                ]
            )
        )
        self._set_status("完成", f"输出目录 {result.output_dir}")
        self._append_log(f"推理完成: {json.dumps(stats, ensure_ascii=False)}")

    @Slot(str)
    def _finish_failed(self, message: str) -> None:
        self._set_status("失败", message)
        self._append_log(f"推理失败: {message}")

    @Slot()
    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.stop_button.setEnabled(False)
        self._refresh_all_result_tabs()

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(ROOT_DIR), "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.model_edit.setText(path)

    def _browse_media_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片或视频",
            str(ROOT_DIR),
            "Media files (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mkv *.mov *.flv *.wmv);;All files (*)",
        )
        if path:
            self.source_edit.setText(path)

    def _browse_folder_source(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择图片文件夹", str(ROOT_DIR))
        if directory:
            self.source_edit.setText(directory)

    def _browse_runtime(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择推理运行配置", str(ROOT_DIR), "YAML (*.yaml *.yml)")
        if path:
            self.runtime_edit.setText(path)

    def _browse_pipeline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择推理 pipeline 配置", str(ROOT_DIR), "YAML (*.yaml *.yml)")
        if path:
            self.pipeline_edit.setText(path)

    def _use_camera(self) -> None:
        self.source_edit.setText("0")

    def _open_last_output_dir(self) -> None:
        if self._last_output_dir is not None:
            _open_path(self._last_output_dir)

    def _refresh_all_result_tabs(self) -> None:
        self._refresh_evaluation_results()
        self._refresh_validation_reports()
        self._refresh_training_results()

    def _refresh_evaluation_results(self) -> None:
        paths = sorted((ROOT_DIR / "runs" / "evaluation").glob("**/odp_audit.json"), reverse=True)
        self._fill_list(self.eval_list, paths)
        if self.eval_list.count() == 0:
            self.eval_detail.setPlainText("暂无模型评估结果。运行 odp-val 后会在这里显示。")

    def _refresh_validation_reports(self) -> None:
        paths = sorted((ROOT_DIR / "runs" / "data_validation").glob("**/report.json"), reverse=True)
        if not paths:
            paths = sorted((ROOT_DIR / "runs" / "data_validation").glob("**/report.md"), reverse=True)
        self._fill_list(self.validation_list, paths)
        if self.validation_list.count() == 0:
            self.validation_detail.setPlainText("暂无数据质检报告。运行 odp-validate 后会在这里显示。")

    def _refresh_training_results(self) -> None:
        result_dirs = sorted({path.parent for path in (ROOT_DIR / "runs").glob("**/results.csv")}, reverse=True)
        self._fill_list(self.training_list, result_dirs)
        if self.training_list.count() == 0:
            self.training_detail.setPlainText("暂无训练结果。运行 odp-train 后会在这里显示。")

    @staticmethod
    def _fill_list(list_widget: QListWidget, paths: list[Path]) -> None:
        list_widget.clear()
        for path in paths:
            item_text = str(path.relative_to(ROOT_DIR)) if _is_relative_to(path, ROOT_DIR) else str(path)
            list_widget.addItem(item_text)
            list_widget.item(list_widget.count() - 1).setData(Qt.ItemDataRole.UserRole, str(path))
        if list_widget.count():
            list_widget.setCurrentRow(0)

    def _show_eval_item(self, item) -> None:
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        payload = _read_json(path)
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
        lines.extend(_format_mapping(metrics, preferred=("fitness", "map50", "map50_95", "precision", "recall")))
        lines.extend(["", "原始 JSON", "-" * 60, json.dumps(payload, ensure_ascii=False, indent=2)])
        self.eval_detail.setPlainText("\n".join(lines))

    def _show_validation_item(self, item) -> None:
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if path.suffix.lower() != ".json":
            self.validation_detail.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
            return
        payload = _read_json(path)
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
            *_format_mapping(counts),
            "",
            "数据集概览",
            "-" * 60,
            *_format_mapping(summary),
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
        self.validation_detail.setPlainText("\n".join(lines))

    def _show_training_item(self, item) -> None:
        if item is None:
            return
        run_dir = Path(item.data(Qt.ItemDataRole.UserRole))
        results_csv = run_dir / "results.csv"
        lines = ["训练结果摘要", "=" * 60, f"训练目录: {run_dir}", ""]
        if results_csv.exists():
            rows = _read_csv_rows(results_csv)
            lines.append(f"results.csv: {results_csv}")
            lines.append(f"epoch 数   : {len(rows)}")
            if rows:
                lines.extend(["", "最后一轮指标", "-" * 60])
                last = rows[-1]
                lines.extend(
                    _format_mapping(
                        last,
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
        self.training_detail.setPlainText("\n".join(lines))

    def _open_selected_eval(self) -> None:
        _open_selected_parent(self.eval_list)

    def _open_selected_validation(self) -> None:
        _open_selected_parent(self.validation_list)

    def _open_selected_training(self) -> None:
        item = self.training_list.currentItem()
        if item is not None:
            _open_path(Path(item.data(Qt.ItemDataRole.UserRole)))

    def _set_status(self, state: str, detail: str) -> None:
        self.status_label.setText(f"状态：{state} | {detail}")

    def _append_log(self, message: str) -> None:
        self.infer_log.append(message)


def _open_path(path: Path) -> None:
    target = path if path.is_dir() else path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def _open_selected_parent(list_widget: QListWidget) -> None:
    item = list_widget.currentItem()
    if item is not None:
        _open_path(Path(item.data(Qt.ItemDataRole.UserRole)))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - result viewer should keep the UI alive.
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except UnicodeDecodeError:
        with path.open("r", encoding="gbk", newline="") as handle:
            return list(csv.DictReader(handle))


def _format_mapping(data: Any, preferred: tuple[str, ...] = ()) -> list[str]:
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


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _with_buttons(line_edit: QLineEdit, buttons: list[tuple[str, object]]) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line_edit)
    for label, callback in buttons:
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
    return container


def _to_pixmap(image) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    q_image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(q_image)


def _default_model() -> Path:
    candidates = sorted((ROOT_DIR / "models" / "trained").glob("**/*best*.pt"))
    if candidates:
        return candidates[-1]
    return ROOT_DIR / "models" / "checkpoints" / "best.pt"


def _default_source() -> str:
    test_images = ROOT_DIR / "data" / "processed" / "steel-surface-defect" / "test" / "images"
    if test_images.exists():
        first = next(iter(sorted(test_images.glob("*"))), None)
        if first is not None:
            return str(first)
    return "0"


def _load_stylesheet() -> str:
    qss = ROOT_DIR / "apps" / "desktop" / "style.qss"
    if qss.exists():
        return qss.read_text(encoding="utf-8")
    return ""


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(_load_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
