"""Minimal PySide6 desktop inference demo for ODPlatform."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PLATFORM_SRC = ROOT_DIR / "apps" / "platform" / "src"
if str(PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(PLATFORM_SRC))

try:
    import cv2
    from PySide6.QtCore import Qt, QThread, Slot
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSpinBox,
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
    """Small operator UI for image, folder, video and camera inference."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ODPlatform 推理演示")
        self.resize(1180, 760)
        self._thread: QThread | None = None
        self._worker: InferWorker | None = None

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
        self.threaded_check = QCheckBox("后台读帧")
        self.threaded_check.setChecked(True)

        self.start_button = QPushButton("启动")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)

        self.image_label = QLabel("等待启动推理")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(860, 560)
        self.image_label.setStyleSheet("background: #111827; color: #d1d5db; border: 1px solid #374151;")
        self.status_label = QLabel("状态：就绪")

        self._build_layout()
        self._connect_signals()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name.
        self._stop_worker()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        form = QFormLayout()
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

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)

        side = QVBoxLayout()
        side.addLayout(form)
        side.addLayout(controls)
        side.addWidget(self.status_label)
        side.addStretch(1)

        root = QHBoxLayout()
        root.addWidget(self.image_label, stretch=1)
        root.addLayout(side)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start_worker)
        self.stop_button.clicked.connect(self._stop_worker)

    @Slot()
    def _start_worker(self) -> None:
        if self._thread is not None:
            return
        try:
            conf = float(self.conf_edit.text().strip())
            iou = float(self.iou_edit.text().strip())
        except ValueError:
            self.status_label.setText("状态：置信度和 IoU 必须是数字")
            return

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
        self.stop_button.setEnabled(True)
        self.status_label.setText("状态：推理运行中")
        self._thread.start()

    @Slot()
    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("状态：正在停止")

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
        self.status_label.setText(
            f"状态：帧 {event.frame_index}/{total}，FPS {event.loop_fps:.2f}，累计检测 {event.detections_total}"
        )

    @Slot(object)
    def _finish_success(self, result) -> None:
        self.status_label.setText(f"状态：完成，输出目录 {result.output_dir}")

    @Slot(str)
    def _finish_failed(self, message: str) -> None:
        self.status_label.setText(f"状态：失败，{message}")

    @Slot()
    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

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


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
