"""ResultBrowserView 纯函数与离屏冒烟（评估/质检）。"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DESKTOP_DIR = Path(__file__).resolve().parents[4] / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402
from views.result_browser_view import (  # noqa: E402
    EvaluationResultView,
    ValidationResultView,
    format_eval_summary,
    format_validation_summary,
)

_app = QApplication.instance() or QApplication([])


class TestRenderPureFunctions(unittest.TestCase):
    def test_format_eval_summary(self) -> None:
        payload = {
            "run_name": "val-1",
            "model_ref": "yolo11n.pt",
            "metrics": {"map50": 0.99, "map50_95": 0.98},
        }
        text = format_eval_summary(payload)
        self.assertIn("模型评估摘要", text)
        self.assertIn("yolo11n.pt", text)
        self.assertIn("0.99", text)

    def test_format_validation_summary(self) -> None:
        payload = {
            "run_id": "20260802_1",
            "overall_severity": "ERROR",
            "counts": {"error": 1},
            "results": [{"severity": "ERROR", "name": "label_format", "summary": "格式错误"}],
            "fix_items": ["修复 A"],
        }
        text = format_validation_summary(payload)
        self.assertIn("数据质检摘要", text)
        self.assertIn("ERROR", text)
        self.assertIn("label_format", text)
        self.assertIn("修复 A", text)


class TestResultBrowserSmoke(unittest.TestCase):
    def _make_runs(self, root: Path) -> None:
        eval_dir = root / "runs" / "evaluation" / "run-1"
        eval_dir.mkdir(parents=True)
        (eval_dir / "odp_audit.json").write_text(
            json.dumps({"run_name": "e1", "metrics": {"map50": 0.9}}), encoding="utf-8"
        )
        val_dir = root / "runs" / "data_validation" / "run-1"
        val_dir.mkdir(parents=True)
        (val_dir / "report.json").write_text(
            json.dumps({"run_id": "v1", "overall_severity": "PASS"}), encoding="utf-8"
        )

    def test_evaluation_view_scans_and_renders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_runs(root)
            view = EvaluationResultView(runs_root=root)
            view.refresh()  # 视图构造不自动刷新
            self.assertEqual(view.list_widget.count(), 1)
            view.list_widget.setCurrentRow(0)
            self.assertIn("模型评估摘要", view.detail.toPlainText())

    def test_validation_view_scans_and_renders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_runs(root)
            view = ValidationResultView(runs_root=root)
            view.refresh()
            self.assertEqual(view.list_widget.count(), 1)
            view.list_widget.setCurrentRow(0)
            self.assertIn("数据质检摘要", view.detail.toPlainText())

    def test_filter_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_runs(root)
            view = EvaluationResultView(runs_root=root)
            view.refresh()
            view.filter_edit.setText("nomatch")
            self.assertEqual(view.list_widget.count(), 0)


if __name__ == "__main__":
    unittest.main()
