"""TrainingResultView 纯函数与离屏冒烟。"""

import csv
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
from views.training_result_view import TrainingResultView, format_training_summary  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _make_run_dir(root: Path) -> Path:
    run_dir = root / "run-1"
    (run_dir / "weights").mkdir(parents=True)
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP50(B)"])
        writer.writeheader()
        writer.writerow({"epoch": "50", "metrics/mAP50(B)": "0.995"})
    (run_dir / "weights" / "best.pt").write_bytes(b"w")
    (run_dir / "train_batch0.png").write_bytes(b"png")
    return run_dir


class TestFormatTrainingSummary(unittest.TestCase):
    def test_summary_contains_metrics_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = _make_run_dir(Path(temp_dir))
            summary, plots = format_training_summary(run_dir)
            self.assertIn("epoch 数   : 1", summary)
            self.assertIn("mAP50", summary)
            self.assertIn("best.pt", summary)
            self.assertEqual(len(plots), 1)

    def test_summary_missing_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "empty"
            run_dir.mkdir()
            summary, plots = format_training_summary(run_dir)
            self.assertIn("未找到 results.csv", summary)
            self.assertEqual(plots, [])


class TestTrainingResultViewSmoke(unittest.TestCase):
    def test_refresh_populates_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_run_dir(root / "runs")
            from unittest.mock import patch

            from services import paths as service_paths

            with patch.object(service_paths, "ROOT_DIR", root):
                view = TrainingResultView()
                self.assertEqual(view.result_list.count(), 1)

    def test_select_shows_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_run_dir(root / "runs")
            from unittest.mock import patch

            from services import paths as service_paths

            with patch.object(service_paths, "ROOT_DIR", root):
                view = TrainingResultView()
                view.result_list.setCurrentRow(0)
                self.assertIn("训练结果摘要", view.detail.toPlainText())


if __name__ == "__main__":
    unittest.main()
