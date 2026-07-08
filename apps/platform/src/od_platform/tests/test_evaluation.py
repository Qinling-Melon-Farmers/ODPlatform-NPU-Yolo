import json
import tempfile
import types
import unittest
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from od_platform.cli.evaluate_model import main as evaluate_main
from od_platform.evaluation import ValMetrics, ValService, evaluate_yolo
from od_platform.evaluation.service import build_evaluation_run_plan
from od_platform.runtime_config.val import YOLOValConfig


class TestEvaluationRuntime(unittest.TestCase):
    def _patch_runtime_paths(self, root: Path):
        return patch.multiple(
            "od_platform.evaluation.service.paths",
            ROOT_DIR=root,
            RUNS_DIR=root / "runs",
            LOGGING_DIR=root / "apps" / "platform" / "logging",
            DATASET_CONFIGS_DIR=root / "apps" / "platform" / "configs" / "datasets",
            RUNTIME_CONFIGS_DIR=root / "apps" / "platform" / "configs" / "runtime",
            TRAINED_MODELS_DIR=root / "models" / "trained",
            CHECKPOINTS_DIR=root / "models" / "checkpoints",
        )

    def _write_dataset_yaml(self, root: Path) -> Path:
        path = root / "apps" / "platform" / "configs" / "datasets" / "steel-surface-defect.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "path: dataset\ntrain: train/images\nval: val/images\nnc: 1\nnames: [defect]\n",
            encoding="utf-8",
        )
        return path

    def _write_val_config(self, root: Path, model: str = "best.pt") -> Path:
        path = root / "apps" / "platform" / "configs" / "runtime" / "val.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(
                [
                    f"model: {model}",
                    "data: steel-surface-defect",
                    "task: detect",
                    "batch: 4",
                    "imgsz: 640",
                    "workers: 0",
                    "device: cpu",
                    "plots: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def _fake_results(self, save_dir: Path):
        save_dir.mkdir(parents=True)
        return types.SimpleNamespace(
            save_dir=save_dir,
            fitness=0.7,
            box=types.SimpleNamespace(map50=0.88, map=0.66, mp=0.85, mr=0.79),
            names={0: "defect"},
            maps=[0.66],
            speed={"preprocess": 0.1, "inference": 2.0},
        )

    def test_build_evaluation_run_plan_uses_val_sequence_and_log_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "runs" / "evaluation" / "detect" / "val-1").mkdir(parents=True)
            with self._patch_runtime_paths(root):
                plan = build_evaluation_run_plan(
                    YOLOValConfig(model="models/trained/steel-best/best.pt", data="steel-surface-defect"),
                    now=datetime(2026, 7, 8, 18, 0, 0),
                )

            self.assertEqual(plan.sequence, 2)
            self.assertEqual(plan.source_run_name, "val-2")
            self.assertEqual(plan.audit_run_name, "val-2-20260708-180000-best")
            self.assertEqual(plan.log_file.name, "val-2-20260708-180000-best.log")

    def test_evaluate_success_writes_summary_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset_yaml(root)
            config_path = self._write_val_config(root)
            weight = root / "models" / "trained" / "steel-best" / "best.pt"
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"pt")
            fake_result = self._fake_results(root / "runs" / "evaluation" / "detect" / "val-1")

            with self._patch_runtime_paths(root), patch.object(ValService, "_run_eval", return_value=fake_result):
                result = ValService().evaluate(config_path, model=str(weight), executor="tester")

            self.assertTrue(result.success)
            self.assertIsInstance(result.metrics, ValMetrics)
            assert result.metrics is not None
            self.assertEqual(result.metrics.map50, 0.88)
            self.assertTrue(result.summary_path and result.summary_path.exists())
            self.assertTrue(result.audit_path and result.audit_path.exists())
            audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["kind"], "val")
            self.assertEqual(audit["executor"], "tester")

    def test_evaluate_missing_weight_fails_without_running_ultralytics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset_yaml(root)
            config_path = self._write_val_config(root, model="missing.pt")

            with self._patch_runtime_paths(root), patch.object(ValService, "_run_eval") as run:
                result = ValService().evaluate(config_path)

            self.assertFalse(result.success)
            self.assertIn("trained model does not exist", result.error or "")
            run.assert_not_called()

    def test_evaluate_catches_ultralytics_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset_yaml(root)
            config_path = self._write_val_config(root)
            weight = root / "models" / "checkpoints" / "best.pt"
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"pt")

            with self._patch_runtime_paths(root), patch.object(ValService, "_run_eval", side_effect=RuntimeError("boom")):
                result = ValService().evaluate(config_path, model="best.pt")

            self.assertFalse(result.success)
            self.assertIn("boom", result.error or "")

    def test_evaluate_yolo_convenience_calls_service(self) -> None:
        with patch.object(ValService, "evaluate", return_value="ok") as evaluate:
            result = evaluate_yolo("val", "best.pt", "steel")

        self.assertEqual(result, "ok")
        evaluate.assert_called_once()

    def test_cli_returns_nonzero_on_failed_evaluation(self) -> None:
        failed = types.SimpleNamespace(success=False, error="missing")
        with patch("od_platform.cli.evaluate_model.get_logger"), patch(
            "od_platform.cli.evaluate_model.evaluate_yolo",
            return_value=failed,
        ):
            with self.assertRaises(SystemExit) as raised:
                evaluate_main(["--config", "val", "--model", "best.pt", "--data", "steel-surface-defect"])

        self.assertEqual(raised.exception.code, 1)

    def test_cli_passes_overrides(self) -> None:
        success = types.SimpleNamespace(success=True)
        captured = {}

        def fake_evaluate(**kwargs):
            captured.update(kwargs)
            return success

        with patch("od_platform.cli.evaluate_model.get_logger"), patch(
            "od_platform.cli.evaluate_model.evaluate_yolo",
            side_effect=fake_evaluate,
        ):
            code = evaluate_main(
                [
                    "--config",
                    "val",
                    "--model",
                    "best.pt",
                    "--data",
                    "steel-surface-defect",
                    "--device",
                    "0",
                    "--batch",
                    "8",
                    "--no-plots",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(captured["model"], "best.pt")
        self.assertEqual(captured["data"], "steel-surface-defect")
        overrides = captured["cli_overrides"]
        self.assertEqual(overrides["device"], "0")
        self.assertEqual(overrides["batch"], 8)
        self.assertFalse(overrides["plots"])

    def test_cli_namespace_overrides_model_and_data(self) -> None:
        namespace = Namespace(model="best.pt", data="steel-surface-defect", batch=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_dataset_yaml(root)
            config_path = self._write_val_config(root, model="ignored.pt")
            weight = root / "models" / "checkpoints" / "best.pt"
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"pt")
            fake_result = self._fake_results(root / "runs" / "evaluation" / "detect" / "val-1")

            with self._patch_runtime_paths(root), patch.object(ValService, "_run_eval", return_value=fake_result):
                result = ValService().evaluate(config_path, cli_overrides=namespace)

            self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
