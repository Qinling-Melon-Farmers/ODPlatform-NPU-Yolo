import json
import logging
import sys
import tempfile
import types
import unittest
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from od_platform.cli.model_train import main as model_train_main
from od_platform.cli.train_model import main as train_main
from od_platform.runtime_config.generator import write_train_template
from od_platform.runtime_config.loaders import load_train_config
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.training.metrics import (
    log_training_report,
    summarize_results_csv,
    write_metrics_summary,
)
from od_platform.training.plots import plot_training_results
from od_platform.training.service import (
    archive_model_weights,
    build_training_run_plan,
    run_training,
)


class TestTrainingRuntime(unittest.TestCase):
    def _patch_runtime_paths(self, root: Path):
        return patch.multiple(
            "od_platform.training.service.paths",
            RUNS_DIR=root / "runs",
            TRAINED_MODELS_DIR=root / "models" / "trained",
            LOGGING_DIR=root / "apps" / "platform" / "logging",
            DATASET_CONFIGS_DIR=root / "apps" / "platform" / "configs" / "datasets",
            RUNTIME_CONFIGS_DIR=root / "apps" / "platform" / "configs" / "runtime",
        )

    def test_train_config_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValueError):
            YOLOTrainConfig.model_validate({"data": "rsod", "unknown": True})

    def test_train_config_exports_ultralytics_kwargs_without_archive_fields(self) -> None:
        config = YOLOTrainConfig(data="rsod", model="yolo11n.pt", extra_args={"cos_lr": True})

        kwargs = config.to_ultralytics_kwargs()

        self.assertNotIn("model", kwargs)
        self.assertTrue(kwargs["cos_lr"])
        self.assertIsInstance(kwargs["batch"], int)
        self.assertNotIn("archive_weights", kwargs)
        self.assertNotIn("copy_archive", kwargs)
        self.assertNotIn("extra_args", kwargs)

    def test_build_training_run_plan_uses_next_train_number_and_named_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "runs" / "detect" / "train-1").mkdir(parents=True)
            (root / "models" / "trained" / "train-3-20260101-000000-yolo11n-best").mkdir(parents=True)
            with self._patch_runtime_paths(root):
                plan = build_training_run_plan(
                    YOLOTrainConfig(data="rsod", model="weights/yolo11n.pt"),
                    now=datetime(2026, 7, 3, 15, 30, 1),
                )

            self.assertEqual(plan.sequence, 4)
            self.assertEqual(plan.source_run_name, "train-4")
            self.assertEqual(plan.archive_run_name, "train-4-20260703-153001-yolo11n")
            self.assertEqual(plan.log_file.name, "train-4-20260703-153001-yolo11n.log")

    def test_build_training_run_plan_honors_custom_project_and_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self._patch_runtime_paths(root):
                plan = build_training_run_plan(
                    YOLOTrainConfig(
                        data="rsod",
                        model="yolo11n.pt",
                        project=str(root / "custom_runs"),
                        name="steel-baseline",
                    ),
                    now=datetime(2026, 7, 6, 16, 0, 0),
                )

            self.assertEqual(plan.source_run_name, "steel-baseline")
            self.assertEqual(plan.ultralytics_project, (root / "custom_runs").resolve())
            self.assertEqual(plan.source_run_dir, (root / "custom_runs").resolve() / "steel-baseline")

    def test_build_training_run_plan_avoids_existing_custom_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "custom_runs" / "steel-baseline").mkdir(parents=True)
            with self._patch_runtime_paths(root):
                plan = build_training_run_plan(
                    YOLOTrainConfig(
                        data="rsod",
                        model="yolo11n.pt",
                        project=str(root / "custom_runs"),
                        name="steel-baseline",
                    ),
                    now=datetime(2026, 7, 7, 13, 30, 0),
                )

            self.assertEqual(plan.source_run_name, "steel-baseline-2")
            self.assertEqual(plan.source_run_dir, (root / "custom_runs").resolve() / "steel-baseline-2")
            self.assertEqual(plan.archive_run_name, "steel-baseline-2-20260707-133000-yolo11n")

    def test_archive_model_weights_copies_best_and_last(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "runs" / "detect" / "train-1"
            weights = source / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")

            with self._patch_runtime_paths(root):
                archived = archive_model_weights(source, "train-1-20260703-153001-yolo11n", "yolo11n.pt")

            self.assertEqual(set(archived), {"best", "last"})
            self.assertEqual(archived["best"].read_bytes(), b"best")
            self.assertTrue((archived["best"].parent / "metadata.json").exists())
            self.assertEqual((weights / "best.pt").read_bytes(), b"best")

    def test_dry_run_writes_manifest_and_named_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_yaml = root / "apps" / "platform" / "configs" / "datasets" / "rsod.yaml"
            dataset_yaml.parent.mkdir(parents=True)
            dataset_yaml.write_text("path: dataset\ntrain: train/images\nval: val/images\nnc: 1\nnames: [ship]\n", encoding="utf-8")

            with self._patch_runtime_paths(root):
                result = run_training(
                    YOLOTrainConfig(data="rsod", model="yolo11n.pt", epochs=1),
                    executor="tester",
                    dry_run=True,
                )

            self.assertTrue(result.dry_run)
            self.assertTrue(result.manifest_path.exists())
            payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["executor"], "tester")
            self.assertEqual(payload["dataset"]["nc"], 1)
            self.assertTrue(result.plan.log_file.exists())

    def test_real_run_does_not_precreate_ultralytics_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_yaml = root / "apps" / "platform" / "configs" / "datasets" / "rsod.yaml"
            dataset_yaml.parent.mkdir(parents=True)
            dataset_yaml.write_text("path: dataset\ntrain: train/images\nval: val/images\nnc: 1\nnames: [ship]\n", encoding="utf-8")
            observed: dict[str, bool] = {}
            test_case = self

            class FakeYOLO:
                def __init__(self, _model: str) -> None:
                    pass

                def train(self, **kwargs) -> None:
                    test_case.assertNotIn("model", kwargs)
                    run_dir = Path(kwargs["project"]) / kwargs["name"]
                    observed["precreated"] = run_dir.exists()
                    (run_dir / "weights").mkdir(parents=True)
                    (run_dir / "weights" / "best.pt").write_bytes(b"best")
                    (run_dir / "weights" / "last.pt").write_bytes(b"last")
                    (run_dir / "results.csv").write_text(
                        "\n".join(
                            [
                                "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2",
                                "1,1.0,1.0,2.0,3.0,0.1,0.2,0.3,0.4,1.1,2.1,3.1,0.001,0.001,0.001",
                                "2,2.5,0.8,1.8,2.8,0.5,0.6,0.7,0.8,0.9,1.9,2.9,0.0005,0.0005,0.0005",
                            ]
                        ),
                        encoding="utf-8",
                    )

            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO, __version__="test")
            with self._patch_runtime_paths(root), patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                result = run_training(YOLOTrainConfig(data="rsod", model="yolo11n.pt", epochs=1), dry_run=False)

            self.assertFalse(observed["precreated"])
            self.assertTrue(result.manifest_path.exists())
            self.assertEqual(set(result.archived_weights), {"best", "last"})
            self.assertTrue((result.plan.source_run_dir / "training_metrics.json").exists())
            self.assertTrue((result.plan.source_run_dir / "training_results.png").exists())

    def test_real_run_uses_actual_ultralytics_save_dir_for_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_yaml = root / "apps" / "platform" / "configs" / "datasets" / "rsod.yaml"
            dataset_yaml.parent.mkdir(parents=True)
            dataset_yaml.write_text("path: dataset\ntrain: train/images\nval: val/images\nnc: 1\nnames: [ship]\n", encoding="utf-8")

            class FakeYOLO:
                def __init__(self, _model: str) -> None:
                    pass

                def train(self, **kwargs):
                    planned = Path(kwargs["project"]) / kwargs["name"]
                    actual = planned.with_name(f"{planned.name}-2")
                    (actual / "weights").mkdir(parents=True)
                    (actual / "weights" / "best.pt").write_bytes(b"best")
                    (actual / "weights" / "last.pt").write_bytes(b"last")
                    (actual / "results.csv").write_text(
                        "\n".join(
                            [
                                "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2",
                                "1,1.0,1.0,2.0,3.0,0.1,0.2,0.3,0.4,1.1,2.1,3.1,0.001,0.001,0.001",
                            ]
                        ),
                        encoding="utf-8",
                    )
                    return types.SimpleNamespace(save_dir=actual)

            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO, __version__="test")
            with self._patch_runtime_paths(root), patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                result = run_training(
                    YOLOTrainConfig(data="rsod", model="yolo11n.pt", epochs=1, name="steel-baseline"),
                    dry_run=False,
                )

            self.assertEqual(result.plan.source_run_name, "steel-baseline-2")
            self.assertEqual(set(result.archived_weights), {"best", "last"})
            self.assertTrue(result.archived_weights["best"].exists())

    def test_generator_loader_and_cli_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "apps" / "platform" / "configs" / "runtime" / "train.yaml"
            dataset_yaml = root / "apps" / "platform" / "configs" / "datasets" / "rsod.yaml"
            dataset_yaml.parent.mkdir(parents=True)
            dataset_yaml.write_text("path: dataset\ntrain: train/images\nval: val/images\nnc: 1\nnames: [ship]\n", encoding="utf-8")

            write_train_template(config_path, overwrite=True)
            self.assertEqual(load_train_config(config_path).data, "rsod")

            with self._patch_runtime_paths(root), patch(
                "od_platform.runtime_config.loaders.paths.RUNTIME_CONFIGS_DIR",
                root / "apps" / "platform" / "configs" / "runtime",
            ):
                code = train_main(["--config", str(config_path), "--dry-run", "--executor", "tester"])

            self.assertEqual(code, 0)

    def test_cli_overrides_yaml_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "train.yaml"
            config_path.write_text("model: yolo11n.pt\ndata: rsod\nepochs: 10\nbatch: 16\n", encoding="utf-8")
            captured = {}

            def fake_run_training(config, **kwargs):
                captured["config"] = config
                captured["kwargs"] = kwargs

            with patch("od_platform.cli.train_model.run_training", side_effect=fake_run_training):
                code = train_main(
                    [
                        "--config",
                        str(config_path),
                        "--epochs",
                        "2",
                        "--batch",
                        "4",
                        "--model",
                        "yolo11s.pt",
                        "--no-archive",
                        "--dry-run",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(captured["config"].epochs, 2)
            self.assertEqual(captured["config"].batch, 4)
            self.assertEqual(captured["config"].model, "yolo11s.pt")
            self.assertFalse(captured["config"].archive_weights)
            self.assertTrue(captured["kwargs"]["dry_run"])

    def test_model_train_compat_entry_points_to_train_cli(self) -> None:
        self.assertIs(model_train_main, train_main)

    def test_results_csv_summary_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "results.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2",
                        "1,1.0,1.0,2.0,3.0,0.1,0.2,0.3,0.4,1.1,2.1,3.1,0.001,0.001,0.001",
                        "2,2.5,0.8,1.8,2.8,0.5,0.6,0.7,0.8,0.9,1.9,2.9,0.0005,0.0005,0.0005",
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_results_csv(csv_path)
            summary_path = write_metrics_summary(csv_path, root / "summary.json")
            figure_path = plot_training_results(csv_path, root / "figure.png")

            self.assertEqual(summary["epochs"], 2)
            self.assertEqual(summary["last"]["map50_95"], 0.8)
            self.assertTrue(summary_path.exists())
            self.assertTrue(figure_path.exists())

    def test_training_report_logs_teaching_style_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "results.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2",
                        "1,1.0,1.0,2.0,3.0,0.1,0.2,0.3,0.4,1.1,2.1,3.1,0.001,0.001,0.001",
                    ]
                ),
                encoding="utf-8",
            )
            summary = summarize_results_csv(csv_path)
            stream = StringIO()
            test_logger = logging.getLogger("od_platform.tests.training_report")
            test_logger.handlers.clear()
            test_logger.setLevel(logging.INFO)
            test_logger.propagate = False
            handler = logging.StreamHandler(stream)
            test_logger.addHandler(handler)
            train_result = types.SimpleNamespace(
                task="detect",
                speed={"preprocess": 0.5, "inference": 2.0, "loss": 0.0, "postprocess": 1.0},
                results_dict={"metrics/precision(B)": 0.7, "metrics/recall(B)": 0.8, "fitness": 0.9},
                names={0: "steel"},
                maps=[0.65],
            )

            log_training_report(summary, run_dir=root, target_logger=test_logger, train_result=train_result)

            output = stream.getvalue()
            self.assertIn("训练结果 (detect)", output)
            self.assertIn("处理速度", output)
            self.assertIn("Precision", output)
            self.assertIn("steel", output)
            test_logger.handlers.clear()


if __name__ == "__main__":
    unittest.main()
