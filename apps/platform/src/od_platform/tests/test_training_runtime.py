import json
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from od_platform.cli.train_model import main as train_main
from od_platform.runtime_config.generator import write_train_template
from od_platform.runtime_config.loaders import load_train_config
from od_platform.runtime_config.train import YOLOTrainConfig
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

        self.assertEqual(kwargs["model"], "yolo11n.pt")
        self.assertTrue(kwargs["cos_lr"])
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

            class FakeYOLO:
                def __init__(self, _model: str) -> None:
                    pass

                def train(self, **kwargs) -> None:
                    run_dir = Path(kwargs["project"]) / kwargs["name"]
                    observed["precreated"] = run_dir.exists()
                    (run_dir / "weights").mkdir(parents=True)
                    (run_dir / "weights" / "best.pt").write_bytes(b"best")
                    (run_dir / "weights" / "last.pt").write_bytes(b"last")

            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO, __version__="test")
            with self._patch_runtime_paths(root), patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                result = run_training(YOLOTrainConfig(data="rsod", model="yolo11n.pt", epochs=1), dry_run=False)

            self.assertFalse(observed["precreated"])
            self.assertTrue(result.manifest_path.exists())
            self.assertEqual(set(result.archived_weights), {"best", "last"})

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


if __name__ == "__main__":
    unittest.main()
