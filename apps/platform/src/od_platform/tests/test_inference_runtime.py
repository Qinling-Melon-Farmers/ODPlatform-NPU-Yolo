import json
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from od_platform.cli.infer_model import main as infer_main
from od_platform.inference.service import build_inference_run_plan, run_inference
from od_platform.runtime_config.generator import default_infer_config, write_infer_template
from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.loaders import load_infer_config


class TestInferenceRuntime(unittest.TestCase):
    def _patch_inference_paths(self, root: Path):
        return patch.multiple(
            "od_platform.inference.service.paths",
            ROOT_DIR=root,
            INFERENCE_RUNS_DIR=root / "runs" / "inference",
            LOGGING_DIR=root / "apps" / "platform" / "logging",
            CHECKPOINTS_DIR=root / "models" / "checkpoints",
        )

    def test_infer_config_rejects_conf_without_txt(self) -> None:
        with self.assertRaises(ValueError):
            YOLOInferConfig(model="best.pt", source="image.jpg", save_conf=True)

    def test_infer_config_exports_predict_kwargs(self) -> None:
        config = YOLOInferConfig(model="best.pt", source="image.jpg", save_txt=True, save_conf=True)

        kwargs = config.to_ultralytics_kwargs()

        self.assertNotIn("model", kwargs)
        self.assertNotIn("data", kwargs)
        self.assertNotIn("task", kwargs)
        self.assertEqual(kwargs["source"], "image.jpg")
        self.assertTrue(kwargs["save_txt"])

    def test_build_inference_run_plan_uses_predict_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "runs" / "inference" / "detect" / "predict-2").mkdir(parents=True)
            with self._patch_inference_paths(root):
                plan = build_inference_run_plan(
                    YOLOInferConfig(model="models/best.pt", source="images"),
                    now=datetime(2026, 7, 7, 9, 0, 1),
                )

            self.assertEqual(plan.sequence, 3)
            self.assertEqual(plan.source_run_name, "predict-3")
            self.assertEqual(plan.audit_run_name, "predict-3-20260707-090001-best")

    def test_generator_loader_and_cli_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "infer.yaml"
            model_path = root / "best.pt"
            source_path = root / "image.jpg"
            model_path.write_bytes(b"model")
            source_path.write_bytes(b"image")
            write_infer_template(config_path, overwrite=True)
            config_path.write_text(
                f"model: {model_path}\nsource: {source_path}\nsave_txt: true\nsave_conf: true\n",
                encoding="utf-8",
            )

            self.assertIn("source", default_infer_config())
            self.assertEqual(load_infer_config(config_path).source, str(source_path))

            with self._patch_inference_paths(root):
                code = infer_main(["--config", str(config_path), "--dry-run", "--executor", "tester"])

            self.assertEqual(code, 0)
            manifest = next((root / "runs" / "inference" / "detect").glob("predict-*/inference_manifest.json"))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["executor"], "tester")

    def test_run_inference_writes_summary_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "best.pt"
            source_path = root / "image.jpg"
            model_path.write_bytes(b"model")
            source_path.write_bytes(b"image")

            class FakeBoxes:
                def __len__(self) -> int:
                    return 2

            class FakeResult:
                path = str(source_path)
                boxes = FakeBoxes()
                speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}

            class FakeYOLO:
                def __init__(self, model: str) -> None:
                    self.model = model

                def predict(self, **kwargs):
                    run_dir = Path(kwargs["project"]) / kwargs["name"]
                    run_dir.mkdir(parents=True)
                    (run_dir / "image.jpg").write_bytes(b"rendered")
                    return [FakeResult()]

            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO, __version__="test")
            with self._patch_inference_paths(root), patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                result = run_inference(
                    YOLOInferConfig(model=str(model_path), source=str(source_path), name="demo"),
                    dry_run=False,
                )

            self.assertFalse(result.dry_run)
            self.assertTrue(result.manifest_path.exists())
            self.assertIsNotNone(result.summary_path)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["detections"], 2)


if __name__ == "__main__":
    unittest.main()
