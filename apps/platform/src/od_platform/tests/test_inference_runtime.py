import json
import logging
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from od_platform.cli.infer_model import main as infer_main
from od_platform.frame_source import SourceType
from od_platform.inference import PauseToken
from od_platform.inference.pipeline_config import load_pipeline_config
from od_platform.inference.service import build_inference_run_plan, run_inference
from od_platform.inference.sinks import OutputSink
from od_platform.runtime_config.generator import default_infer_config, write_infer_template
from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.loaders import load_infer_config


class TestInferenceRuntime(unittest.TestCase):
    def _close_project_logger(self) -> None:
        logger = logging.getLogger("od_platform")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

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

    def test_pause_token_toggle_and_resume(self) -> None:
        token = PauseToken()

        self.assertFalse(token.is_paused())
        self.assertTrue(token.toggle())
        self.assertTrue(token.is_paused())
        self.assertFalse(token.toggle())
        self.assertFalse(token.is_paused())
        self.assertTrue(token.wait_while_paused())

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

    def test_pipeline_config_parses_visualization_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "infer_pipeline.yaml"
            path.write_text(
                """
frame_source:
  camera:
    width: 640
    height: 480
    backend: msmf
visualization:
  enabled: true
  use_label_mapping: true
  label_mapping:
    scratch: 划痕
  color_mapping:
    scratch: [0, 0, 255]
  default_color: [0, 255, 0]
  style:
    box_thickness: 3
    text_color: [255, 255, 255]
""",
                encoding="utf-8",
            )

            config = load_pipeline_config(path)

        self.assertEqual(config.build_camera_config().width, 640)
        self.assertEqual(config.label_mapping["scratch"], "划痕")
        self.assertEqual(config.color_mapping["scratch"], (0, 0, 255))
        self.assertEqual(config.normalized_style_overrides()["line_width"], 3)
        self.assertEqual(config.normalized_style_overrides()["text_color"], (255, 255, 255))

    def test_infer_cli_pipeline_writes_audit_with_fake_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "best.pt"
            source_path = root / "image.jpg"
            model_path.write_bytes(b"model")
            source_path.write_bytes(b"image")

            class FakeBoxes:
                data = np.array([[1, 2, 30, 40, 0.91, 0]], dtype=float)

            class FakeResult:
                boxes = FakeBoxes()
                names = {0: "scratch"}
                speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}

                def plot(self):
                    return np.zeros((20, 30, 3), dtype=np.uint8)

            class FakeYOLO:
                names = {0: "scratch"}

                def __init__(self, model: str) -> None:
                    self.model = model

                def __call__(self, image, **kwargs):
                    return [FakeResult()]

            class FakeSource:
                def __init__(self, source, camera_config=None, stride=1):
                    self.source = source
                    self._done = False

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, exc_tb):
                    return False

                def __iter__(self):
                    return self

                def __next__(self):
                    if self._done:
                        raise StopIteration
                    self._done = True
                    from od_platform.frame_source import Frame, FrameInfo

                    return Frame(
                        image=np.zeros((20, 30, 3), dtype=np.uint8),
                        info=FrameInfo(
                            width=30,
                            height=20,
                            source_type=SourceType.IMAGE,
                            source_path=str(source_path),
                            filename="image.jpg",
                        ),
                    )

                def get_source_type(self):
                    return SourceType.IMAGE

            class CollectingSink(OutputSink):
                def __init__(self):
                    self.frames = 0

                def open(self, output_dir: Path, source_type: SourceType) -> None:
                    self.output_dir = output_dir

                def write(self, frame, annotated: np.ndarray) -> None:
                    self.frames += 1

                def close(self) -> None:
                    return None

            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO, __version__="test")
            with (
                self._patch_inference_paths(root),
                patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
                patch("od_platform.inference.pipeline.create_frame_source", FakeSource),
            ):
                try:
                    code = infer_main(
                        [
                            "--model",
                            str(model_path),
                            "--source",
                            str(source_path),
                            "--name",
                            "demo",
                            "--max-frames",
                            "1",
                            "--no-save",
                        ]
                    )
                finally:
                    self._close_project_logger()

            self.assertEqual(code, 0)
            audit = root / "runs" / "inference" / "detect" / "demo" / "odp_audit.json"
            self.assertTrue(audit.exists())
            payload = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(payload["stats"]["frames"], 1)
            self.assertEqual(payload["stats"]["detections"], 1)

    def test_infer_cli_threaded_video_uses_staged_bounded_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "best.pt"
            source_path = root / "demo.mp4"
            model_path.write_bytes(b"model")
            source_path.write_bytes(b"video")

            class FakeBoxes:
                data = np.array([[1, 2, 30, 40, 0.91, 0]], dtype=float)

            class FakeResult:
                boxes = FakeBoxes()
                names = {0: "scratch"}
                speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}

                def plot(self):
                    return np.zeros((20, 30, 3), dtype=np.uint8)

            class FakeYOLO:
                names = {0: "scratch"}

                def __init__(self, model: str) -> None:
                    self.model = model

                def __call__(self, image, **kwargs):
                    return [FakeResult()]

            class FakeFrameSource:
                def __init__(
                    self,
                    source,
                    camera_config=None,
                    *,
                    stride=1,
                    **_options,
                ):
                    self.source = source
                    self.stride = stride
                    self._done = False

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, exc_tb):
                    return False

                def __iter__(self):
                    return self

                def __next__(self):
                    if self._done:
                        raise StopIteration
                    self._done = True
                    from od_platform.frame_source import Frame, FrameInfo

                    return Frame(
                        image=np.zeros((20, 30, 3), dtype=np.uint8),
                        info=FrameInfo(
                            width=30,
                            height=20,
                            source_type=SourceType.VIDEO,
                            source_path=str(source_path),
                            filename="demo.mp4",
                            fps=30.0,
                            total_frames=1,
                        ),
                    )

                def get_source_type(self):
                    return SourceType.VIDEO

            fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO, __version__="test")
            with (
                self._patch_inference_paths(root),
                patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
                patch("od_platform.inference.pipeline.create_frame_source", FakeFrameSource),
            ):
                try:
                    code = infer_main(
                        [
                            "--model",
                            str(model_path),
                            "--source",
                            str(source_path),
                            "--name",
                            "threaded-video",
                            "--max-frames",
                            "1",
                            "--no-save",
                            "--threaded",
                        ]
                    )
                finally:
                    self._close_project_logger()

            self.assertEqual(code, 0)
            audit = root / "runs" / "inference" / "detect" / "threaded-video" / "odp_audit.json"
            payload = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(payload["stats"]["source_mode"], "threaded")
            self.assertEqual(payload["stats"]["source_buffer"], "bounded")
            self.assertEqual(payload["stats"]["pipeline_stages"], ["read", "infer", "render", "output"])


if __name__ == "__main__":
    unittest.main()
