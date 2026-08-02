import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from od_platform.annotation.cli.auto_annotate import main as auto_annotate_main


class _Args(SimpleNamespace):
    """最小参数容器（供 _resolve_classes 测试）。"""


class TestAutoAnnotateCli(unittest.TestCase):
    def test_main_requires_dataset_and_classes(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            auto_annotate_main(["--base-url", "https://example.com/v1", "--model", "m"])
        self.assertEqual(ctx.exception.code, 2)

    def test_main_requires_base_url_and_model(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            auto_annotate_main(["--dataset", "demo", "--classes", "aircraft"])
        self.assertEqual(ctx.exception.code, 2)

    def test_main_base_url_from_environment(self) -> None:
        """--base-url 缺省时从 OPENAI_BASE_URL 兜底（Agent 调用路径）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            labels_dir = root / "annotations"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            (images_dir / "a.jpg").write_bytes(b"img")

            with patch.dict("os.environ", {"OPENAI_BASE_URL": "https://env.example.com/v1"}, clear=False):
                with patch("od_platform.annotation.cli.auto_annotate.run_vlm_annotation") as fake_run:
                    code = auto_annotate_main(
                        [
                            "--dataset",
                            "demo",
                            "--classes",
                            "aircraft",
                            "--model",
                            "qwen-vl-max",
                            "--images-dir",
                            str(images_dir),
                            "--labels-dir",
                            str(labels_dir),
                            "--dry-run",
                        ]
                    )
            self.assertEqual(code, 0)
            self.assertEqual(fake_run.call_args.kwargs["config"].base_url, None)  # client 层读环境变量

    def test_main_no_base_url_no_env_exits_tool_error(self) -> None:
        """--base-url 与环境变量都缺失时，dry-run 也走失败路径（client 构造抛 ValueError）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            labels_dir = root / "annotations"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            (images_dir / "a.jpg").write_bytes(b"img")

            with patch.dict("os.environ", {}, clear=True):
                with patch("od_platform.annotation.cli.auto_annotate.run_vlm_annotation") as fake_run:
                    fake_run.side_effect = ValueError("缺少 API 基地址")
                    code = auto_annotate_main(
                        [
                            "--dataset",
                            "demo",
                            "--classes",
                            "aircraft",
                            "--model",
                            "qwen-vl-max",
                            "--images-dir",
                            str(images_dir),
                            "--labels-dir",
                            str(labels_dir),
                            "--dry-run",
                        ]
                    )
            self.assertEqual(code, 2)

    def test_main_dry_run_exits_zero_no_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            labels_dir = root / "annotations"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            (images_dir / "a.jpg").write_bytes(b"img")

            with patch("od_platform.annotation.cli.auto_annotate.run_vlm_annotation") as fake_run:
                code = auto_annotate_main(
                    [
                        "--dataset",
                        "demo",
                        "--classes",
                        "aircraft",
                        "--base-url",
                        "https://example.com/v1",
                        "--model",
                        "qwen-vl-max",
                        "--images-dir",
                        str(images_dir),
                        "--labels-dir",
                        str(labels_dir),
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            fake_run.assert_called_once()
            kwargs = fake_run.call_args.kwargs
            self.assertTrue(kwargs["dry_run"])
            self.assertEqual(kwargs["classes"], ["aircraft"])

    def test_classes_from_yaml(self) -> None:
        """--classes-from-yaml 从 dataset yaml 的 names 读取类别。"""
        from od_platform.annotation.cli.auto_annotate import _resolve_classes
        from od_platform.common import paths as common_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            configs_dir = Path(temp_dir) / "configs" / "datasets"
            configs_dir.mkdir(parents=True)
            yaml_path = configs_dir / "liftrace.yaml"
            yaml_path.write_text(
                "names:\n  0: bridge\n  1: panzer\n  2: pillbox\n  3: tent\n  4: tank\n",
                encoding="utf-8",
            )
            with patch.object(common_paths, "DATASET_CONFIGS_DIR", configs_dir):
                classes = _resolve_classes(_Args(classes=[], classes_from_yaml=True, dataset="liftrace"))
            self.assertEqual(classes, ["bridge", "panzer", "pillbox", "tent", "tank"])

    def test_classes_from_yaml_missing_config(self) -> None:
        from od_platform.annotation.cli.auto_annotate import _resolve_classes

        with tempfile.TemporaryDirectory() as temp_dir:
            from od_platform.common import paths as common_paths

            with patch.object(common_paths, "DATASET_CONFIGS_DIR", Path(temp_dir) / "none"):
                with self.assertRaises(FileNotFoundError):
                    _resolve_classes(_Args(classes=[], classes_from_yaml=True, dataset="liftrace"))

    def test_main_runs_with_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            labels_dir = root / "annotations"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            (images_dir / "a.jpg").write_bytes(b"img")

            with patch("od_platform.annotation.cli.auto_annotate.run_vlm_annotation") as fake_run:
                fake_run.return_value.annotated_new = 1
                fake_run.return_value.box_count = 3
                fake_run.return_value.failed = []
                fake_run.return_value.total_images = 1
                code = auto_annotate_main(
                    [
                        "--dataset",
                        "demo",
                        "--classes",
                        "aircraft",
                        "ship",
                        "--prompt",
                        "框出所有飞机和船",
                        "--base-url",
                        "https://example.com/v1",
                        "--model",
                        "glm-4.5v-turbo",
                        "--images-dir",
                        str(images_dir),
                        "--labels-dir",
                        str(labels_dir),
                        "--limit",
                        "5",
                    ]
                )
            self.assertEqual(code, 0)
            kwargs = fake_run.call_args.kwargs
            self.assertEqual(kwargs["config"].prompt, "框出所有飞机和船")
            self.assertEqual(kwargs["limit"], 5)


if __name__ == "__main__":
    unittest.main()
