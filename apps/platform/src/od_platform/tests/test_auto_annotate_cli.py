import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from od_platform.annotation.cli.auto_annotate import main as auto_annotate_main


class TestAutoAnnotateCli(unittest.TestCase):
    def test_main_requires_dataset_and_classes(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            auto_annotate_main(["--base-url", "https://example.com/v1", "--model", "m"])
        self.assertEqual(ctx.exception.code, 2)

    def test_main_requires_base_url_and_model(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            auto_annotate_main(["--dataset", "demo", "--classes", "aircraft"])
        self.assertEqual(ctx.exception.code, 2)

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
