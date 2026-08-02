import sys
import unittest
from pathlib import Path

DESKTOP_DIR = Path(__file__).resolve().parents[4] / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

from task_builder import TASK_NAMES, build_task_command  # noqa: E402


class TestTaskBuilder(unittest.TestCase):
    def test_task_names_eleven(self) -> None:
        self.assertEqual(
            list(TASK_NAMES),
            [
                "导入数据集",
                "数据转换",
                "数据质检",
                "模型评估",
                "模型训练",
                "项目重置",
                "训练曲线生成",
                "列出模型",
                "数据标注",
                "自动标注",
                "AI 任务",
            ],
        )

    def test_import_dataset(self) -> None:
        module, args = build_task_command(
            "导入数据集",
            {"zip_path": "x.zip", "dataset": "demo", "format": "yolo", "overwrite": True, "extra_args": ["--x"]},
        )
        self.assertEqual(module, "od_platform.cli.import_dataset")
        self.assertEqual(args, ["x.zip", "--name", "demo", "--format", "yolo", "--overwrite", "--x"])

    def test_import_dataset_missing_zip(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_task_command("导入数据集", {})
        self.assertIn("zip", str(ctx.exception))

    def test_transform_voc_mapping(self) -> None:
        _module, args = build_task_command("数据转换", {"dataset": "d", "format": "voc", "task": "detect"})
        self.assertIn("--format", args)
        self.assertEqual(args[args.index("--format") + 1], "pascal_voc")

    def test_train_explicit_params(self) -> None:
        _module, args = build_task_command(
            "模型训练",
            {"model": "yolo11n.pt", "dataset": "d", "epochs": 100, "batch": 16, "workers": 4, "dry_run": True},
        )
        self.assertEqual(args[args.index("--epochs") + 1], "100")
        self.assertEqual(args[args.index("--batch") + 1], "16")
        self.assertEqual(args[args.index("--workers") + 1], "4")
        self.assertIn("--dry-run", args)

    def test_train_defaults(self) -> None:
        _module, args = build_task_command("模型训练", {"model": "m", "dataset": "d"})
        self.assertEqual(args[args.index("--epochs") + 1], "1")

    def test_train_missing_model(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_task_command("模型训练", {"dataset": "d"})
        self.assertIn("模型权重", str(ctx.exception))

    def test_reset_flags_dashed(self) -> None:
        _module, args = build_task_command("项目重置", {"dry_run": True, "backup": True})
        self.assertIn("--dry-run", args)
        self.assertIn("--backup", args)
        self.assertNotIn("--dry_run", args)

    def test_list_models_with_family_and_recommend(self) -> None:
        _module, args = build_task_command(
            "列出模型",
            {"family": "yolov8", "recommend": "最准", "limit": 5, "json": True},
        )
        self.assertEqual(args, ["--family", "yolov8", "--recommend", "最准", "--limit", "5", "--json"])

    def test_list_models_no_family(self) -> None:
        _module, args = build_task_command("列出模型", {"family": None, "limit": 10})
        self.assertNotIn("--family", args)

    def test_annotate_classes_and_edit(self) -> None:
        _module, args = build_task_command(
            "数据标注",
            {"dataset": "d", "classes": ["cat", "dog"], "resume": True, "edit": True},
        )
        self.assertEqual(args, ["--dataset", "d", "--classes", "cat", "dog", "--resume", "--edit"])

    def test_annotate_missing_classes(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_task_command("数据标注", {"dataset": "d"})
        self.assertIn("类别", str(ctx.exception))

    def test_auto_annotate(self) -> None:
        _module, args = build_task_command(
            "自动标注",
            {
                "dataset": "d",
                "classes": ["cat"],
                "prompt": "框出猫",
                "base_url": "https://x/v1",
                "model": "qwen",
                "api_key": "k",
                "limit": 5,
                "dry_run": True,
            },
        )
        self.assertEqual(
            args,
            [
                "--dataset",
                "d",
                "--classes",
                "cat",
                "--prompt",
                "框出猫",
                "--base-url",
                "https://x/v1",
                "--model",
                "qwen",
                "--api-key",
                "k",
                "--limit",
                "5",
                "--dry-run",
            ],
        )

    def test_ai_task(self) -> None:
        _module, args = build_task_command(
            "AI 任务",
            {"prompt": "训练 d", "base_url": "https://x/v1", "model": "deepseek", "api_key": "k"},
        )
        self.assertEqual(
            args,
            ["--base-url", "https://x/v1", "--model", "deepseek", "--task", "训练 d", "--api-key", "k"],
        )

    def test_unknown_task(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_task_command("未知任务", {})
        self.assertIn("未知任务", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
