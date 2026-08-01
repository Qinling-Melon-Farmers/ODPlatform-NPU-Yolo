import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from od_platform.common.constants import Task
from od_platform.model_catalog import (
    BUILTIN_MODELS,
    ModelInfo,
    get_model_info,
    list_families,
    list_models,
    recommend_model,
)
from od_platform.model_catalog import loader as model_loader


class TestModelCatalog(unittest.TestCase):
    ALL_YOLO_FAMILIES = ("yolov5", "yolov7", "yolov8", "yolov9", "yolov10", "yolo11", "yolo12")

    def test_builtin_catalog_covers_all_yolo_families(self) -> None:
        self.assertEqual(len(BUILTIN_MODELS), 34)
        for family in self.ALL_YOLO_FAMILIES:
            family_models = list_models(family=family)
            self.assertTrue(family_models, f"系列 {family} 缺失模型")

    def test_builtin_catalog_contains_classic_variants(self) -> None:
        for name in ("yolov8n.pt", "yolo11n.pt", "yolov5nu.pt", "yolov9t.pt", "yolov10n.pt", "yolo12n.pt"):
            self.assertIn(name, BUILTIN_MODELS)

    def test_get_model_info_known(self) -> None:
        info = get_model_info("yolo11n.pt")
        self.assertIsNotNone(info)
        if info is not None:
            self.assertEqual(info.family, "yolo11")
            self.assertEqual(info.task, Task.DETECT)
            self.assertEqual(info.backend, "ultralytics")
            self.assertIn("map50_95", info.metrics)
            self.assertIsNotNone(info.primary_metric)

    def test_get_model_info_unknown_returns_none(self) -> None:
        self.assertIsNone(get_model_info("does-not-exist.pt"))

    def test_list_models_filters_by_task_and_family(self) -> None:
        models = list_models(task=Task.DETECT, family="yolov8")
        self.assertEqual(len(models), 5)
        self.assertTrue(all(info.family == "yolov8" for info in models))

    def test_list_families(self) -> None:
        families = list_families(task=Task.DETECT)
        self.assertEqual(families, list(self.ALL_YOLO_FAMILIES))

    def test_recommend_fastest(self) -> None:
        models = recommend_model("最快")
        min_speed = min(info.speed_cpu_ms for info in models)
        self.assertEqual(models[0].speed_cpu_ms, min_speed)

    def test_recommend_most_accurate(self) -> None:
        models = recommend_model("最准")
        best_metric = max((info.primary_metric or 0.0) for info in models)
        self.assertAlmostEqual(models[0].primary_metric or 0.0, best_metric)

    def test_recommend_balanced(self) -> None:
        models = recommend_model("平衡")
        self.assertEqual(models[0].size_category, "medium")

    def test_recommend_lightweight(self) -> None:
        models = recommend_model("轻量")
        min_params = min(info.params_m for info in models)
        self.assertEqual(models[0].params_m, min_params)

    def test_recommend_english_keywords(self) -> None:
        models = recommend_model("fastest")
        min_speed = min(info.speed_cpu_ms for info in models)
        self.assertEqual(models[0].speed_cpu_ms, min_speed)
        models = recommend_model("most accurate")
        best_metric = max((info.primary_metric or 0.0) for info in models)
        self.assertAlmostEqual(models[0].primary_metric or 0.0, best_metric)

    def test_recommend_family_filter_from_keyword(self) -> None:
        models = recommend_model("yolov8 最准")
        self.assertEqual(models[0].name, "yolov8x.pt")
        self.assertTrue(all(info.family == "yolov8" for info in models))

    def test_recommend_respects_limit(self) -> None:
        models = recommend_model("最快", limit=3)
        self.assertEqual(len(models), 3)

    def test_extra_models_merge_with_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extra_yaml = Path(temp_dir) / "models.yaml"
            extra_yaml.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - name: resnet18",
                        "    family: resnet",
                        "    variant: '18'",
                        "    task: classify",
                        "    backend: torchvision",
                        "    size_category: small",
                        "    params_m: 11.7",
                        "    metrics:",
                        "      top1_acc: 69.8",
                        "    speed_cpu_ms: 5.0",
                        "    description: ResNet18 分类模型",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with patch("od_platform.model_catalog.loader._EXTRA_MODELS_PATH", extra_yaml), patch.object(
                model_loader, "_extra_models_cache", None
            ):
                info = get_model_info("resnet18")
                self.assertIsNotNone(info)
                if info is not None:
                    self.assertEqual(info.task, "classify")
                    self.assertEqual(info.backend, "torchvision")
                self.assertIsNotNone(get_model_info("yolo11n.pt"))
                # 内置 + 扩展合并后按任务过滤可单独列出扩展模型
                names = [m.name for m in list_models(task="classify")]
                self.assertEqual(names, ["resnet18"])

    def test_extra_models_missing_file_returns_empty(self) -> None:
        with patch(
            "od_platform.model_catalog.loader._EXTRA_MODELS_PATH", Path("no-such-dir") / "models.yaml"
        ), patch.object(model_loader, "_extra_models_cache", None):
            self.assertEqual(get_model_info("anything.pt"), None)
            self.assertEqual(len(list_models()), 34)

    def test_extra_models_invalid_yaml_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extra_yaml = Path(temp_dir) / "models.yaml"
            extra_yaml.write_text("models: [broken", encoding="utf-8")
            with patch("od_platform.model_catalog.loader._EXTRA_MODELS_PATH", extra_yaml), patch.object(
                model_loader, "_extra_models_cache", None
            ):
                self.assertEqual(len(list_models()), 34)

    def test_model_info_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        info = ModelInfo(
            name="x.pt",
            family="x",
            variant="x",
            task=Task.DETECT,
            backend="ultralytics",
            size_category="nano",
            params_m=1.0,
        )
        with self.assertRaises(FrozenInstanceError):
            info.name = "y.pt"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
