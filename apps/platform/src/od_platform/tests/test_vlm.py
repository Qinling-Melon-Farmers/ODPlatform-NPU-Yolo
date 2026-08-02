import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from od_platform.agent.client import APIError
from od_platform.annotation.vlm import (
    VLMConfig,
    VLMError,
    annotate_image_with_vlm,
    build_annotation_prompt,
    extract_json_object,
    parse_vlm_boxes,
    run_vlm_annotation,
)
from od_platform.annotation.writer import read_yolo_label


class FakeVLMClient:
    """按预设文本序列响应的假 VLM 客户端。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def chat(self, messages, *, model, temperature=0.0, **kwargs) -> dict:
        self.calls += 1
        content = self.responses.pop(0)
        if isinstance(content, Exception):
            raise content
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class TestPrompt(unittest.TestCase):
    def test_build_annotation_prompt_includes_classes_and_ids(self) -> None:
        prompt = build_annotation_prompt(classes=["aircraft", "ship"], user_prompt="框出所有飞机")
        self.assertIn("0: aircraft", prompt)
        self.assertIn("1: ship", prompt)
        self.assertIn("框出所有飞机", prompt)
        self.assertIn("x_center", prompt)
        self.assertIn("boxes 为空数组", prompt)


class TestJsonExtract(unittest.TestCase):
    def test_extract_plain_json(self) -> None:
        payload = extract_json_object('{"boxes": []}')
        self.assertEqual(payload, {"boxes": []})

    def test_extract_fenced_json(self) -> None:
        text = '```json\n{"boxes": [{"class_id": 0}]}\n```'
        payload = extract_json_object(text)
        self.assertEqual(payload, {"boxes": [{"class_id": 0}]})

    def test_extract_with_surrounding_text(self) -> None:
        text = '好的，以下是结果：\n{"boxes": [{"class_id": 1}]}\n仅供参考'
        payload = extract_json_object(text)
        self.assertEqual(payload, {"boxes": [{"class_id": 1}]})

    def test_extract_invalid_returns_none(self) -> None:
        self.assertIsNone(extract_json_object("没有 JSON"))
        self.assertIsNone(extract_json_object("{broken"))


class TestParseBoxes(unittest.TestCase):
    def test_parse_valid_boxes(self) -> None:
        payload = {
            "boxes": [
                {"class_id": 0, "x_center": 0.5, "y_center": 0.4, "width": 0.2, "height": 0.3},
                {"class_id": 1, "x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1},
            ]
        }
        boxes, discarded = parse_vlm_boxes(payload, classes=["aircraft", "ship"])
        self.assertEqual(len(boxes), 2)
        self.assertEqual(discarded, 0)
        self.assertEqual(boxes[0].class_id, 0)
        self.assertAlmostEqual(boxes[0].x_center, 0.5)

    def test_parse_filters_bad_class_and_geometry(self) -> None:
        payload = {
            "boxes": [
                {"class_id": 5, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2},  # 越界类别
                {"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.0, "height": 0.2},  # 零宽
                {"class_id": "x", "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2},  # 非法类型
                {"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2},  # 合法
            ]
        }
        boxes, discarded = parse_vlm_boxes(payload, classes=["aircraft"])
        self.assertEqual(len(boxes), 1)
        self.assertEqual(discarded, 3)

    def test_parse_clamps_coordinates(self) -> None:
        payload = {"boxes": [{"class_id": 0, "x_center": 1.5, "y_center": -0.5, "width": 2.0, "height": 0.2}]}
        boxes, _discarded = parse_vlm_boxes(payload, classes=["a"])
        self.assertAlmostEqual(boxes[0].x_center, 1.0)
        self.assertAlmostEqual(boxes[0].y_center, 0.0)
        self.assertAlmostEqual(boxes[0].width, 1.0)

    def test_parse_respects_max_boxes(self) -> None:
        payload = {
            "boxes": [
                {"class_id": 0, "x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1},
                {"class_id": 0, "x_center": 0.2, "y_center": 0.2, "width": 0.1, "height": 0.1},
                {"class_id": 0, "x_center": 0.3, "y_center": 0.3, "width": 0.1, "height": 0.1},
            ]
        }
        boxes, _discarded = parse_vlm_boxes(payload, classes=["a"], max_boxes=2)
        self.assertEqual(len(boxes), 2)


class TestAnnotateImage(unittest.TestCase):
    def _make_image(self, root: Path, name: str = "img.jpg") -> Path:
        image_path = root / name
        image_path.write_bytes(b"fake-jpeg")
        return image_path

    def test_annotate_image_success(self) -> None:
        client = FakeVLMClient(['{"boxes": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}]}'])
        config = VLMConfig(model="qwen-vl-max", api_key="k", base_url="https://example.com/v1")
        with tempfile.TemporaryDirectory() as temp_dir:
            image = self._make_image(Path(temp_dir))
            result = annotate_image_with_vlm(client, image, config=config, classes=["aircraft"])
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0].class_id, 0)
        self.assertIn("boxes", result.raw_response)
        self.assertEqual(result.discarded, 0)
        self.assertGreaterEqual(result.latency_ms, 0)

    def test_annotate_empty_boxes_means_no_target(self) -> None:
        client = FakeVLMClient(['{"boxes": []}'])
        config = VLMConfig(model="m", api_key="k", base_url="https://example.com/v1")
        with tempfile.TemporaryDirectory() as temp_dir:
            image = self._make_image(Path(temp_dir))
            result = annotate_image_with_vlm(client, image, config=config, classes=["aircraft"])
        self.assertEqual(result.boxes, [])

    def test_annotate_retries_invalid_json_then_raises(self) -> None:
        client = FakeVLMClient(["不是 JSON", "也不是 JSON", "还是不对"])
        config = VLMConfig(model="m", api_key="k", base_url="https://example.com/v1", retries=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            image = self._make_image(Path(temp_dir))
            with self.assertRaises(VLMError):
                annotate_image_with_vlm(client, image, config=config, classes=["aircraft"])
        self.assertEqual(client.calls, 3)  # 初始 + 2 次重试

    def test_annotate_recovers_after_retry(self) -> None:
        client = FakeVLMClient(["不是 JSON", '{"boxes": []}'])
        config = VLMConfig(model="m", api_key="k", base_url="https://example.com/v1", retries=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            image = self._make_image(Path(temp_dir))
            result = annotate_image_with_vlm(client, image, config=config, classes=["aircraft"])
        self.assertEqual(result.boxes, [])
        self.assertEqual(client.calls, 2)


class TestRunBatch(unittest.TestCase):
    def _make_dataset(self, root: Path, names: list[str]) -> tuple[Path, Path]:
        images_dir = root / "images"
        labels_dir = root / "annotations"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        for name in names:
            (images_dir / name).write_bytes(b"fake-jpeg")
        return images_dir, labels_dir

    def test_run_batch_and_resume(self) -> None:
        json_text = '{"boxes": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}]}'
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir, labels_dir = self._make_dataset(root, ["a.jpg", "b.jpg", "c.jpg"])
            config = VLMConfig(model="m", api_key="k", base_url="https://example.com/v1")

            with patch(
                "od_platform.annotation.vlm.OpenAIClient",
                lambda **kwargs: FakeVLMClient([json_text, json_text]),
            ):
                report = run_vlm_annotation(
                    dataset="demo",
                    classes=["aircraft"],
                    config=config,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    limit=2,
                )
            self.assertEqual(report.annotated_new, 2)
            self.assertEqual(report.box_count, 2)
            self.assertTrue((labels_dir / "a.txt").exists())
            self.assertTrue((labels_dir / "b.txt").exists())
            self.assertFalse((labels_dir / "c.txt").exists())

            # 断点续跑：重建后跳过已标注，只处理 c
            with patch(
                "od_platform.annotation.vlm.OpenAIClient",
                lambda **kwargs: FakeVLMClient([json_text]),
            ):
                report2 = run_vlm_annotation(
                    dataset="demo",
                    classes=["aircraft"],
                    config=config,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                )
            self.assertEqual(report2.annotated_new, 1)
            self.assertTrue((labels_dir / "c.txt").exists())
            loaded = read_yolo_label(labels_dir / "c.txt")
            self.assertEqual(len(loaded), 1)

    def test_run_skips_api_failure_keeps_going(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir, labels_dir = self._make_dataset(root, ["a.jpg", "b.jpg"])
            config = VLMConfig(model="m", api_key="k", base_url="https://example.com/v1")

            with patch(
                "od_platform.annotation.vlm.OpenAIClient",
                lambda **kwargs: FakeVLMClient([APIError(429, "限流"), '{"boxes": []}']),
            ):
                report = run_vlm_annotation(
                    dataset="demo",
                    classes=["aircraft"],
                    config=config,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                )
            self.assertEqual(report.annotated_new, 1)
            self.assertEqual(report.failed, ["a.jpg"])
            self.assertTrue((labels_dir / "b.txt").exists())

    def test_run_writes_audit_artifacts(self) -> None:
        """审计产物：annotation_report.json + raw_responses + review_queue.csv。"""
        import json as jsonlib

        json_text = '{"boxes": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}]}'
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir, labels_dir = self._make_dataset(root, ["a.jpg", "b.jpg"])
            config = VLMConfig(model="m", api_key="k", base_url="https://example.com/v1")

            with patch("od_platform.annotation.vlm.paths.RUNS_DIR", Path(temp_dir) / "runs"):
                with patch(
                    "od_platform.annotation.vlm.OpenAIClient",
                    lambda **kwargs: FakeVLMClient([json_text, '{"boxes": []}']),
                ):
                    report = run_vlm_annotation(
                        dataset="demo",
                        classes=["aircraft"],
                        config=config,
                        images_dir=images_dir,
                        labels_dir=labels_dir,
                    )

            audit_dirs = list((Path(temp_dir) / "runs" / "annotation").iterdir())
            self.assertEqual(len(audit_dirs), 1)
            audit_dir = audit_dirs[0]

            report_path = audit_dir / "annotation_report.json"
            self.assertTrue(report_path.exists())
            payload = jsonlib.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["annotated_count"], 2)
            self.assertEqual(payload["total_boxes"], 1)
            self.assertEqual(payload["model"], "m")
            self.assertEqual(payload["class_names"], ["aircraft"])

            # 原始响应保存
            self.assertTrue((audit_dir / "raw_responses" / "a.txt").exists())
            self.assertTrue((audit_dir / "raw_responses" / "b.txt").exists())

            # 复核队列：b 为空标注 → empty 原因
            review_path = audit_dir / "review_queue.csv"
            self.assertTrue(review_path.exists())
            review_lines = review_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(review_lines), 2)  # 表头 + 至少一行
            self.assertIn("empty", review_lines[1])

            self.assertEqual(report.annotated_new, 2)

    def test_run_dry_run_no_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir, labels_dir = self._make_dataset(root, ["a.jpg"])
            config = VLMConfig(model="m", api_key="k", base_url="https://example.com/v1")
            with patch(
                "od_platform.annotation.vlm.OpenAIClient",
                return_value=FakeVLMClient([]),
            ) as fake_factory:
                report = run_vlm_annotation(
                    dataset="demo",
                    classes=["aircraft"],
                    config=config,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    dry_run=True,
                )
            fake_factory.assert_not_called()
            self.assertEqual(report.annotated_new, 0)
            self.assertFalse((labels_dir / "a.txt").exists())


if __name__ == "__main__":
    unittest.main()
