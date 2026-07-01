import tempfile
import unittest
from pathlib import Path

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.materializer import SplitOutputDirs, materialize
from od_platform.data_pipeline.split.registry import SplitOptions, list_strategies
from od_platform.data_pipeline.split.service import collect_yolo_pairs, split_pairs


class TestDataPipelineSplit(unittest.TestCase):
    def _make_yolo_sample(self, root: Path, index: int) -> tuple[Path, Path]:
        image = root / "images" / f"sample_{index:03d}.jpg"
        label = root / "labels" / f"sample_{index:03d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        label.write_text("0 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")
        return image, label

    def test_split_registry_lists_random_strategy(self) -> None:
        strategies = list_strategies()

        self.assertIn(SplitStrategy.RANDOM, strategies)

    def test_split_registry_lists_all_strategies(self) -> None:
        """验证 RANDOM / STRATIFIED / STRATIFIED_MULTILABEL 均已注册。"""
        strategies = list_strategies()

        self.assertIn(SplitStrategy.RANDOM, strategies)
        self.assertIn(SplitStrategy.STRATIFIED, strategies)
        self.assertIn(SplitStrategy.STRATIFIED_MULTILABEL, strategies)
        self.assertIn("随机", strategies[SplitStrategy.RANDOM])

    def test_stratified_raises_not_implemented(self) -> None:
        """分层策略目前为骨架，调用时抛出 NotImplementedError。"""
        with self.assertRaises(NotImplementedError):
            split_pairs([], SplitStrategy.STRATIFIED)

    def test_stratified_multilabel_raises_not_implemented(self) -> None:
        """多标签分层策略目前为骨架，调用时抛出 NotImplementedError。"""
        with self.assertRaises(NotImplementedError):
            split_pairs([], SplitStrategy.STRATIFIED_MULTILABEL)

    def test_collect_yolo_pairs_requires_matching_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            (images_dir / "a.jpg").write_bytes(b"")

            with self.assertRaises(FileNotFoundError):
                collect_yolo_pairs(images_dir, labels_dir)

    def test_random_split_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pairs = [self._make_yolo_sample(root, index) for index in range(10)]
            options = SplitOptions(train_rate=0.6, val_rate=0.2, test_rate=0.2, random_state=42)

            manifest_a = split_pairs(pairs, SplitStrategy.RANDOM, options)
            manifest_b = split_pairs(pairs, SplitStrategy.RANDOM, options)

            self.assertEqual(manifest_a.summary(), {"train": 6, "val": 2, "test": 2, "total": 10})
            self.assertEqual(manifest_a.all_pairs(), manifest_b.all_pairs())
            self.assertEqual(manifest_a.random_state, 42)
            self.assertEqual(manifest_a.strategy, SplitStrategy.RANDOM)

    def test_random_split_rejects_invalid_rates(self) -> None:
        with self.assertRaises(ValueError):
            split_pairs(
                [],
                SplitStrategy.RANDOM,
                SplitOptions(train_rate=0.8, val_rate=0.2, test_rate=0.2),
            )

    def test_materialize_writes_yolo_train_val_test_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "dataset"
            pairs = [self._make_yolo_sample(source, index) for index in range(5)]
            manifest = split_pairs(
                pairs,
                SplitStrategy.RANDOM,
                SplitOptions(train_rate=0.6, val_rate=0.2, test_rate=0.2, random_state=0),
            )

            counts = materialize(manifest, SplitOutputDirs.for_dataset_root(output))

            self.assertEqual(counts, {"train": 3, "val": 1, "test": 1})
            self.assertEqual(len(list((output / "train" / "images").glob("*.jpg"))), 3)
            self.assertEqual(len(list((output / "train" / "labels").glob("*.txt"))), 3)
            self.assertEqual(len(list((output / "val" / "images").glob("*.jpg"))), 1)
            self.assertEqual(len(list((output / "test" / "labels").glob("*.txt"))), 1)
            self.assertEqual(len(list((source / "images").glob("*.jpg"))), 5)
            self.assertEqual(len(list((source / "labels").glob("*.txt"))), 5)


if __name__ == "__main__":
    unittest.main()
