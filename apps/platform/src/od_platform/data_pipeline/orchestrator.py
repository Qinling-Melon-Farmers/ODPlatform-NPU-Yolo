"""End-to-end dataset transformation pipeline."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from od_platform.common import paths
from od_platform.common.constants import (
    COVERAGE_HARD_THRESHOLD,
    COVERAGE_SOFT_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_STRATEGY,
    IMAGE_EXTENSIONS,
    AnnotationFormat,
    Task,
)
from od_platform.common.refs import resolve_dataset
from od_platform.data_pipeline.convert.registry import ConvertOptions
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
from od_platform.data_pipeline.report import analyze_class_balance, render_balance_report
from od_platform.data_pipeline.split.manifest import PairList
from od_platform.data_pipeline.split.materializer import SplitOutputDirs, materialize
from od_platform.data_pipeline.split.service import split_pairs
from od_platform.data_pipeline.split.yaml_writer import write_dataset_yaml

logger = logging.getLogger(__name__)


class DatasetPipeline:
    """Convert, report, split, materialize and write yaml for one dataset."""

    def __init__(
        self,
        dataset: str,
        annotation_format: str,
        *,
        task: str = Task.DETECT,
        train_rate: float = 0.8,
        val_rate: float = 0.1,
        classes: list[str] | None = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    ) -> None:
        if annotation_format == AnnotationFormat.YOLO and not classes:
            raise ValueError("YOLO 源数据需要显式提供 --classes，避免 yaml 与 txt class id 不一致")

        self.annotation_format = annotation_format
        self.task = Task.ensure_end_to_end(task)
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.random_state = random_state
        self.split_strategy = split_strategy
        self.options = ConvertOptions(task=self.task, classes=classes)

        self.raw_root = resolve_dataset(dataset)
        self.dataset_name = self.raw_root.name
        self.raw_images = self.raw_root / "images"
        self.raw_annotations = self.raw_root / "annotations"
        self.processed_root = paths.dataset_processed_dir(self.dataset_name)
        self.output_dirs = SplitOutputDirs.for_dataset_root(self.processed_root)
        self.yaml_out = paths.dataset_yaml_path(self.dataset_name)

    def run(self) -> dict[str, object]:
        """Run the full pipeline."""
        logger.info(
            "开始处理数据集 %s: format=%s, task=%s, split=%s",
            self.dataset_name,
            self.annotation_format,
            self.task,
            self.split_strategy,
        )
        self._check_raw()

        with tempfile.TemporaryDirectory(prefix="odp_pipe_") as temp_dir:
            staging_labels = Path(temp_dir) / "labels"
            classes = convert_data_to_yolo(
                self.raw_annotations,
                staging_labels,
                self.annotation_format,
                self.options,
            )
            pairs = self._pair_images_with_labels(staging_labels)
            if not pairs:
                raise ValueError("转换后没有可配对的图片和标签")

            labels_per_image = self._build_labels_per_image(pairs, classes)
            report = analyze_class_balance(labels_per_image, classes, self.train_rate, self.val_rate)
            for line, is_warning in render_balance_report(report, self.annotation_format):
                (logger.warning if is_warning else logger.info)(line)

            manifest = split_pairs(
                pairs,
                strategy=self.split_strategy,
                train_rate=self.train_rate,
                val_rate=self.val_rate,
                random_state=self.random_state,
                labels_per_image=labels_per_image,
            )
            counts = materialize(manifest, self.output_dirs)
            yaml_path = write_dataset_yaml(
                self.yaml_out,
                dataset_root=self.processed_root,
                classes=classes,
                dirs=self.output_dirs,
                manifest=manifest,
                dataset_name=self.dataset_name,
                source_format=self.annotation_format,
                task=self.task,
            )

        return {"counts": counts, "yaml": str(yaml_path)}

    def _check_raw(self) -> None:
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"数据集目录不存在: {self.raw_root}")
        if not self.raw_images.is_dir():
            raise FileNotFoundError(f"缺少 images 子目录: {self.raw_images}")
        if not self.raw_annotations.is_dir():
            raise FileNotFoundError(f"缺少 annotations 子目录: {self.raw_annotations}")
        self._check_coverage()

    def _check_coverage(self) -> None:
        images = self._list_images(self.raw_images)
        if not images:
            raise FileNotFoundError(f"{self.raw_images} 下没有任何支持的图片")
        if self.annotation_format == AnnotationFormat.COCO:
            logger.debug("COCO 使用单 JSON 标注，跳过逐图片覆盖率检查")
            return

        annotations = [path for path in self.raw_annotations.iterdir() if path.is_file()]
        coverage = len(annotations) / len(images)
        logger.info("图片-标注覆盖率: %d/%d = %.1f%%", len(annotations), len(images), coverage * 100)
        if coverage < COVERAGE_HARD_THRESHOLD:
            raise ValueError(
                f"图片-标注覆盖率 {coverage:.1%} 低于硬阈值 {COVERAGE_HARD_THRESHOLD:.0%}，"
                "请检查 annotations/ 或 --format"
            )
        if coverage < COVERAGE_SOFT_THRESHOLD:
            logger.warning(
                "图片-标注覆盖率 %.1f%% 低于建议阈值 %.0f%%，可继续但建议核对",
                coverage * 100,
                COVERAGE_SOFT_THRESHOLD * 100,
            )

    def _pair_images_with_labels(self, labels_dir: Path) -> PairList:
        image_index = {image.stem: image for image in self._list_images(self.raw_images)}
        pairs: PairList = []
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = image_index.get(label_path.stem)
            if image_path is not None:
                pairs.append((image_path, label_path))
        return pairs

    def _build_labels_per_image(self, pairs: PairList, classes: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for image_path, label_path in pairs:
            labels: list[str] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                if 0 <= class_id < len(classes):
                    labels.append(classes[class_id])
            result[image_path.stem] = labels
        return result

    @staticmethod
    def _list_images(images_dir: Path) -> list[Path]:
        suffixes = {suffix.lower() for suffix in IMAGE_EXTENSIONS}
        return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
