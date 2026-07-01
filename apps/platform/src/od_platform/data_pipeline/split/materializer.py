"""数据集划分结果落盘工具。

@FileName:   materializer.py
@Function:   将 SplitManifest 三组样本落盘到 train/val/test 目录
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from od_platform.data_pipeline.split.manifest import PairList, SplitManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitOutputDirs:
    """划分后数据集输出目录集合。"""

    train_images: Path
    val_images: Path
    test_images: Path
    train_labels: Path
    val_labels: Path
    test_labels: Path

    @classmethod
    def for_dataset_root(cls, root: Path) -> SplitOutputDirs:
        """按 YOLO 数据集标准目录生成输出目录集合。"""
        return cls(
            train_images=root / "train" / "images",
            val_images=root / "val" / "images",
            test_images=root / "test" / "images",
            train_labels=root / "train" / "labels",
            val_labels=root / "val" / "labels",
            test_labels=root / "test" / "labels",
        )

    def all_dirs(self) -> tuple[Path, ...]:
        """返回全部输出目录。"""
        return (
            self.train_images,
            self.train_labels,
            self.val_images,
            self.val_labels,
            self.test_images,
            self.test_labels,
        )

    def yaml_rel_paths(self) -> dict[str, str]:
        """返回 dataset.yaml 所需的 train/val/test 相对路径。

        约定：train/val/test 各映射到对应的 images 子目录。
        这是 yaml_writer 获取路径真相的唯一来源——禁止在别处硬编码
        ``"train/images"`` 等字符串。
        """
        return {
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
        }


def _place(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _materialize_one(pairs: PairList, images_dir: Path, labels_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        _place(image_path, images_dir / image_path.name)
        _place(label_path, labels_dir / label_path.name)
    return len(pairs)


def materialize(manifest: SplitManifest, dirs: SplitOutputDirs) -> dict[str, int]:
    """将划分结果落盘为 YOLO train/val/test 目录结构。

    Args:
        manifest: 划分结果。
        dirs: 输出目录集合。

    Returns:
        每个 split 的样本数量。
    """
    for directory in dirs.all_dirs():
        if directory.exists():
            shutil.rmtree(directory)

    counts = {
        "train": _materialize_one(manifest.train, dirs.train_images, dirs.train_labels),
        "val": _materialize_one(manifest.val, dirs.val_images, dirs.val_labels),
        "test": _materialize_one(manifest.test, dirs.test_images, dirs.test_labels),
    }
    logger.info("materialized %s samples to %s", counts, dirs)
    return counts
