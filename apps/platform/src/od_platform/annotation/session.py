"""标注会话管理。

会话状态完全由文件系统推导（``labels_dir`` 中已存在的标注文件即进度），
无独立状态文件，避免状态漂移。中断后以 ``--resume`` 重新进入即可从
未标注图片继续。

@FileName:   session.py
@Function:   图片列表、进度跟踪、标注读写
"""

from __future__ import annotations

import logging
from pathlib import Path

from od_platform.annotation.writer import BBox, read_yolo_label, write_yolo_label
from od_platform.common.constants import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {suffix.lower() for suffix in IMAGE_EXTENSIONS}


class AnnotationSession:
    """管理一个数据集目录下的标注任务。

    Attributes:
        images_dir:  原始图片目录。
        labels_dir:  标注输出目录（YOLO txt）。
        classes:     类别名称列表，index 即类别 ID。
    """

    def __init__(
        self,
        *,
        images_dir: Path,
        labels_dir: Path,
        classes: list[str],
        annotated_stems: set[str] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.classes = list(classes)
        self._annotated_stems: set[str] = set(annotated_stems or self._discover_annotated())

    def _discover_annotated(self) -> set[str]:
        """从 labels_dir 中已存在的标注文件推导已标注图片 stem 集合。"""
        if not self.labels_dir.exists():
            return set()
        return {path.stem for path in self.labels_dir.glob("*.txt")}

    @property
    def images(self) -> list[Path]:
        """按文件名排序的图片路径列表。"""
        if not self.images_dir.exists():
            return []
        return sorted(
            (path for path in self.images_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES),
            key=lambda path: path.name,
        )

    @property
    def total_images(self) -> int:
        return len(self.images)

    @property
    def annotated_stems(self) -> set[str]:
        return self._annotated_stems

    @property
    def annotated_count(self) -> int:
        return len(self._annotated_stems)

    @property
    def remaining_count(self) -> int:
        return sum(1 for image in self.images if image.stem not in self._annotated_stems)

    @property
    def progress(self) -> tuple[int, int]:
        """返回 (已标注数, 总数)。"""
        return self.annotated_count, self.total_images

    def is_annotated(self, image_stem: str) -> bool:
        return image_stem in self._annotated_stems

    def next_unannotated(self) -> Path | None:
        """返回下一张未标注图片路径；全部完成时返回 None。"""
        for image in self.images:
            if image.stem not in self._annotated_stems:
                return image
        return None

    def label_path_for(self, image_stem: str) -> Path:
        return self.labels_dir / f"{image_stem}.txt"

    def load_labels(self, image_stem: str) -> list[BBox]:
        return read_yolo_label(self.label_path_for(image_stem))

    def save_labels(self, image_stem: str, boxes: list[BBox]) -> Path:
        """保存标注并更新进度集合。

        越界类别 ID 的框会被过滤（避免产出无法训练的标注）。

        Args:
            image_stem: 图片文件名（不含扩展名）。
            boxes:      归一化标注列表；空列表也视为已标注（背景图）。

        Returns:
            写入的标注文件路径。
        """
        kept: list[BBox] = []
        for box in boxes:
            if box.class_id < 0 or box.class_id >= len(self.classes):
                logger.warning(
                    "标注类别 ID %d 超出范围 [0, %d)，该框已被过滤",
                    box.class_id,
                    len(self.classes),
                )
                continue
            kept.append(box)
        path = write_yolo_label(self.label_path_for(image_stem), kept)
        self._annotated_stems.add(image_stem)
        logger.debug("已保存标注: %s (%d 框)", path, len(boxes))
        return path
