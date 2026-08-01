"""YOLO 格式标注读写。

纯逻辑层，不依赖 GUI：坐标在「原图像素」与「归一化 [0,1]」之间转换，
标注文件为每图一个 ``<class_id> <x_center> <y_center> <width> <height>``
的文本行。写入时对越界坐标做裁剪，保证输出始终可被训练流水线消费。

@FileName:   writer.py
@Function:   标注数据结构、坐标转换与 YOLO txt 读写
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BBox:
    """一条归一化边界框标注。

    Attributes:
        class_id: 类别 ID（0 起）。
        x_center: 中心点 x（归一化 [0, 1]）。
        y_center: 中心点 y（归一化 [0, 1]）。
        width:    框宽（归一化 [0, 1]）。
        height:   框高（归一化 [0, 1]）。
    """

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def is_valid(self) -> bool:
        """宽高必须为正，坐标允许为 0（贴边框）。"""
        return self.width > 0.0 and self.height > 0.0

    def to_pixels(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """归一化坐标转原图像素左上/右下角 (x1, y1, x2, y2)。"""
        x1 = int(round((self.x_center - self.width / 2.0) * image_width))
        y1 = int(round((self.y_center - self.height / 2.0) * image_height))
        x2 = int(round((self.x_center + self.width / 2.0) * image_width))
        y2 = int(round((self.y_center + self.height / 2.0) * image_height))
        return x1, y1, x2, y2


def bbox_from_pixels(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> BBox:
    """由原图像素坐标构造归一化 BBox，并对越界坐标裁剪。

    Args:
        class_id: 类别 ID。
        x1, y1:   框左上角像素坐标。
        x2, y2:   框右下角像素坐标。
        image_width, image_height: 原图尺寸。

    Returns:
        归一化 BBox，坐标被裁剪到 [0, 1] 区间。
    """
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)

    width = max(0.0, right - left)
    height = max(0.0, bottom - top)

    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    return BBox(
        class_id=int(class_id),
        x_center=_clamp((left + right) / 2.0 / image_width),
        y_center=_clamp((top + bottom) / 2.0 / image_height),
        width=_clamp(width / image_width),
        height=_clamp(height / image_height),
    )


def write_yolo_label(label_path: Path, boxes: list[BBox]) -> Path:
    """将标注列表写入 YOLO txt 文件。

    Args:
        label_path: 目标 .txt 路径。
        boxes:      归一化标注列表；无效框（宽或高为 0）将被跳过。

    Returns:
        写入的文件路径。
    """
    lines: list[str] = []
    for box in boxes:
        if not box.is_valid:
            logger.debug("跳过无效标注: %s", box)
            continue
        lines.append(
            f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} "
            f"{box.width:.6f} {box.height:.6f}"
        )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return label_path


def read_yolo_label(label_path: Path) -> list[BBox]:
    """从 YOLO txt 文件读取标注列表。

    Args:
        label_path: 标注文件路径。

    Returns:
        标注列表；文件不存在或行为非法时返回空列表。
    """
    if not label_path.exists():
        return []

    boxes: list[BBox] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if not parts:
            continue
        try:
            values = [float(part) for part in parts]
            if len(values) < 5:
                logger.debug("跳过非 YOLO 行 %s:%d: %s", label_path, line_number, raw)
                continue
            box = BBox(
                class_id=int(values[0]),
                x_center=values[1],
                y_center=values[2],
                width=values[3],
                height=values[4],
            )
            if box.is_valid:
                boxes.append(box)
        except (ValueError, TypeError):
            logger.debug("跳过非法标注行 %s:%d: %s", label_path, line_number, raw)
    return boxes
