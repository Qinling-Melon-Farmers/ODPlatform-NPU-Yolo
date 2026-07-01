"""Class balance report utilities.

This module only analyzes and renders warnings. It never filters classes or
modifies data.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from od_platform.common.constants import (
    CLASS_MIN_BOX_SHARE,
    CLASS_MIN_BOXES_WARN,
    CLASS_MIN_IMAGES_HARD,
    AnnotationFormat,
)
from od_platform.common.string_utils import (
    format_table_row,
    format_table_separator,
    get_display_width,
)

_CLASSES_FILTERABLE = {AnnotationFormat.PASCAL_VOC, AnnotationFormat.COCO}

_OK = "OK"
_BOTH = "严重不足(图/框)"
_BOX = "框数偏低"
_IMG = "图数偏低"


@dataclass(frozen=True)
class ClassStat:
    """Statistics for one class."""

    name: str
    image_count: int
    box_count: int
    image_pct: float
    box_pct: float
    status: str

    @property
    def ok(self) -> bool:
        return self.status == _OK


@dataclass(frozen=True)
class ClassBalanceReport:
    """A read-only class balance report."""

    stats: list[ClassStat]
    total_images: int
    total_boxes: int
    usefulness_img_floor: int

    @property
    def flagged(self) -> list[ClassStat]:
        return [stat for stat in self.stats if not stat.ok]

    @property
    def has_warnings(self) -> bool:
        return bool(self.flagged)

    def keeper_classes(self) -> list[str]:
        return [stat.name for stat in self.stats if stat.ok]


def analyze_class_balance(
    labels_per_image: dict[str, list[str]],
    classes: list[str],
    train_rate: float,
    val_rate: float,
) -> ClassBalanceReport:
    """Analyze image-level and box-level class balance."""
    total_images = len(labels_per_image)
    test_rate = max(0.0, 1.0 - train_rate - val_rate)

    image_count = dict.fromkeys(classes, 0)
    box_count = dict.fromkeys(classes, 0)
    for labels in labels_per_image.values():
        for name, count in Counter(labels).items():
            if name in box_count:
                box_count[name] += count
                image_count[name] += 1

    total_boxes = sum(box_count.values())
    split_rates = [rate for rate in (val_rate, test_rate) if rate > 0]
    smallest_eval_rate = min(split_rates) if split_rates else 0.0
    floor = (
        max(CLASS_MIN_IMAGES_HARD, math.ceil(1.0 / smallest_eval_rate))
        if smallest_eval_rate > 0
        else CLASS_MIN_IMAGES_HARD
    )

    stats: list[ClassStat] = []
    for name in classes:
        images = image_count[name]
        boxes = box_count[name]
        box_share = boxes / total_boxes if total_boxes else 0.0
        img_starved = images < floor
        box_starved = boxes < CLASS_MIN_BOXES_WARN or box_share < CLASS_MIN_BOX_SHARE
        status = _BOTH if img_starved and box_starved else _BOX if box_starved else _IMG if img_starved else _OK
        stats.append(
            ClassStat(
                name=name,
                image_count=images,
                box_count=boxes,
                image_pct=images / total_images if total_images else 0.0,
                box_pct=box_share,
                status=status,
            )
        )

    return ClassBalanceReport(
        stats=stats,
        total_images=total_images,
        total_boxes=total_boxes,
        usefulness_img_floor=floor,
    )


def render_balance_report(report: ClassBalanceReport, source_format: str) -> list[tuple[str, bool]]:
    """Render report lines as ``(line, is_warning)`` pairs."""
    flagged = {stat.name for stat in report.flagged}

    name_w = max(get_display_width("类别"), max((get_display_width(s.name) for s in report.stats), default=0)) + 2
    img_w = max(get_display_width("图片数"), max((len(str(s.image_count)) for s in report.stats), default=0)) + 2
    box_w = max(get_display_width("框数"), max((len(str(s.box_count)) for s in report.stats), default=0)) + 2
    pct_w = max(get_display_width("图占比"), get_display_width("框占比"), 7) + 2
    status_w = max(get_display_width("状态"), max((get_display_width(s.status) for s in report.stats), default=0)) + 1
    widths = [name_w, img_w, box_w, pct_w, pct_w, status_w]
    aligns = ["left", "right", "right", "right", "right", "left"]

    out: list[tuple[str, bool]] = []
    out.append(("数据平衡性报告".center(sum(widths) + 3 * (len(widths) - 1), "="), False))
    out.append((format_table_row(["类别", "图片数", "框数", "图占比", "框占比", "状态"], widths, aligns), False))
    out.append((format_table_separator(widths), False))
    for stat in report.stats:
        line = format_table_row(
            [
                stat.name,
                str(stat.image_count),
                str(stat.box_count),
                f"{stat.image_pct * 100:.2f}%",
                f"{stat.box_pct * 100:.2f}%",
                stat.status,
            ],
            widths,
            aligns,
        )
        out.append((line, stat.name in flagged))

    out.append((format_table_separator(widths), False))
    out.append(
        (
            f"合计: {report.total_images} 张图, {report.total_boxes} 个框; "
            f"图数可用下限 >= {report.usefulness_img_floor}",
            False,
        )
    )

    if report.has_warnings:
        out.append(("", False))
        out.append((f"以下类别偏弱: {', '.join(stat.name for stat in report.flagged)}", False))
        out.append(("这只是提醒，工具不会自动删类或改数据。", False))
        if source_format in _CLASSES_FILTERABLE:
            keepers = report.keeper_classes()
            if keepers and len(keepers) < len(report.stats):
                classes_arg = " ".join(f'"{name}"' for name in keepers)
                out.append((f"可考虑补样本，或重跑时显式使用: --classes {classes_arg}", False))
        else:
            out.append(("YOLO 源数据不建议用 --classes 删类，避免 yaml 与 txt class id 不一致。", False))

    return out
