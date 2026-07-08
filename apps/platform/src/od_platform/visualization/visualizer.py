"""Beautified visualization for YOLO-style detection results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from od_platform.visualization.core.data_types import Detection, DrawStyle
from od_platform.visualization.core.draw_utils import LayoutCalculator, RoundedRect
from od_platform.visualization.core.renderers import PillowTextRenderer
from od_platform.visualization.core.text_cache import TextSizeCache


class BeautifyVisualizer:
    """Draw rounded boxes and Chinese-capable labels on detection results."""

    def __init__(
        self,
        labels: Sequence[str],
        label_mapping: Mapping[str, str] | None = None,
        color_mapping: Mapping[str, tuple[int, int, int]] | None = None,
        default_color: tuple[int, int, int] = (0, 255, 0),
        font_path: str | None = None,
        font_sizes: tuple[int, ...] | None = None,
    ) -> None:
        self.labels = list(labels)
        self.label_mapping = dict(label_mapping or {})
        self.color_mapping = dict(color_mapping or {})
        self.default_color = default_color
        self._size_cache = TextSizeCache(
            labels=self.labels,
            label_mapping=self.label_mapping,
            font_path=font_path,
            font_sizes=font_sizes,
        )
        self._renderer = PillowTextRenderer(size_cache=self._size_cache)

    def draw(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        style: DrawStyle | None = None,
        *,
        use_label_mapping: bool = False,
    ) -> np.ndarray:
        """Draw detections on a copy of ``image`` and return the annotated image."""
        if not detections:
            return image.copy()

        h, w = image.shape[:2]
        style = style or DrawStyle.from_image_size(h, w)
        result = image.copy()
        texts: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            color = self.color_mapping.get(detection.label, detection.color or self.default_color)
            display_label = (
                self.label_mapping.get(detection.label, detection.label)
                if use_label_mapping
                else detection.label
            )
            label_text = f"{display_label} {detection.confidence * 100:.1f}%"
            text_size = self._size_cache.get_size(display_label, style.font_size)
            layout = LayoutCalculator.compute(detection.box, text_size, (h, w), style)

            RoundedRect.bordered(
                result,
                (x1, y1),
                (x2, y2),
                color,
                style.line_width,
                style.radius,
                LayoutCalculator.get_corners(layout, for_detection=True),
            )
            lx1, ly1, lx2, ly2 = layout.box
            RoundedRect.filled(
                result,
                (lx1, ly1),
                (lx2, ly2),
                color,
                style.radius,
                LayoutCalculator.get_corners(layout, for_detection=False),
            )
            texts.append((label_text, layout.text_pos, style.text_color))

        return self._renderer.render_batch(result, texts, style)

    @staticmethod
    def from_yolo_result(
        result: Any,
        *,
        names: Mapping[int, str] | Sequence[str] | None = None,
        color_mapping: Mapping[str, tuple[int, int, int]] | None = None,
    ) -> list[Detection]:
        """Convert one Ultralytics result object into :class:`Detection` objects."""
        return detections_from_yolo_result(result, names=names, color_mapping=color_mapping)


def detections_from_yolo_result(
    result: Any,
    *,
    names: Mapping[int, str] | Sequence[str] | None = None,
    color_mapping: Mapping[str, tuple[int, int, int]] | None = None,
) -> list[Detection]:
    """Convert one Ultralytics result object into visualization detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = names or getattr(result, "names", None) or {}
    color_mapping = color_mapping or {}
    data = getattr(boxes, "data", None)
    if data is None:
        return []
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()

    detections: list[Detection] = []
    for row in data:
        if len(row) < 6:
            continue
        class_id = int(row[5])
        if isinstance(names, Mapping):
            label = names.get(class_id, f"cls_{class_id}")
        else:
            label = names[class_id] if 0 <= class_id < len(names) else f"cls_{class_id}"
        detections.append(
            Detection(
                box=(int(row[0]), int(row[1]), int(row[2]), int(row[3])),
                confidence=float(row[4]),
                label=label,
                color=color_mapping.get(label, (0, 255, 0)),
            )
        )
    return detections
