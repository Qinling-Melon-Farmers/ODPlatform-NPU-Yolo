"""Core primitives used by :mod:`od_platform.visualization`."""

from __future__ import annotations

from od_platform.visualization.core.data_types import (
    Detection,
    DrawStyle,
    LabelLayout,
    LabelPosition,
)
from od_platform.visualization.core.draw_utils import LayoutCalculator, RoundedRect
from od_platform.visualization.core.renderers import PillowTextRenderer
from od_platform.visualization.core.text_cache import TextSizeCache

__all__ = [
    "Detection",
    "DrawStyle",
    "LabelLayout",
    "LabelPosition",
    "LayoutCalculator",
    "PillowTextRenderer",
    "RoundedRect",
    "TextSizeCache",
]
