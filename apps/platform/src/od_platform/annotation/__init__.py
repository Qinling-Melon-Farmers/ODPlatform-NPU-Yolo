"""数据标注子系统：手动边界框标注并输出 YOLO 格式。

分层设计：``writer`` 为纯逻辑层（可单测），``session`` 管理会话状态，
``canvas`` 提供 OpenCV 交互画布。标注产物写入
``data/raw/<dataset>/annotations/``，可直接进入训练流水线。
"""

from __future__ import annotations

from od_platform.annotation.canvas import AnnotationCanvas, CanvasResult, compute_display_scale
from od_platform.annotation.session import AnnotationSession
from od_platform.annotation.writer import BBox, bbox_from_pixels, read_yolo_label, write_yolo_label

__all__ = [
    "AnnotationCanvas",
    "AnnotationSession",
    "BBox",
    "CanvasResult",
    "bbox_from_pixels",
    "compute_display_scale",
    "read_yolo_label",
    "write_yolo_label",
]
