"""标注复核视图：加载 VLM 审计复核队列，人工精修并保存。

数据流：
runs/annotation/<run_id>/review_queue.csv
  → 样本列表（image + reason）
  → 图片画布（已有标注框叠加）
  → 框编辑表格（类别/坐标可改，可删行）
  → 保存覆盖 data/raw/<dataset>/annotations/<stem>.txt
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from od_platform.annotation.writer import BBox

# ---- 纯函数（可单测） ----


def list_audit_runs(runs_dir: Path) -> list[Path]:
    """列出 runs/annotation/ 下的审计目录（按名称倒序）。"""
    annotation_root = runs_dir / "annotation"
    if not annotation_root.exists():
        return []
    return sorted(
        (path for path in annotation_root.iterdir() if path.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )


def load_review_queue(audit_dir: Path) -> list[dict[str, str]]:
    """解析 review_queue.csv 为行字典列表。"""
    queue_path = audit_dir / "review_queue.csv"
    if not queue_path.exists():
        return []
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_image_path(label_path: Path, image_name: str) -> Path | None:
    """由标注文件路径推导图片路径：annotations/ → images/（同 stem 任意扩展名）。"""
    images_dir = label_path.parent.parent / "images"
    if not images_dir.exists():
        return None
    stem = Path(image_name).stem
    candidates = sorted(images_dir.glob(f"{stem}.*"))
    return candidates[0] if candidates else None


def draw_boxes(image: np.ndarray, boxes: list[BBox]) -> np.ndarray:
    """在图像上叠加标注框（BGR），返回新图像。"""
    frame = image.copy()
    height, width = frame.shape[:2]
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.to_pixels(width, height)
        color = (255, 56, 56) if index == 0 else (120, 200, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"cls{box.class_id}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def boxes_to_rows(boxes: list[BBox]) -> list[list[str]]:
    """BBox 列表 → 表格行（class_id/x_center/y_center/width/height）。"""
    return [
        [str(box.class_id), f"{box.x_center:.6f}", f"{box.y_center:.6f}", f"{box.width:.6f}", f"{box.height:.6f}"]
        for box in boxes
    ]


def rows_to_boxes(rows: list[list[str]]) -> list[BBox]:
    """表格行 → BBox 列表（跳过非法行）。"""
    boxes: list[BBox] = []
    for row in rows:
        try:
            box = BBox(
                class_id=int(float(row[0])),
                x_center=float(row[1]),
                y_center=float(row[2]),
                width=float(row[3]),
                height=float(row[4]),
            )
            if box.is_valid:
                boxes.append(box)
        except (ValueError, IndexError):
            continue
    return boxes
