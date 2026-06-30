"""COCO JSON 转 YOLO 转换器。

@FileName:   coco.py
@Function:   COCO detection JSON 转 YOLO txt labels
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register

logger = logging.getLogger(__name__)


def _find_coco_json(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    json_files = sorted(input_path.glob("*.json"))
    if len(json_files) != 1:
        raise FileNotFoundError(f"{input_path} 下需要恰好 1 个 COCO JSON，实际找到 {len(json_files)} 个")
    return json_files[0]


def _image_stem(file_name: str) -> str:
    return Path(file_name).stem


def _build_classes(categories: list[dict[str, Any]], options: ConvertOptions) -> list[str]:
    if options.classes is not None:
        return list(options.classes)
    return [str(item["name"]) for item in sorted(categories, key=lambda item: int(item["id"]))]


def _format_yolo_line(
    cls_id: int,
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> str | None:
    x, y, width, height = bbox
    x_min = max(0.0, x)
    y_min = max(0.0, y)
    x_max = min(image_width, x + width)
    y_max = min(image_height, y + height)
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    center_x = (x_min + clipped_width / 2) / image_width
    center_y = (y_min + clipped_height / 2) / image_height
    norm_width = clipped_width / image_width
    norm_height = clipped_height / image_height
    return f"{cls_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"


@register(AnnotationFormat.COCO, supported_tasks=(Task.DETECT,))
def convert_coco(input_path: Path, output_labels_dir: Path, options: ConvertOptions) -> list[str]:
    """将 COCO detection JSON 转换为 YOLO label 目录。

    Args:
        input_path: COCO JSON 文件，或仅包含一个 JSON 的目录。
        output_labels_dir: YOLO label 输出目录。
        options: 转换选项。``classes`` 为空时按 COCO category id 升序生成类别表。

    Returns:
        转换过程中使用的类别列表。
    """
    coco_json = _find_coco_json(input_path)
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    if not images:
        raise ValueError(f"{coco_json} 缺少 images")
    if not categories:
        raise ValueError(f"{coco_json} 缺少 categories")

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    classes = _build_classes(categories, options)
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    image_info = {
        int(item["id"]): {
            "stem": _image_stem(str(item["file_name"])),
            "width": float(item["width"]),
            "height": float(item["height"]),
        }
        for item in images
    }
    lines_by_image = {image_id: [] for image_id in image_info}

    for ann in annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_info:
            logger.warning("annotation %s 指向未知 image_id=%s，跳过", ann.get("id"), image_id)
            continue
        if int(ann.get("iscrowd", 0)) == 1:
            logger.debug("annotation %s 是 crowd，跳过", ann.get("id"))
            continue
        category_name = category_names.get(int(ann["category_id"]))
        if category_name is None or category_name not in classes:
            continue
        bbox = [float(value) for value in ann.get("bbox", [])]
        if len(bbox) != 4:
            logger.warning("annotation %s bbox 非法，跳过", ann.get("id"))
            continue
        info = image_info[image_id]
        if info["width"] <= 0 or info["height"] <= 0:
            logger.warning("image_id=%s 尺寸非法，跳过", image_id)
            continue
        line = _format_yolo_line(
            cls_id=classes.index(category_name),
            bbox=bbox,
            image_width=info["width"],
            image_height=info["height"],
        )
        if line is not None:
            lines_by_image[image_id].append(line)

    for image_id, info in image_info.items():
        label_path = output_labels_dir / f"{info['stem']}.txt"
        label_path.write_text("\n".join(lines_by_image[image_id]), encoding="utf-8")

    logger.info("COCO 转换完成: %d 张图片, %d 个类别", len(images), len(classes))
    return classes
