"""YOLO 自身验证转换器。

@FileName:   yolo.py
@Function:   校验 YOLO detection txt labels，并可复制到输出目录
"""

from __future__ import annotations

import logging
from pathlib import Path

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register

logger = logging.getLogger(__name__)


def _parse_yolo_line(line: str, label_path: Path, line_no: int) -> tuple[int, list[float]]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"{label_path}:{line_no} YOLO 行必须是 5 列，实际 {len(parts)} 列")
    try:
        cls_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_no} YOLO 行包含非数值字段") from exc
    if cls_id < 0:
        raise ValueError(f"{label_path}:{line_no} class id 不能为负数")
    center_x, center_y, width, height = values
    if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
        raise ValueError(f"{label_path}:{line_no} center 坐标必须在 [0, 1] 内")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError(f"{label_path}:{line_no} width/height 必须在 (0, 1] 内")
    return cls_id, values


def _validate_label(label_path: Path, max_class_id: int | None) -> tuple[list[str], set[int]]:
    cleaned_lines: list[str] = []
    discovered_ids: set[int] = set()
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        cls_id, values = _parse_yolo_line(line, label_path, line_no)
        if max_class_id is not None and cls_id > max_class_id:
            raise ValueError(f"{label_path}:{line_no} class id {cls_id} 超出类别表范围")
        discovered_ids.add(cls_id)
        cleaned_lines.append(
            f"{cls_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"
        )
    return cleaned_lines, discovered_ids


@register(AnnotationFormat.YOLO, supported_tasks=(Task.DETECT,))
def convert_yolo(input_dir: Path, output_labels_dir: Path, options: ConvertOptions) -> list[str]:
    """验证 YOLO label 目录，并输出规范化后的 label 文件。

    Args:
        input_dir: YOLO txt label 目录。
        output_labels_dir: 验证后 label 输出目录。与输入目录相同时只做校验。
        options: 转换选项。提供 ``classes`` 时会校验 class id 不越界。

    Returns:
        类别列表；未提供类别名时返回发现到的 class id 字符串列表。
    """
    label_files = sorted(input_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"在 {input_dir} 下未找到任何 YOLO txt label")

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    max_class_id = len(options.classes) - 1 if options.classes is not None else None
    discovered_ids: set[int] = set()

    for label_path in label_files:
        cleaned_lines, ids = _validate_label(label_path, max_class_id)
        discovered_ids.update(ids)
        target_path = output_labels_dir / label_path.name
        if target_path.resolve() == label_path.resolve():
            continue
        target_path.write_text("\n".join(cleaned_lines), encoding="utf-8")

    if options.classes is not None:
        classes = list(options.classes)
    else:
        classes = [str(class_id) for class_id in sorted(discovered_ids)]

    logger.info("YOLO 校验完成: %d 个文件, %d 个类别", len(label_files), len(classes))
    return classes
