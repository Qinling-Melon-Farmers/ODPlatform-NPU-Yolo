"""Pascal VOC 转 YOLO 转换器。

@FileName:   pascal_voc.py
@Function:   Pascal VOC XML detection labels 转 YOLO txt labels
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register

logger = logging.getLogger(__name__)


def _class_id(name: str, classes: list[str], discovering: bool) -> int | None:
    if name not in classes:
        if not discovering:
            logger.debug("%s 不在类别白名单中，跳过", name)
            return None
        classes.append(name)
    return classes.index(name)


@register(AnnotationFormat.PASCAL_VOC, supported_tasks=(Task.DETECT,))
def convert_voc(input_path: Path, output_labels_dir: Path, options: ConvertOptions) -> list[str]:
    """将 Pascal VOC XML 目录转换为 YOLO label 目录。

    Args:
        input_path: XML 标注目录。
        output_labels_dir: YOLO label 输出目录。
        options: 转换选项。

    Returns:
        转换过程中使用的类别列表。
    """
    xml_files = sorted(input_path.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"在 {input_path} 下未找到任何 XML")

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    classes = list(options.classes) if options.classes else []
    discovering = options.classes is None

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None:
            logger.warning("%s 缺少 <size>，跳过", xml_path.name)
            continue
        image_width = float(size.findtext("width", "0"))
        image_height = float(size.findtext("height", "0"))
        if image_width <= 0 or image_height <= 0:
            logger.warning("%s 尺寸非法，跳过", xml_path.name)
            continue

        lines: list[str] = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            bbox = obj.find("bndbox")
            if not name or bbox is None:
                continue
            cls_id = _class_id(name, classes, discovering)
            if cls_id is None:
                continue
            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))
            box_width = xmax - xmin
            box_height = ymax - ymin
            if box_width <= 0 or box_height <= 0:
                logger.warning("%s 存在非法 bbox，已跳过", xml_path.name)
                continue
            center_x = (xmin + xmax) / 2 / image_width
            center_y = (ymin + ymax) / 2 / image_height
            norm_width = box_width / image_width
            norm_height = box_height / image_height
            lines.append(f"{cls_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")

        (output_labels_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    logger.info("VOC 转换完成: %d 个文件, %d 个类别", len(xml_files), len(classes))
    return classes
