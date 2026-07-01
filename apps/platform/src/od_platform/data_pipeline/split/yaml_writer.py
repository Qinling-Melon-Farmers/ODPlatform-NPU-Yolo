"""YOLO dataset.yaml 生成工具。

@FileName:   yaml_writer.py
@Function:   根据划分结果生成 ultralytics 可直接消费的 dataset.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

from od_platform.data_pipeline.split.materializer import SplitOutputDirs

logger = logging.getLogger(__name__)


def write_dataset_yaml(
    dataset_root: Path,
    classes: list[str],
    dirs: SplitOutputDirs,
    output_path: Path | None = None,
) -> Path:
    """生成 ultralytics 兼容的 dataset.yaml。

    Args:
        dataset_root: 数据集根目录（绝对路径，写入 yaml 的 ``path`` 字段）。
        classes: 类别名称列表，下标即 class_id。
        dirs: 划分输出目录集合，从中读取 train/val/test 相对路径。
        output_path: yaml 输出路径，默认 ``dataset_root / "dataset.yaml"``。

    Returns:
        写入的 yaml 文件路径。
    """
    rel_paths = dirs.yaml_rel_paths()
    output = output_path or dataset_root / "dataset.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# ODPlatform auto-generated dataset config")
    lines.append(f"path: {dataset_root}")
    lines.append("")
    lines.append(f"train: {rel_paths['train']}")
    lines.append(f"val: {rel_paths['val']}")
    lines.append(f"test: {rel_paths['test']}")
    lines.append("")
    lines.append(f"nc: {len(classes)}")
    lines.append("names:")
    for idx, name in enumerate(classes):
        lines.append(f"  {idx}: {name}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("dataset.yaml 已写入: %s（%d 个类别）", output, len(classes))
    return output
