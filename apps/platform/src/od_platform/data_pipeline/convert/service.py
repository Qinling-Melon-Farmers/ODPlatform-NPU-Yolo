"""数据转换服务层。

服务层不关心具体格式解析逻辑，只负责从注册表中找到转换器并执行。

@FileName:   service.py
@Function:   转换服务入口 convert_data_to_yolo
"""

from __future__ import annotations

from pathlib import Path

from od_platform.data_pipeline.convert.registry import ConvertOptions, get_converter


def convert_data_to_yolo(
    input_dir: Path,
    output_labels_dir: Path,
    annotation_format: str,
    options: ConvertOptions,
) -> list[str]:
    """将指定标注格式转换为 YOLO label 目录。

    Args:
        input_dir: 输入标注路径，可以是目录或具体文件，取决于格式转换器。
        output_labels_dir: 输出 YOLO labels 目录。
        annotation_format: 输入标注格式名称。
        options: 转换选项。

    Returns:
        转换过程中使用的类别列表。

    Raises:
        ValueError: 转换器不存在或不支持指定任务时抛出。
    """
    entry = get_converter(annotation_format)
    if not entry.supports(options.task):
        raise ValueError(
            f"格式 {annotation_format!r} 不支持 task={options.task!r}。"
            f"支持: {entry.supported_tasks}"
        )
    return entry.func(input_dir, output_labels_dir, options)
