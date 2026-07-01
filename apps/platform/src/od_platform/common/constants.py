"""ODPlatform 公共常量。

集中定义跨模块共享的任务类型、标注格式等稳定常量。

@FileName:   constants.py
@Function:   标注格式与任务类型常量
"""

from __future__ import annotations


class AnnotationFormat:
    """数据集标注格式常量。"""

    PASCAL_VOC = "pascal_voc"
    COCO = "coco"
    YOLO = "yolo"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """返回全部支持的标注格式名称。"""
        return cls.PASCAL_VOC, cls.COCO, cls.YOLO


class Task:
    """视觉任务类型常量。"""

    DETECT = "detect"
    SEGMENT = "segment"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """返回全部任务类型名称。"""
        return cls.DETECT, cls.SEGMENT


class SplitStrategy:
    """数据集划分策略名称常量。"""

    RANDOM = "random"
    STRATIFIED = "stratified"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """返回全部预期支持的划分策略名称。"""
        return cls.RANDOM, cls.STRATIFIED
