"""ODPlatform 公共常量。

集中定义跨模块共享的任务类型、标注格式、划分策略等稳定常量。

@FileName:   constants.py
@Function:   标注格式、任务类型、划分策略常量
"""

from __future__ import annotations

# 浮点容差：吸收浮点舍入误差（1.0 - 0.7 - 0.3 在浮点里不是 0，而是 5.55e-17）。
# 校验 test_rate >= 0 时用它吸收这种误差，否则正常的 70/30 切分会被误判成"比例越界"。
RATE_EPSILON: float = 1e-6


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
    """数据集划分策略名称常量。

    各策略含义：
      - RANDOM: 纯随机，对稀有类不公平。
      - STRATIFIED: 主类别分层，稀有类不至于整体消失。
      - STRATIFIED_MULTILABEL: 多标签分层，连次要稀有类也照顾。
    """

    RANDOM = "random"
    STRATIFIED = "stratified"
    STRATIFIED_MULTILABEL = "stratified_multilabel"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """返回全部预期支持的划分策略名称。"""
        return cls.RANDOM, cls.STRATIFIED, cls.STRATIFIED_MULTILABEL
