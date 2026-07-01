"""分层数据集划分策略。

@FileName:   stratified.py
@Function:   主类别分层划分 + 多标签分层划分（骨架，后继阶段实现）
"""

from __future__ import annotations

import logging

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.manifest import PairList, SplitManifest
from od_platform.data_pipeline.split.registry import SplitOptions, register

logger = logging.getLogger(__name__)


@register(
    SplitStrategy.STRATIFIED,
    description="按主类别分层：保持各 split 中每类占比与原始一致，稀有类不至于整体消失",
)
def split_stratified(pairs: PairList, options: SplitOptions) -> SplitManifest:
    """按主类别分层划分样本。

    保证 train/val/test 各子集中每类的占比与全体一致。
    适用于单标签数据集，常见于分类和目标检测场景。
    此时类别数可能不多但某个类样本极少——分层确保它不会在划分中被整类淹没。

    Args:
        pairs: ``(image_path, label_path)`` 样本对列表。
        options: 划分参数。

    Returns:
        可复现的划分结果。

    Raises:
        NotImplementedError: 本策略尚未实现。
    """
    raise NotImplementedError(
        "分层划分策略 (stratified) 尚未实现，将在后续阶段完成。"
        "预期行为：读取每个 label 的主类别 → 按类别频率均衡分配 → 保证各 split 中每类占比一致。"
    )


@register(
    SplitStrategy.STRATIFIED_MULTILABEL,
    description="多标签分层：对每个次级稀有类也保持分布，适合多标签场景",
)
def split_stratified_multilabel(pairs: PairList, options: SplitOptions) -> SplitManifest:
    """按多标签层级分层划分样本。

    适用于多标签数据集（一图多框、多类别）。
    与主类别分层不同：它会考虑**所有**出现在每张图里的类别，
    连次要稀有类也纳入均衡策略。

    Args:
        pairs: ``(image_path, label_path)`` 样本对列表。
        options: 划分参数。

    Returns:
        可复现的划分结果。

    Raises:
        NotImplementedError: 本策略尚未实现。
    """
    raise NotImplementedError(
        "多标签分层划分策略 (stratified_multilabel) 尚未实现，将在后续阶段完成。"
        "预期行为：读取每个 label 的全部类别 → 对多标签样本按稀有度权重分配 → 次要稀有类也不丢失。"
    )
