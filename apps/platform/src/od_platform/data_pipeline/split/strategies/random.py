"""随机数据集划分策略。

@FileName:   random.py
@Function:   按固定随机种子随机划分 train/val/test
"""

from __future__ import annotations

import random

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.manifest import PairList, SplitManifest
from od_platform.data_pipeline.split.registry import SplitOptions, register


def _validate_rates(options: SplitOptions) -> None:
    rates = (options.train_rate, options.val_rate, options.test_rate)
    if any(rate < 0 for rate in rates):
        raise ValueError(f"划分比例不能为负数: {rates}")
    if abs(sum(rates) - 1.0) > 1e-8:
        raise ValueError(f"划分比例之和必须为 1.0，实际为 {sum(rates):.6f}")


@register(SplitStrategy.RANDOM, description="按固定随机种子随机划分 train/val/test")
def split_random(pairs: PairList, options: SplitOptions) -> SplitManifest:
    """按固定随机种子随机划分样本。

    Args:
        pairs: ``(image_path, label_path)`` 样本对列表。
        options: 划分参数。

    Returns:
        可复现的划分结果。
    """
    _validate_rates(options)
    shuffled = list(pairs)
    random.Random(options.random_state).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * options.train_rate)
    val_count = int(total * options.val_rate)
    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]

    return SplitManifest(
        train=train,
        val=val,
        test=test,
        train_rate=options.train_rate,
        val_rate=options.val_rate,
        test_rate=options.test_rate,
        random_state=options.random_state,
        strategy=SplitStrategy.RANDOM,
    )
