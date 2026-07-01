"""数据集划分结果载体。

@FileName:   manifest.py
@Function:   保存 train/val/test 三组样本与可复现元数据
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from od_platform.common.constants import SplitStrategy

Pair = tuple[Path, Path]
PairList = list[Pair]


@dataclass
class SplitManifest:
    """数据集划分结果与复现参数。"""

    train: PairList = field(default_factory=list)
    val: PairList = field(default_factory=list)
    test: PairList = field(default_factory=list)
    train_rate: float = 0.8
    val_rate: float = 0.1
    test_rate: float = 0.1
    random_state: int = 1210
    strategy: str = SplitStrategy.RANDOM

    def summary(self) -> dict[str, int]:
        """返回三组样本数量摘要。"""
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "total": len(self.train) + len(self.val) + len(self.test),
        }

    def all_pairs(self) -> PairList:
        """按 train、val、test 顺序返回全部样本。"""
        return [*self.train, *self.val, *self.test]
