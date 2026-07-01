"""Random train/val/test split strategy."""

from __future__ import annotations

import random

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.manifest import PairList, SplitManifest
from od_platform.data_pipeline.split.registry import SplitOptions, register
from od_platform.data_pipeline.split.strategies._common import (
    seeded_shuffle,
    three_way_counts,
    validate_rates,
)


@register(SplitStrategy.RANDOM, description="随机划分 train/val/test")
def split_random(pairs: PairList, options: SplitOptions) -> SplitManifest:
    """Split samples randomly with a fixed seed."""
    test_rate = validate_rates(options.train_rate, options.val_rate, options.test_rate)
    rng = random.Random(options.random_state)
    shuffled = seeded_shuffle(sorted(pairs, key=lambda pair: pair[0].stem), rng)

    n_train, n_val, _ = three_way_counts(len(shuffled), options.train_rate, options.val_rate)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return SplitManifest(
        train=train,
        val=val,
        test=test,
        train_rate=options.train_rate,
        val_rate=options.val_rate,
        test_rate=test_rate,
        random_state=options.random_state,
        strategy=SplitStrategy.RANDOM,
    )
