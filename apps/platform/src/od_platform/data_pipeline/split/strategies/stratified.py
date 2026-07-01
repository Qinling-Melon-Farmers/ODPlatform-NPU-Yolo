"""Label-aware split strategies."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.manifest import Pair, PairList, SplitManifest
from od_platform.data_pipeline.split.registry import SplitOptions, register
from od_platform.data_pipeline.split.strategies._common import (
    seeded_shuffle,
    three_way_counts,
    validate_rates,
)

_BACKGROUND = "__background__"


def _labels_for(pair: Pair, options: SplitOptions) -> list[str]:
    if options.labels_per_image is None:
        return []
    return options.labels_per_image.get(pair[0].stem, [])


def _primary_label(pair: Pair, options: SplitOptions) -> str:
    if options.group_per_image and pair[0].stem in options.group_per_image:
        return options.group_per_image[pair[0].stem]
    labels = sorted(set(_labels_for(pair, options)))
    return labels[0] if labels else _BACKGROUND


def _multilabel_key(pair: Pair, options: SplitOptions) -> str:
    labels = sorted(set(_labels_for(pair, options)))
    return "|".join(labels) if labels else _BACKGROUND


def _split_by_key(
    pairs: PairList,
    options: SplitOptions,
    *,
    strategy: str,
    key_func: Callable[[Pair, SplitOptions], str],
) -> SplitManifest:
    test_rate = validate_rates(options.train_rate, options.val_rate, options.test_rate)
    rng = random.Random(options.random_state)

    groups: dict[str, PairList] = defaultdict(list)
    for pair in sorted(pairs, key=lambda item: item[0].stem):
        groups[key_func(pair, options)].append(pair)

    train: PairList = []
    val: PairList = []
    test: PairList = []
    for key in sorted(groups):
        group = seeded_shuffle(groups[key], rng)
        n_train, n_val, _ = three_way_counts(len(group), options.train_rate, options.val_rate)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    train = seeded_shuffle(train, rng)
    val = seeded_shuffle(val, rng)
    test = seeded_shuffle(test, rng)

    return SplitManifest(
        train=train,
        val=val,
        test=test,
        train_rate=options.train_rate,
        val_rate=options.val_rate,
        test_rate=test_rate,
        random_state=options.random_state,
        strategy=strategy,
    )


@register(
    SplitStrategy.STRATIFIED,
    description="按主类别分层划分 train/val/test",
    requires_labels=True,
)
def split_stratified(pairs: PairList, options: SplitOptions) -> SplitManifest:
    """Split by each image's primary class label."""
    return _split_by_key(
        pairs,
        options,
        strategy=SplitStrategy.STRATIFIED,
        key_func=_primary_label,
    )


@register(
    SplitStrategy.STRATIFIED_MULTILABEL,
    description="按多标签组合分层划分 train/val/test",
    requires_labels=True,
)
def split_stratified_multilabel(pairs: PairList, options: SplitOptions) -> SplitManifest:
    """Split by each image's full label set."""
    return _split_by_key(
        pairs,
        options,
        strategy=SplitStrategy.STRATIFIED_MULTILABEL,
        key_func=_multilabel_key,
    )
