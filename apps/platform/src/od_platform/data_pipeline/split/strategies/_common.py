"""Shared helpers for split strategies."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

from od_platform.common.constants import RATE_EPSILON

T = TypeVar("T")


def validate_rates(train_rate: float, val_rate: float, test_rate: float | None = None) -> float:
    """Validate train/val/test rates and return the computed test rate."""
    resolved_test_rate = 1.0 - train_rate - val_rate if test_rate is None else test_rate
    total = train_rate + val_rate + resolved_test_rate

    if not (0 <= train_rate <= 1 and 0 <= val_rate <= 1 and -RATE_EPSILON <= resolved_test_rate <= 1):
        raise ValueError(
            f"比例越界: train={train_rate}, val={val_rate}, test={resolved_test_rate}"
        )
    if abs(total - 1.0) > RATE_EPSILON:
        raise ValueError(f"划分比例之和必须为 1.0，实际为 {total:.6f}")
    if abs(resolved_test_rate) <= RATE_EPSILON:
        return 0.0
    return max(0.0, resolved_test_rate)


def three_way_counts(n: int, train_rate: float, val_rate: float) -> tuple[int, int, int]:
    """Return stable train/val/test counts for a list length."""
    n_train = int(round(n * train_rate))
    n_val = int(round(n * val_rate))
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    return n_train, n_val, n - n_train - n_val


def seeded_shuffle(seq: Sequence[T], rng: random.Random) -> list[T]:
    """Return a shuffled copy using the provided random generator."""
    out = list(seq)
    rng.shuffle(out)
    return out
