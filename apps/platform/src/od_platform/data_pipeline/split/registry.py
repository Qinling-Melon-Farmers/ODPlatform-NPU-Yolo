"""Dataset split strategy registry."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from od_platform.common.constants import DEFAULT_RANDOM_STATE
from od_platform.common.registry_utils import import_submodules
from od_platform.data_pipeline.split.manifest import PairList, SplitManifest

logger = logging.getLogger(__name__)


@dataclass
class SplitOptions:
    """Shared parameters for split strategies."""

    train_rate: float = 0.8
    val_rate: float = 0.1
    test_rate: float | None = None
    random_state: int = DEFAULT_RANDOM_STATE
    labels_per_image: dict[str, list[str]] | None = field(default=None)
    group_per_image: dict[str, str] | None = field(default=None)


SplitFunc = Callable[[PairList, SplitOptions], SplitManifest]


@dataclass(frozen=True)
class SplitterEntry:
    """One registered split strategy."""

    func: SplitFunc
    description: str
    requires_labels: bool = False


_REGISTRY: dict[str, SplitterEntry] = {}
_LAZY_INITIALIZED = False


def register(
    strategy_name: str,
    *,
    description: str,
    requires_labels: bool = False,
) -> Callable[[SplitFunc], SplitFunc]:
    """Register a split strategy function."""

    def decorator(func: SplitFunc) -> SplitFunc:
        if strategy_name in _REGISTRY:
            logger.warning("Split strategy %s is already registered and will be overwritten", strategy_name)
        _REGISTRY[strategy_name] = SplitterEntry(
            func=func,
            description=description,
            requires_labels=requires_labels,
        )
        return func

    return decorator


def get_splitter(strategy_name: str) -> SplitterEntry:
    """Return one registered split strategy by name."""
    _lazy_init()
    if strategy_name not in _REGISTRY:
        raise ValueError(f"未注册的划分策略: {strategy_name!r}, 已注册: {list(_REGISTRY)}")
    return _REGISTRY[strategy_name]


def list_strategies() -> dict[str, str]:
    """Return registered strategy names and descriptions."""
    _lazy_init()
    return {strategy_name: entry.description for strategy_name, entry in _REGISTRY.items()}


def _lazy_init() -> None:
    """Import strategy modules lazily so decorators can populate the registry."""
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return

    from od_platform.data_pipeline.split import strategies

    import_submodules(strategies)
    _LAZY_INITIALIZED = True
