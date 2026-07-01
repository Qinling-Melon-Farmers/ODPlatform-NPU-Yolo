"""数据集划分策略注册表。

@FileName:   registry.py
@Function:   划分策略注册、查询与能力枚举
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from od_platform.common.registry_utils import import_modules_from_package
from od_platform.data_pipeline.split.manifest import PairList, SplitManifest

logger = logging.getLogger(__name__)


@dataclass
class SplitOptions:
    """划分策略共用参数包。"""

    train_rate: float = 0.8
    val_rate: float = 0.1
    test_rate: float = 0.1
    random_state: int = 1210


SplitFunc = Callable[[PairList, SplitOptions], SplitManifest]


@dataclass(frozen=True)
class SplitterEntry:
    """注册表中的划分策略条目。"""

    func: SplitFunc
    description: str


_REGISTRY: dict[str, SplitterEntry] = {}
_LAZY_INITIALIZED = False


def register(strategy_name: str, *, description: str) -> Callable[[SplitFunc], SplitFunc]:
    """注册划分策略函数。

    Args:
        strategy_name: 策略名称。
        description: 策略说明。

    Returns:
        装饰器函数。
    """

    def decorator(func: SplitFunc) -> SplitFunc:
        if strategy_name in _REGISTRY:
            logger.warning("划分策略 %s 被重复注册，后者将覆盖前者", strategy_name)
        _REGISTRY[strategy_name] = SplitterEntry(func=func, description=description)
        return func

    return decorator


def get_splitter(strategy_name: str) -> SplitterEntry:
    """按策略名获取划分策略条目。"""
    _lazy_init()
    if strategy_name not in _REGISTRY:
        raise ValueError(f"划分策略 {strategy_name!r} 未注册，已注册策略: {list(_REGISTRY)}")
    return _REGISTRY[strategy_name]


def list_strategies() -> dict[str, str]:
    """返回已注册划分策略及说明。"""
    _lazy_init()
    return {strategy_name: entry.description for strategy_name, entry in _REGISTRY.items()}


def _lazy_init() -> None:
    """延迟导入 strategies 包下所有策略模块，触发装饰器注册。"""
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return

    from od_platform.data_pipeline.split import strategies

    import_modules_from_package(strategies.__name__, strategies.__path__)
    _LAZY_INITIALIZED = True
