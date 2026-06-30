"""标注转换器注册表。

使用装饰器注册不同标注格式的转换器，服务层按格式名查询并调度。

@FileName:   registry.py
@Function:   转换器注册、查询与能力枚举
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConvertOptions:
    """所有转换器共用的参数包。"""

    task: str = "detect"
    classes: list[str] | None = field(default=None)
    coco_cls91to80: bool = False


ConverterFunc = Callable[[Path, Path, ConvertOptions], list[str]]


@dataclass(frozen=True)
class ConverterEntry:
    """注册表中的转换器条目。"""

    func: ConverterFunc
    supported_tasks: tuple[str, ...]

    def supports(self, task: str) -> bool:
        """判断转换器是否支持指定任务。"""
        return task in self.supported_tasks


_REGISTRY: dict[str, ConverterEntry] = {}
_LAZY_INITIALIZED = False


def register(
    format_name: str,
    *,
    supported_tasks: tuple[str, ...],
) -> Callable[[ConverterFunc], ConverterFunc]:
    """注册转换器函数。

    Args:
        format_name: 标注格式名称。
        supported_tasks: 转换器支持的任务列表。

    Returns:
        装饰器函数。
    """

    def decorator(func: ConverterFunc) -> ConverterFunc:
        if format_name in _REGISTRY:
            logger.warning("格式 %s 被重复注册，后者将覆盖前者", format_name)
        _REGISTRY[format_name] = ConverterEntry(
            func=func,
            supported_tasks=tuple(supported_tasks),
        )
        logger.debug("注册 converter: format=%s, tasks=%s", format_name, supported_tasks)
        return func

    return decorator


def get_converter(format_name: str) -> ConverterEntry:
    """按格式名获取转换器条目。

    Args:
        format_name: 标注格式名称。

    Returns:
        注册表中的转换器条目。

    Raises:
        ValueError: 格式未注册时抛出。
    """
    _lazy_init()
    if format_name not in _REGISTRY:
        raise ValueError(f"格式 {format_name!r} 未注册，已注册格式: {list(_REGISTRY)}")
    return _REGISTRY[format_name]


def list_capabilities() -> dict[str, tuple[str, ...]]:
    """返回当前已注册格式与支持任务。"""
    _lazy_init()
    return {format_name: entry.supported_tasks for format_name, entry in _REGISTRY.items()}


def _lazy_init() -> None:
    """延迟导入 converters 包下所有转换器模块，触发装饰器注册。"""
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    _LAZY_INITIALIZED = True

    from od_platform.data_pipeline.convert import converters

    for module_info in pkgutil.iter_modules(converters.__path__):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{converters.__name__}.{module_info.name}")
