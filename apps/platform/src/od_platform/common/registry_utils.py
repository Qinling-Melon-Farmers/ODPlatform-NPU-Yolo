"""注册表自动发现辅助工具。

@FileName:   registry_utils.py
@Function:   扫描包内模块并触发注册装饰器
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from pathlib import Path


def import_modules_from_package(package_name: str, package_paths: Iterable[str | Path]) -> None:
    """导入包路径下所有非私有子模块。

    Args:
        package_name: 包名，例如 ``od_platform.data_pipeline.convert.converters``。
        package_paths: 包的 ``__path__``。

    Returns:
        None。
    """
    for module_info in pkgutil.iter_modules([str(path) for path in package_paths]):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{package_name}.{module_info.name}")
