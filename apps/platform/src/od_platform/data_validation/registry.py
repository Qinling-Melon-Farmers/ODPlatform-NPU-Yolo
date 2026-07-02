"""Registry and shared contracts for dataset validation checks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from od_platform.common.registry_utils import import_submodules

logger = logging.getLogger(__name__)


class CheckSeverity:
    """Validation severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    PASS = "PASS"

    _ORDER = {INFO: 1, WARNING: 2, ERROR: 3, PASS: 0}

    @classmethod
    def rank(cls, level: str) -> int:
        return cls._ORDER.get(level, 0)


@dataclass
class CheckResult:
    """Unified result returned by one validation check."""

    name: str
    severity: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.severity in (CheckSeverity.PASS, CheckSeverity.INFO)


@dataclass
class CheckContext:
    """Input shared by all validation checks."""

    yaml_path: Path
    config: dict[str, Any]
    dataset_root: Path
    task: str = "detect"

    def split_image_dir(self, split: str) -> Path | None:
        value = self.config.get(split)
        if value is None:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else self.dataset_root / path

    def split_label_dir(self, split: str) -> Path | None:
        image_dir = self.split_image_dir(split)
        if image_dir is None:
            return None
        if image_dir.name == "images":
            return image_dir.parent / "labels"
        return image_dir.parent / "labels"

    @property
    def classes(self) -> list[str]:
        names = self.config.get("names", [])
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
        if isinstance(names, list):
            return [str(item) for item in names]
        return []


CheckFunc = Callable[[CheckContext], CheckResult]


@dataclass(frozen=True)
class CheckEntry:
    """One immutable registry entry."""

    name: str
    func: CheckFunc


_REGISTRY: dict[str, CheckEntry] = {}
_LAZY_INITIALIZED = False


def check(name: str) -> Callable[[CheckFunc], CheckFunc]:
    """Register one validation check."""

    def decorator(func: CheckFunc) -> CheckFunc:
        if name in _REGISTRY:
            raise ValueError(f"check {name!r} 重复注册，第二次出现在 {func.__module__}.{func.__name__}")
        _REGISTRY[name] = CheckEntry(name=name, func=func)
        return func

    return decorator


def get_all_checks() -> list[CheckEntry]:
    """Return all registered checks."""
    _lazy_init()
    return list(_REGISTRY.values())


def get_check(name: str) -> CheckEntry:
    """Return one registered check by name."""
    _lazy_init()
    if name not in _REGISTRY:
        raise ValueError(f"check {name!r} 未注册，已经注册的检查有: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_check_names() -> list[str]:
    """Return all registered check names."""
    _lazy_init()
    return list(_REGISTRY)


def _lazy_init() -> None:
    """Import checks lazily so decorators can populate the registry."""
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return

    from od_platform.data_validation import checks

    import_submodules(checks)
    _LAZY_INITIALIZED = True
