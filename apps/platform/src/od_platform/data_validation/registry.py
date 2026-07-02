"""Registry and shared contracts for dataset validation checks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from od_platform.common.registry_utils import import_submodules
from od_platform.data_validation.snapshot import DatasetSnapshot, build_snapshot

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
    """Input and cached state shared by all validation checks."""

    yaml_path: Path
    task: str = "detect"
    snapshot: DatasetSnapshot | None = None

    def __post_init__(self) -> None:
        self.yaml_path = self.yaml_path.resolve()
        if self.snapshot is None:
            self.snapshot = build_snapshot(self.yaml_path, task_type=self.task)

    @property
    def config(self) -> dict[str, Any]:
        return self.snapshot.yaml_data if self.snapshot is not None else {}

    @property
    def yaml_error(self) -> str | None:
        return self.snapshot.yaml_load_error if self.snapshot is not None else "snapshot is not initialized"

    @property
    def has_valid_yaml(self) -> bool:
        return self.yaml_error is None and bool(self.config)

    @property
    def dataset_root(self) -> Path:
        return self.snapshot.data_root if self.snapshot is not None else self.yaml_path.parent

    @property
    def classes(self) -> list[str]:
        return list(self.snapshot.class_names) if self.snapshot is not None else []


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
            raise ValueError(f"check {name!r} is already registered by {func.__module__}.{func.__name__}")
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
        raise ValueError(f"check {name!r} is not registered; known checks: {list(_REGISTRY)}")
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
