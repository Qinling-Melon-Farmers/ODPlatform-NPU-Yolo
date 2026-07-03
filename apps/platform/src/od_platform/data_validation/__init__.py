"""Dataset validation subsystem."""

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckEntry,
    CheckResult,
    CheckSeverity,
    check,
    get_all_checks,
    get_check,
    list_check_names,
)
from od_platform.data_validation.report import ValidationReport
from od_platform.data_validation.service import run_all_checks, validate_dataset

__all__ = [
    "CheckContext",
    "CheckEntry",
    "CheckResult",
    "CheckSeverity",
    "ValidationReport",
    "check",
    "get_all_checks",
    "get_check",
    "list_check_names",
    "run_all_checks",
    "validate_dataset",
]
