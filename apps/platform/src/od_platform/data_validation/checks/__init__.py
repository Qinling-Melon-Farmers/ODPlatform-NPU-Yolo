"""Built-in dataset validation checks."""

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
)


def _skip_if_yaml_invalid(ctx: CheckContext, name: str) -> CheckResult | None:
    """Return a skip result if the snapshot's yaml is invalid, otherwise None.

    Checks that depend on dataset paths (pair_existence, label_format,
    split_uniqueness) should call this first so they don't crash on bad yaml.
    """
    if ctx.snapshot is not None and ctx.snapshot.yaml_load_error is None:
        return None
    return CheckResult(
        name=name,
        severity=CheckSeverity.INFO,
        summary="yaml_schema 未通过，跳过依赖数据集路径的检查",
        details={"reason": "skip_due_to_yaml_error"},
    )
