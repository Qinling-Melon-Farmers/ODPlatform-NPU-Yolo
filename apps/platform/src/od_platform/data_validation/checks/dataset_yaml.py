"""Checks for dataset yaml structure."""

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("yaml_required_fields")
def check_yaml_required_fields(ctx: CheckContext) -> CheckResult:
    """Validate required dataset yaml fields."""
    required = ("path", "train", "val", "names", "nc")
    missing = [field for field in required if field not in ctx.config]
    if missing:
        return CheckResult(
            name="yaml_required_fields",
            severity=CheckSeverity.ERROR,
            summary=f"dataset yaml 缺少必填字段: {', '.join(missing)}",
            details={"missing": missing, "action": "补齐 dataset.yaml 必填字段"},
        )

    classes = ctx.classes
    nc = ctx.config.get("nc")
    if not isinstance(nc, int) or nc != len(classes):
        return CheckResult(
            name="yaml_required_fields",
            severity=CheckSeverity.ERROR,
            summary=f"nc 与 names 数量不一致: nc={nc}, names={len(classes)}",
            details={"nc": nc, "names_count": len(classes), "action": "同步 nc 与 names"},
        )

    return CheckResult(
        name="yaml_required_fields",
        severity=CheckSeverity.PASS,
        summary="dataset yaml 必填字段完整",
        details={"class_count": len(classes)},
    )
