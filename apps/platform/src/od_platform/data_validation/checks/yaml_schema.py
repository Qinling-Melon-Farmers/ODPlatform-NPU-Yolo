"""Check dataset yaml schema and field consistency."""

from __future__ import annotations

from typing import Any

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("yaml_schema")
def validate_yaml_schema(ctx: CheckContext) -> CheckResult:
    """Validate dataset yaml existence, parseability and class fields."""
    snapshot = ctx.snapshot
    if snapshot is None:
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary="snapshot is not initialized",
            details={"reason": "snapshot_missing", "problems": ["snapshot is not initialized"]},
        )

    if snapshot.yaml_load_error is not None:
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary=snapshot.yaml_load_error,
            details={
                "reason": "yaml_load_error",
                "yaml_path": str(snapshot.yaml_path),
                "problems": [snapshot.yaml_load_error],
            },
        )

    cfg = snapshot.yaml_data
    problems: list[str] = []

    nc = cfg.get("nc")
    if not isinstance(nc, int) or nc <= 0:
        problems.append(f"nc 字段不存在或者不是正整数: {nc}")
        nc = None

    names_count, names_problem = _validate_names(cfg.get("names"))
    if names_problem:
        problems.append(names_problem)

    if nc is not None and names_count is not None and nc != names_count:
        problems.append(f"nc 和 names 长度不一致, nc: {nc}, names: {names_count}")

    # 检查类别名是否有重复 (同名会导致报告与训练统计混淆)
    dupes = _check_duplicate_names(cfg.get("names"))
    if dupes:
        problems.append(f"names 中存在重复类别名: {', '.join(dupes)}")

    if problems:
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary=f"yaml 字段不一致: 共有 {len(problems)} 个问题",
            details={
                "reason": "field_inconsistency",
                "problems": problems,
                "nc": nc,
                "names_count": names_count,
                "action": "修正 dataset.yaml 中的 nc 和 names 字段",
            },
        )

    return CheckResult(
        name="yaml_schema",
        severity=CheckSeverity.INFO,
        summary=f"yaml 字段一致 (nc={nc}, names_count={names_count})",
        details={"nc": nc, "names_count": names_count},
    )


def _validate_names(names_raw: Any) -> tuple[int | None, str]:
    if isinstance(names_raw, list):
        if not names_raw:
            return None, "names 缺失或者是空列表"
        if not all(isinstance(name, str) and name for name in names_raw):
            return None, "names 列表中包含非空字符串以外的元素"
        return len(names_raw), ""

    if isinstance(names_raw, dict):
        if not names_raw:
            return None, "names 缺失或者是空字典"
        try:
            keys = [int(key) for key in names_raw]
        except (TypeError, ValueError):
            return None, "names 字典的键必须能转换为整数"
        if keys != sorted(keys):
            return None, "names 字典的键必须按类别 ID 顺序排列"
        if not all(isinstance(name, str) and name for name in names_raw.values()):
            return None, "names 字典的值必须是非空字符串"
        return len(names_raw), ""

    return None, f"names 缺失或者不是合法的列表/字典: {type(names_raw).__name__}"


def _check_duplicate_names(names_raw: Any) -> list[str]:
    """Return list of duplicate class names, empty if all unique."""
    if isinstance(names_raw, list):
        seen: set[str] = set()
        dupes: list[str] = []
        for name in names_raw:
            if not isinstance(name, str):
                continue
            if name in seen:
                dupes.append(name)
            else:
                seen.add(name)
        return dupes

    if isinstance(names_raw, dict):
        seen: set[str] = set()
        dupes: list[str] = []
        for value in names_raw.values():
            if not isinstance(value, str):
                continue
            if value in seen:
                dupes.append(value)
            else:
                seen.add(value)
        return dupes

    return []
