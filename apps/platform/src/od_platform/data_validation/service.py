"""Dataset validation service layer."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from od_platform.common import paths
from od_platform.common.performance_utils import time_it
from od_platform.data_validation.registry import (
    CheckContext,
    CheckEntry,
    CheckResult,
    CheckSeverity,
    get_all_checks,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Aggregated validation report."""

    yaml_path: Path
    results: list[CheckResult]
    report_path: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def max_severity(self) -> str:
        if not self.results:
            return CheckSeverity.INFO
        return max(self.results, key=lambda item: CheckSeverity.rank(item.severity)).severity

    @property
    def exit_code(self) -> int:
        if any(result.severity == CheckSeverity.ERROR for result in self.results):
            return 2
        if any(result.severity == CheckSeverity.WARNING for result in self.results):
            return 1
        return 0

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.severity] = counts.get(result.severity, 0) + 1
        return counts


def validate_dataset(
    *,
    yaml_path: Path,
    task_type: str = "detect",
    write_report: bool = True,
) -> ValidationReport:
    """Validate one generated dataset yaml."""
    ctx = _build_context(yaml_path=yaml_path, task=task_type)
    results = run_all_checks(ctx)
    report = ValidationReport(yaml_path=yaml_path, results=results)
    if write_report:
        report.report_path = write_report_json(report)
    return report


@time_it(name="所有检测耗时总计", logger_instance=logger, iterations=1)
def run_all_checks(ctx: CheckContext) -> list[CheckResult]:
    """Run every registered check and collect all results."""
    entries = get_all_checks()
    logger.info("开始执行 %d 个 checks", len(entries))
    results: list[CheckResult] = []
    for entry in entries:
        result = _safe_run_one(entry, ctx)
        _log_check_result(result)
        results.append(result)
    _log_summary(results)
    return results


@time_it(name=lambda entry, _ctx: f"检查: {entry.name}", logger_instance=logger, iterations=1)
def _safe_run_one(entry: CheckEntry, ctx: CheckContext) -> CheckResult:
    """Run one check and convert unexpected exceptions into ERROR results."""
    try:
        return entry.func(ctx)
    except Exception as exc:
        logger.exception("check %s 出现异常，已捕获为 ERROR 级结果", entry.name)
        return CheckResult(
            name=entry.name,
            severity=CheckSeverity.ERROR,
            summary=f"check {entry.name} 出现异常: {type(exc).__name__}: {exc}",
            details={"exception_type": type(exc).__name__, "exception_msg": str(exc)},
        )


def write_report_json(report: ValidationReport) -> Path:
    """Write a machine-readable report and return its path."""
    output_dir = paths.RUNS_DIR / "data_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{report.yaml_path.stem}-validation.json"
    payload = {
        "yaml_path": str(report.yaml_path),
        "max_severity": report.max_severity,
        "exit_code": report.exit_code,
        "summary": report.summary,
        "results": [
            {
                "name": result.name,
                "severity": result.severity,
                "summary": result.summary,
                "details": result.details,
            }
            for result in report.results
        ],
        "fix_items": [
            {
                "name": result.name,
                "severity": result.severity,
                "summary": result.summary,
                "action": result.details.get("action", "人工复核"),
            }
            for result in report.results
            if result.severity in (CheckSeverity.WARNING, CheckSeverity.ERROR)
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def render_to_logger(
    report: ValidationReport,
    target_logger: logging.Logger,
    *,
    report_path: Path | None = None,
) -> None:
    """Render a validation report to logs."""
    target_logger.info("数据质检完成: %s", report.yaml_path)
    target_logger.info("结果统计: %s, exit_code=%d", report.summary, report.exit_code)
    for result in report.results:
        _log_check_result(result, target_logger)
    if report_path is not None:
        target_logger.info("质检报告: %s", report_path)


def _build_context(*, yaml_path: Path, task: str) -> CheckContext:
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset yaml 不存在: {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"dataset yaml 内容必须是映射: {yaml_path}")
    dataset_root_value = config.get("path")
    if not dataset_root_value:
        raise ValueError("dataset yaml 缺少 path 字段")
    dataset_root = Path(str(dataset_root_value))
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()
    return CheckContext(
        yaml_path=yaml_path,
        config=config,
        dataset_root=dataset_root,
        task=task,
    )


def _log_check_result(result: CheckResult, target_logger: logging.Logger = logger) -> None:
    log_method = {
        CheckSeverity.ERROR: target_logger.error,
        CheckSeverity.WARNING: target_logger.warning,
        CheckSeverity.INFO: target_logger.info,
        CheckSeverity.PASS: target_logger.debug,
    }.get(result.severity, target_logger.info)
    log_method("[%s] %s: %s", result.severity, result.name, result.summary)


def _log_summary(results: list[CheckResult]) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.severity] = counts.get(result.severity, 0) + 1
    parts = [f"{severity}={count}" for severity, count in sorted(counts.items())]
    logger.info("检查完成，结果如下: %s", " / ".join(parts))
