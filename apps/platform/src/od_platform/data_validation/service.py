"""Dataset validation service layer."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from od_platform.common import paths
from od_platform.common.performance_utils import time_it
from od_platform.common.system_utils import log_device_info
from od_platform.data_validation.export import write_rework_csv
from od_platform.data_validation.registry import (
    CheckContext,
    CheckEntry,
    CheckResult,
    CheckSeverity,
    get_all_checks,
)
from od_platform.data_validation.render import (
    render_to_html,
    render_to_markdown,
)
from od_platform.data_validation.report import ValidationReport
from od_platform.data_validation.snapshot import build_snapshot

logger = logging.getLogger(__name__)


def validate_dataset(
    *,
    yaml_path: Path,
    task_type: str = "detect",
    write_report: bool = True,
    run_id: str | None = None,
    executor: str | None = None,
) -> ValidationReport:
    """端到端验证: 构造 snapshot → 跑 check → 包装 report → 可选写盘。

    Args:
        yaml_path:    数据集 yaml 文件路径
        task_type:    'detect' / 'segment'
        write_report: 是否写 JSON 报告到 run_dir/report.json
        run_id:       手动指定运行 ID; None 表示自动用时间戳
        executor:     执行人姓名, 记录到报告审计字段
    """
    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = paths.validation_run_dir(resolved_run_id) if write_report else None

    if write_report and run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    started_iso = datetime.now(timezone.utc).isoformat()

    device_info = log_device_info(logger)
    if executor:
        logger.info("执行人: %s", executor)

    snapshot = build_snapshot(yaml_path=yaml_path, task_type=task_type)
    ctx = CheckContext(yaml_path=yaml_path, snapshot=snapshot)
    results = run_all_checks(ctx)

    duration = time.perf_counter() - t0

    report = ValidationReport(
        run_id=resolved_run_id,
        yaml_path=yaml_path,
        snapshot=snapshot,
        results=results,
        duration_seconds=duration,
        started_at_iso=started_iso,
        run_dir=run_dir,
        executor=executor,
        device_info=device_info,
    )

    if write_report and run_dir is not None:
        report_path = run_dir / "report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("JSON 报告已写入: %s", report_path)

        md_path = run_dir / "report.md"
        md_path.write_text(render_to_markdown(report, executor=executor), encoding="utf-8")
        logger.info("Markdown 报告已写入: %s", md_path)

        try:
            html_path = run_dir / "report.html"
            html_path.write_text(render_to_html(report, executor=executor), encoding="utf-8")
            logger.info("HTML 报告已写入: %s", html_path)
        except Exception as exc:
            logger.warning("HTML 报告生成失败 (不阻断验证): %s", exc)

        try:
            csv_path = run_dir / "rework.csv"
            write_rework_csv(report, csv_path)
        except Exception as exc:
            logger.warning("CSV 返工清单生成失败 (不阻断验证): %s", exc)

    return report


@time_it(name="所有检查耗时总计", logger_instance=logger, iterations=1)
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
    """Write a machine-readable report using report.to_dict()."""
    run_dir = report.run_dir
    if run_dir is None:
        run_dir = paths.validation_run_dir(report.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "report.json"
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


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
