"""render.py — D4 报告的展示层 (纯展示, 不动数据)。

公开 API:
    render_to_logger(report, logger, report_path=None)
    render_to_markdown(report, executor=None) -> str
    render_to_html(report, executor=None) -> str

输出结构 (五段式):
    1. 报告头 (run_id / yaml_path / task / 耗时 / 执行人)
    2. 数据集摘要 (类别 / 各 split 的图像数/标注数/实例数)
    3. 检查项一览 (每个 check 一行)
    4. 失败详情 (仅当有非 PASS check 时出现, 按 check 类型展开 details)
    5. 报告尾 (JSON 报告路径 + 建议)
"""

from __future__ import annotations

import logging
from pathlib import Path

from od_platform.data_validation.registry import CheckResult

H1_LINE = "=" * 72
H2_LINE = "-" * 72


# ============================================================
# render_to_logger — 五段式终端报告
# ============================================================


def render_to_logger(
    report: object,
    target_logger: logging.Logger,
    *,
    report_path: Path | None = None,
) -> None:
    """把报告渲染成五段式日志输出。兼容新旧 ValidationReport。"""
    # 兼容新旧 report — 旧版无 snapshot 属性
    snapshot = getattr(report, "snapshot", None)
    if snapshot is not None:
        _render_header(report, target_logger)
        _render_dataset_summary(snapshot, target_logger)
        _render_check_overview(report.results, target_logger)
        failed = getattr(report, "failed_results", [r for r in report.results if not r.passed])
        if failed:
            _render_failure_details(failed, target_logger)
        _render_footer(report_path or getattr(report, "report_path", None), target_logger)
    else:
        _render_legacy(report, target_logger, report_path)


def _render_legacy(report: object, target_logger: logging.Logger, report_path: Path | None) -> None:
    """旧版报告简单渲染 (向后兼容)。"""
    target_logger.info("数据质检完成: %s", report.yaml_path)
    counts = getattr(report, "summary", getattr(report, "counts_by_severity", None))
    if callable(counts):
        counts = counts()
    if counts:
        target_logger.info("结果统计: %s, exit_code=%d", counts, report.exit_code)
    for r in report.results:
        sev = r.severity
        log_method = {
            "ERROR": target_logger.error,
            "WARNING": target_logger.warning,
            "INFO": target_logger.info,
            "PASS": target_logger.debug,
        }.get(sev, target_logger.info)
        log_method("[%s] %s: %s", sev, r.name, r.summary)
    if report_path:
        target_logger.info("质检报告: %s", report_path)


# ============================================================
# 段 1: 报告头
# ============================================================


def _render_header(report: object, logger: logging.Logger) -> None:
    logger.info(H1_LINE)
    logger.info("                       YOLO 数据集验证报告")
    logger.info(H1_LINE)
    logger.info("  run_id:   %s", report.run_id)
    logger.info("  yaml:     %s", report.yaml_path)
    executor = getattr(report, "executor", None)
    executor_str = f"  执行人:   {executor}" if executor else "  执行人:   (未指定)"
    logger.info(executor_str)
    logger.info(
        "  task:     %-8s  耗时: %.2fs  severity: %s",
        report.snapshot.task_type,
        report.duration_seconds,
        report.overall_severity,
    )


# ============================================================
# 段 2: 数据集摘要
# ============================================================


def _render_dataset_summary(snapshot: object, logger: logging.Logger) -> None:
    logger.info(H2_LINE)
    logger.info("  ▸ 数据集摘要")

    if snapshot.class_names:
        names_str = ", ".join(snapshot.class_names)
        logger.info("    类别:  %s  (nc=%s)", names_str, snapshot.nc)
    else:
        logger.info("    类别:  (未取到 — yaml_schema 应已报错)")

    if not snapshot.stats_per_split:
        logger.info("    (无任何 split 可统计)")
        return

    for split, stat in snapshot.stats_per_split.items():
        logger.info(
            "    %-6s: %6d 张  /  %6d 标注  /  %6d 实例",
            split,
            stat.image_count,
            stat.annotated_count,
            stat.total_instances,
        )


# ============================================================
# 段 3: 检查项一览
# ============================================================


def _render_check_overview(results: list, logger: logging.Logger) -> None:
    logger.info(H2_LINE)
    logger.info("  ▸ 检查项一览")
    for r in results:
        logger.info("    [%7s]  %-18s  %s", r.severity, r.name, r.summary)


# ============================================================
# 段 4: 失败详情 (well-known keys 模式)
# ============================================================


def _render_failure_details(failed: list, logger: logging.Logger) -> None:
    logger.info(H2_LINE)
    logger.info("  ▸ 失败详情")
    for r in failed:
        _render_one_check_details(r, logger)


def _render_one_check_details(r: CheckResult, logger: logging.Logger) -> None:
    """按 check name 选择性展开 details 里的 well-known 字段。"""
    logger.info("")
    logger.info("    >> %s  [%s]", r.name, r.severity)
    det = r.details

    if r.name == "yaml_schema" and "problems" in det:
        for p in det["problems"]:
            logger.info("        - %s", p)

    elif r.name == "pair_existence":
        mps = det.get("missing_per_split", {})
        if mps:
            parts = ", ".join(f"{s}={n}" for s, n in mps.items())
            logger.info("        各 split 缺失:  %s", parts)
        ex = det.get("missing_examples", {})
        for split, paths in ex.items():
            preview_count = min(5, len(paths))
            logger.info("        示例 (%s, 前 %d 条):", split, preview_count)
            for p in paths[:5]:
                logger.info("          %s", p)

    elif r.name == "label_format":
        kinds = det.get("error_kinds", {})
        if kinds:
            parts = ", ".join(f"{k}={v}" for k, v in kinds.items())
            logger.info("        错误类型:  %s", parts)
        for e in det.get("errors_preview", [])[:5]:
            label_name = Path(e["label"]).name if "label" in e else "?"
            logger.info(
                "        - %s:%s  %s  %s",
                label_name,
                e.get("line_no", "?"),
                e.get("kind", "?"),
                e.get("detail", ""),
            )

    elif r.name == "split_uniqueness" and det.get("overlaps"):
        for o in det["overlaps"]:
            logger.info(
                "        %s ↔ %s: %d 张重复",
                o["split_a"],
                o["split_b"],
                o["count"],
            )
            for stem in o["preview"][:5]:
                logger.info("          %s", stem)

    else:
        if "reason" in det:
            logger.info("        reason: %s", det["reason"])


# ============================================================
# 段 5: 报告尾
# ============================================================


def _render_footer(report_path: Path | None, logger: logging.Logger) -> None:
    logger.info(H2_LINE)
    if report_path is not None:
        logger.info("  详细报告:  %s", report_path)
    logger.info(H1_LINE)


# ============================================================
# render_to_markdown — 占位 (任务6实现)
# ============================================================


def render_to_markdown(report: object, *, executor: str | None = None) -> str:
    """生成 Markdown 格式的验证报告 (占位, 任务 6 实现)。"""
    raise NotImplementedError("render_to_markdown 尚未实现")


# ============================================================
# render_to_html — 占位 (任务7实现)
# ============================================================


def render_to_html(report: object, *, executor: str | None = None) -> str:
    """生成自包含 HTML 格式的验证报告 (占位, 任务 7 实现)。"""
    raise NotImplementedError("render_to_html 尚未实现")
