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
    """生成 Markdown 格式的验证报告。纯标准库实现，零新依赖。

    报告结构:
        1. 标题 + 元信息表格
        2. 数据集摘要表格
        3. 检查结果一览表格 (emoji 标识严重度)
        4. 失败详情 (按 check 分组展开)
        5. 检测建议
        6. 审计信息 (工具版本、设备信息、签字栏)
    """
    executor = executor or getattr(report, "executor", None)
    lines: list[str] = []

    # ---- 标题 ----
    lines.append("# YOLO 数据集质量验证报告")
    lines.append("")

    # ---- 元信息表格 ----
    severity_emoji = SEVERITY_STYLE.get(report.overall_severity, _SEVERITY_DEFAULT)["emoji"]
    lines.append("| 项目 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| run_id | {report.run_id} |")
    lines.append(f"| yaml_path | {report.yaml_path} |")
    lines.append(f"| task_type | {report.snapshot.task_type} |")
    lines.append(f"| 执行人 | {executor or '(未指定)'} |")
    lines.append(f"| 开始时间 | {report.started_at_iso} |")
    lines.append(f"| 耗时 | {report.duration_seconds:.2f}s |")
    lines.append(f"| 综合评级 | {severity_emoji} {report.overall_severity} |")
    lines.append("")

    # ---- 数据集摘要 ----
    lines.append("## 数据集摘要")
    lines.append("")
    if report.snapshot.class_names:
        lines.append(f"- 类别 (nc={report.snapshot.nc}): {', '.join(report.snapshot.class_names)}")
    else:
        lines.append("- 类别: (未取到)")
    lines.append("")

    if report.snapshot.stats_per_split:
        lines.append("| Split | 图像数 | 标注数 | 实例数 |")
        lines.append("|-------|--------|--------|--------|")
        for split, stat in report.snapshot.stats_per_split.items():
            lines.append(f"| {split} | {stat.image_count} | {stat.annotated_count} | {stat.total_instances} |")
        lines.append("")

    # ---- 检查结果一览 ----
    lines.append("## 检查结果一览")
    lines.append("")
    lines.append("| 检查项 | 严重度 | 摘要 |")
    lines.append("|--------|--------|------|")
    for r in report.results:
        emoji = SEVERITY_STYLE.get(r.severity, _SEVERITY_DEFAULT)["emoji"]
        summary = r.summary.replace("|", "\\|")
        lines.append(f"| {r.name} | {emoji} {r.severity} | {summary} |")
    lines.append("")

    # ---- 失败详情 ----
    failed = getattr(report, "failed_results", [r for r in report.results if not r.passed])
    if failed:
        lines.append("## 失败详情")
        lines.append("")
        for r in failed:
            lines.append(f"### {r.name} [{r.severity}]")
            lines.append("")
            det = r.details

            if r.name == "yaml_schema" and "problems" in det:
                for p in det["problems"]:
                    lines.append(f"- {p}")
            elif r.name == "pair_existence":
                mps = det.get("missing_per_split", {})
                if mps:
                    parts = ", ".join(f"{s}={n}" for s, n in mps.items())
                    lines.append(f"- 各 split 缺失: {parts}")
                ex = det.get("missing_examples", {})
                for split, paths in ex.items():
                    lines.append(f"- 示例 ({split}):")
                    for p in paths[:5]:
                        lines.append(f"  - `{p}`")
            elif r.name == "label_format":
                kinds = det.get("error_kinds", {})
                if kinds:
                    parts = ", ".join(f"{k}={v}" for k, v in kinds.items())
                    lines.append(f"- 错误类型: {parts}")
                for e in det.get("errors_preview", [])[:5]:
                    lines.append(f"- `{Path(e['label']).name}:{e['line_no']}` {e['kind']}: {e['detail']}")
            elif r.name == "split_uniqueness" and det.get("overlaps"):
                for o in det["overlaps"]:
                    lines.append(f"- {o['split_a']} ↔ {o['split_b']}: {o['count']} 张重复")
                    for stem in o["preview"][:5]:
                        lines.append(f"  - `{stem}`")
            else:
                if "reason" in det:
                    lines.append(f"- reason: {det['reason']}")
            lines.append("")

    # ---- 检测建议 ----
    suggestions = getattr(report, "suggestions", [])
    if suggestions:
        lines.append("## 检测建议")
        lines.append("")
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")

    # ---- 审计信息 ----
    lines.append("## 审计信息")
    lines.append("")
    lines.append("- 工具: ODPlatform odp-validate")
    lines.append(f"- 报告生成时间: {report.started_at_iso}")
    lines.append(f"- 执行人: {executor or '(未指定)'}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*签字: _____________ &nbsp;&nbsp;&nbsp;&nbsp; 日期: _____________*")
    lines.append("")

    return "\n".join(lines)


SEVERITY_STYLE: dict[str, dict[str, str]] = {
    "ERROR":   {"emoji": "🔴", "color": "red"},
    "WARNING": {"emoji": "🟡", "color": "yellow"},
    "INFO":    {"emoji": "🔵", "color": "blue"},
    "PASS":    {"emoji": "🟢", "color": "green"},
}
_SEVERITY_DEFAULT = {"emoji": "⚪", "color": "blue"}


# ============================================================
# render_to_html
# ============================================================


def render_to_html(report: object, *, executor: str | None = None) -> str:
    """生成自包含 HTML 格式的验证报告。

    设计原则:
        - 自包含 HTML (embedded CSS, 无外部 CDN 依赖)
        - 可视化图表使用纯 CSS 条形图 + inline SVG, 无 JavaScript
        - 失败详情使用 HTML <details> 折叠
        - 颜色编码: ERROR=红, WARNING=黄, INFO=蓝, PASS=绿
    """
    executor = executor or getattr(report, "executor", None)
    parts: list[str] = []

    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="zh-CN">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append(f"<title>数据质检报告 — {report.run_id}</title>")
    parts.append(_html_css())
    parts.append("</head>")
    parts.append("<body>")

    # 标题
    parts.append('<h1>🔍 YOLO 数据集质量验证报告</h1>')
    parts.append(f'<p class="run-id">Run ID: {report.run_id}</p>')

    # 元信息
    severity_color = SEVERITY_STYLE.get(report.overall_severity, _SEVERITY_DEFAULT)["color"]
    parts.append('<div class="meta-grid">')
    parts.append(_meta_card("YAML 路径", str(report.yaml_path)))
    parts.append(_meta_card("任务类型", report.snapshot.task_type))
    parts.append(_meta_card("执行人", executor or "(未指定)"))
    parts.append(_meta_card("耗时", f"{report.duration_seconds:.2f}s"))
    parts.append(_meta_card("综合评级", f'<span class="badge {severity_color}">{report.overall_severity}</span>'))
    parts.append('</div>')

    # 严重度分布 CSS 条形图
    counts = report.counts_by_severity
    max_count = max(counts.values()) if counts else 1
    parts.append('<h2>📊 严重度分布</h2>')
    parts.append('<div class="bar-chart">')
    for sev, color in [("PASS", "green"), ("INFO", "blue"), ("WARNING", "yellow"), ("ERROR", "red")]:
        count = counts.get(sev, 0)
        pct = (count / max_count * 100) if max_count else 0
        parts.append('<div class="bar-row">')
        parts.append(f'<span class="bar-label">{sev}</span>')
        parts.append(f'<div class="bar-track"><div class="bar-fill {color}" style="width:{pct:.0f}%">{count}</div></div>')
        parts.append("</div>")
    parts.append('</div>')

    # 数据集摘要
    parts.append('<h2>📋 数据集摘要</h2>')
    if report.snapshot.class_names:
        parts.append(f'<p>类别 (nc={report.snapshot.nc}): {", ".join(report.snapshot.class_names)}</p>')
    if report.snapshot.stats_per_split:
        parts.append('<table><thead><tr><th>Split</th><th>图像数</th><th>标注数</th><th>实例数</th></tr></thead><tbody>')
        for split, stat in report.snapshot.stats_per_split.items():
            parts.append(f'<tr><td>{split}</td><td>{stat.image_count}</td><td>{stat.annotated_count}</td><td>{stat.total_instances}</td></tr>')
        parts.append('</tbody></table>')

    # 检查结果一览
    parts.append('<h2>✅ 检查结果一览</h2>')
    parts.append('<table><thead><tr><th>检查项</th><th>严重度</th><th>摘要</th></tr></thead><tbody>')
    for r in report.results:
        color = SEVERITY_STYLE.get(r.severity, _SEVERITY_DEFAULT)["color"]
        parts.append(f'<tr><td>{r.name}</td><td><span class="badge {color}">{r.severity}</span></td><td>{r.summary}</td></tr>')
    parts.append('</tbody></table>')

    # 失败详情 (折叠)
    failed = getattr(report, "failed_results", [r for r in report.results if not r.passed])
    if failed:
        parts.append('<h2>🔎 失败详情</h2>')
        for r in failed:
            color = SEVERITY_STYLE.get(r.severity, _SEVERITY_DEFAULT)["color"]
            parts.append('<details class="check-detail">')
            parts.append(f'<summary><span class="badge {color}">{r.severity}</span> {r.name}</summary>')
            parts.append('<div class="detail-body">')

            det = r.details
            if r.name == "yaml_schema" and "problems" in det:
                parts.append('<ul>')
                for p in det["problems"]:
                    parts.append(f'<li>{p}</li>')
                parts.append('</ul>')
            elif r.name == "pair_existence":
                mps = det.get("missing_per_split", {})
                if mps:
                    parts.append('<ul>')
                    for s, n in mps.items():
                        parts.append(f'<li>{s}: {n} 缺失</li>')
                    parts.append('</ul>')
            elif r.name == "label_format":
                kinds = det.get("error_kinds", {})
                if kinds:
                    parts.append('<ul>')
                    for k, v in kinds.items():
                        parts.append(f'<li>{k}: {v}</li>')
                    parts.append('</ul>')
            elif r.name == "split_uniqueness" and det.get("overlaps"):
                for o in det["overlaps"]:
                    parts.append(f'<p>{o["split_a"]} ↔ {o["split_b"]}: {o["count"]} 张重复</p>')

            if "action" in det:
                parts.append(f'<p class="action">💡 {det["action"]}</p>')
            parts.append('</div>')
            parts.append('</details>')

    # 检测建议
    suggestions = getattr(report, "suggestions", [])
    if suggestions:
        parts.append('<h2>💡 检测建议</h2>')
        parts.append('<ul>')
        for s in suggestions:
            parts.append(f'<li>{s}</li>')
        parts.append('</ul>')

    # 审计
    parts.append('<footer>')
    parts.append(f'<p>工具: ODPlatform odp-validate | 生成时间: {report.started_at_iso} | 执行人: {executor or "(未指定)"}</p>')
    parts.append('</footer>')

    parts.append("</body></html>")
    return "\n".join(parts)


def _html_css() -> str:
    """Return embedded CSS stylesheet."""
    return """<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 960px; margin: 0 auto; padding: 24px; color: #333; background: #f8f9fa; }
h1 { font-size: 24px; margin-bottom: 4px; color: #1a1a2e; }
h2 { font-size: 18px; margin: 24px 0 12px; color: #16213e; border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; }
.run-id { color: #666; font-size: 13px; margin-bottom: 20px; }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; margin-bottom: 24px; }
.meta-card { background: white; border-radius: 8px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.meta-card .label { font-size: 12px; color: #888; text-transform: uppercase; }
.meta-card .value { font-size: 15px; margin-top: 4px; word-break: break-all; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; color: white; }
.badge.red { background: #e74c3c; }
.badge.yellow { background: #f39c12; }
.badge.blue { background: #3498db; }
.badge.green { background: #27ae60; }
.bar-chart { background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.bar-row { display: flex; align-items: center; margin: 8px 0; }
.bar-label { width: 80px; font-size: 13px; font-weight: 600; }
.bar-track { flex: 1; background: #e9ecef; border-radius: 6px; height: 24px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; color: white; font-size: 12px; padding: 3px 8px; min-width: 28px; transition: width .4s; }
.bar-fill.red { background: #e74c3c; }
.bar-fill.yellow { background: #f39c12; }
.bar-fill.blue { background: #3498db; }
.bar-fill.green { background: #27ae60; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 16px; }
th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }
th { background: #f1f3f5; font-weight: 600; color: #555; }
.check-detail { background: white; border-radius: 8px; margin: 8px 0; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.check-detail summary { cursor: pointer; font-weight: 600; font-size: 15px; padding: 4px 0; }
.detail-body { margin-top: 10px; padding-left: 12px; border-left: 3px solid #ddd; }
.detail-body ul { padding-left: 20px; }
.detail-body li { margin: 4px 0; font-size: 13px; }
.action { color: #e67e22; font-weight: 600; margin-top: 8px; }
footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #ddd; font-size: 12px; color: #999; }
</style>"""


def _meta_card(label: str, value: str) -> str:
    """Return a meta info card HTML snippet."""
    return f'<div class="meta-card"><div class="label">{label}</div><div class="value">{value}</div></div>'
