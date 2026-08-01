"""Command line entry for plotting Ultralytics training results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.training.metrics import write_metrics_summary
from od_platform.training.plots import plot_training_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odp-plot-training", description="Plot and summarize Ultralytics results.csv.")
    parser.add_argument("csv_path", type=Path, help="Path to Ultralytics results.csv.")
    parser.add_argument("--output", "-o", type=Path, help="Output image path.")
    parser.add_argument("--summary", type=Path, help="Output summary json path.")
    parser.add_argument("--matplotx", action="store_true", help="Use matplotx style when installed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_logger(
        base_path=LOGGING_DIR,
        log_type="plot_training",
        log_level=logging.INFO,
        temp_log=False,
        logger_name="od_platform.plot_training",
    )
    warn_cli_if_not_expected_environment(logger=logger)
    try:
        plot_training_results(args.csv_path, args.output, use_matplotx=args.matplotx)
        if args.summary:
            write_metrics_summary(args.csv_path, args.summary)
            _log_summary(logger, args.summary)
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


def _fmt_metric(value: object) -> str:
    """格式化指标；缺失/非数值时输出 '-'，避免 0.0 误导 Agent 观察。"""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "-"


def _log_summary(logger: logging.Logger, summary_path: Path) -> None:
    """输出摘要关键指标（Agent 观察依赖 CLI 的日志输出）。"""
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        last = payload.get("last") or {}
        best = payload.get("best") or {}
        best_map50 = best.get("map50")
        best_epoch = best_map50.get("epoch", "-") if isinstance(best_map50, dict) else "-"
        logger.info(
            "训练摘要: %s 轮; 末轮 mAP50=%s mAP50-95=%s P=%s R=%s; 最佳 mAP50 在第 %s 轮",
            payload.get("epochs", 0),
            _fmt_metric(last.get("map50")),
            _fmt_metric(last.get("map50_95")),
            _fmt_metric(last.get("precision")),
            _fmt_metric(last.get("recall")),
            best_epoch,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        logger.info("训练摘要已写入: %s", summary_path)


if __name__ == "__main__":
    sys.exit(main())
