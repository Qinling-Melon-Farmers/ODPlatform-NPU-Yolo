"""Command line entry for listing available CV models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

from od_platform.common.constants import Task
from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.common.string_utils import format_table_row, format_table_separator
from od_platform.model_catalog.catalog import PRIMARY_METRIC, ModelInfo, list_models
from od_platform.model_catalog.recommender import recommend_model

EXIT_OK = 0
EXIT_TOOL_ERROR = 2

#: CLI 可查询的任务集合；classify 为未来扩展预留给用户扩展目录。
TASK_CHOICES: tuple[str, ...] = (*Task.all(), "classify")

_TABLE_WIDTHS = (12, 8, 8, 8, 10, 10, 8, 44)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-list-models",
        description="列出可用 CV 模型及其性能指标，支持自然语言推荐",
    )
    parser.add_argument("--task", choices=TASK_CHOICES, help="按任务过滤 (detect/classify)")
    parser.add_argument("--family", help="按模型系列过滤，如 yolo11、yolov8")
    parser.add_argument(
        "--recommend",
        metavar="PREFERENCE",
        help="自然语言偏好描述，如 '最快'、'最准'、'yolov8 最准'",
    )
    parser.add_argument("--limit", type=int, default=10, help="返回数量上限（默认 10）")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_logger(
        base_path=LOGGING_DIR,
        log_type="list_models",
        log_level=logging.INFO,
        temp_log=False,
        logger_name="od_platform.list_models",
    )
    warn_cli_if_not_expected_environment(logger=logger)

    try:
        if args.recommend:
            models = recommend_model(
                args.recommend,
                task=args.task or Task.DETECT,
                family=args.family,
                limit=args.limit,
            )
        else:
            models = list_models(task=args.task, family=args.family)[: args.limit]

        if args.json:
            _render_json(logger, models)
        else:
            _render_table(logger, models)
        return EXIT_OK
    except Exception:
        logger.exception("未预期异常：CLI 或工具自身失败")
        return EXIT_TOOL_ERROR


def _render_table(logger: logging.Logger, models: list[ModelInfo]) -> None:
    """以 CJK 对齐的文本表格输出模型目录。"""
    if not models:
        logger.info("没有匹配的模型")
        return

    # 多任务混排（内置 detect + 扩展 classify）时指标键不唯一，回退 "metric"
    metric_keys = {PRIMARY_METRIC.get(info.task, "metric") for info in models}
    metric_key = metric_keys.pop() if len(metric_keys) == 1 else "metric"
    header = ("名称", "系列", "任务", "后端", "参数量(M)", metric_key, "CPU(ms)", "描述")
    logger.info(format_table_row(header, _TABLE_WIDTHS))
    logger.info(format_table_separator(_TABLE_WIDTHS))

    for info in models:
        metric = info.primary_metric
        metric_text = f"{metric:.1f}" if metric is not None else "-"
        logger.info(
            format_table_row(
                (
                    info.name,
                    info.family,
                    info.task,
                    info.backend,
                    f"{info.params_m:.1f}",
                    metric_text,
                    f"{info.speed_cpu_ms:.0f}",
                    info.description,
                ),
                _TABLE_WIDTHS,
            )
        )
    logger.info("共 %d 个模型", len(models))


def _render_json(logger: logging.Logger, models: list[ModelInfo]) -> None:
    """以 JSON 数组输出模型目录。"""
    payload = [asdict(info) for info in models]
    logger.info(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
