"""Command line entry for dataset validation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from od_platform.common.constants import Task
from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR, dataset_yaml_path
from od_platform.data_validation.render import render_to_logger
from od_platform.data_validation.service import validate_dataset

EXIT_OK = 0
EXIT_WARNING = 1
EXIT_DATA_ERROR = 2
EXIT_TOOL_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-validate",
        description="执行数据集质量检查并返回 CI 可用退出码",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", help="数据集配置名称，解析为 configs/datasets/<name>.yaml")
    target.add_argument("--yaml", type=Path, help="dataset.yaml 路径")
    parser.add_argument("--task", default=Task.DETECT, choices=Task.end_to_end())
    parser.add_argument("--no-report", action="store_true", help="只输出日志，不写 JSON 报告")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    parser.add_argument("--executor", "-e", help="执行人姓名，记录到报告审计字段")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_logger(
        base_path=LOGGING_DIR,
        log_type="validate_data",
        log_level=logging.DEBUG if args.verbose else logging.INFO,
        temp_log=False,
        logger_name="od_platform.validate_data",
    )
    warn_cli_if_not_expected_environment(logger=logger)

    try:
        Task.ensure_end_to_end(args.task)
        if args.dataset:
            yaml_path = dataset_yaml_path(args.dataset)
        else:
            yaml_path = args.yaml.resolve()

        report = validate_dataset(
            yaml_path=yaml_path,
            task_type=args.task,
            write_report=not args.no_report,
            executor=args.executor,
        )
        render_to_logger(report, logger, report_path=report.report_path)
        return report.exit_code
    except KeyboardInterrupt:
        logger.error("用户中断")
        return EXIT_TOOL_ERROR
    except Exception:
        logger.exception("未预期异常：CLI 或工具自身失败，不是数据结论")
        return EXIT_TOOL_ERROR


if __name__ == "__main__":
    sys.exit(main())
