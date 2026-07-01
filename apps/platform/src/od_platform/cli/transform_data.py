"""Command line entry for dataset transformation."""

from __future__ import annotations

import argparse
import logging
import sys

from od_platform.common.constants import AnnotationFormat, SplitStrategy, Task
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.data_pipeline.orchestrator import DatasetPipeline

EXIT_OK = 0
EXIT_DATA_ERR = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-transform",
        description="转换、划分并落盘一个可训练的 YOLO 数据集",
    )
    parser.add_argument("--dataset", required=True, help="数据集名称或路径")
    parser.add_argument("--format", required=True, choices=AnnotationFormat.all(), dest="annotation_format")
    parser.add_argument("--task", default=Task.DETECT, choices=Task.all())
    parser.add_argument("--split-strategy", default=SplitStrategy.RANDOM, choices=SplitStrategy.all())
    parser.add_argument("--classes", nargs="+", default=None, help="类别白名单/类别顺序")
    parser.add_argument("--train-rate", type=float, default=0.8)
    parser.add_argument("--val-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1210)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_logger(
        base_path=LOGGING_DIR,
        log_type="transform_data",
        temp_log=False,
        logger_name="od_platform.transform_data",
    )

    try:
        result = DatasetPipeline(
            args.dataset,
            args.annotation_format,
            task=args.task,
            train_rate=args.train_rate,
            val_rate=args.val_rate,
            classes=args.classes,
            random_state=args.seed,
            split_strategy=args.split_strategy,
        ).run()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("处理失败: %s", exc)
        return EXIT_DATA_ERR
    except KeyboardInterrupt:
        logger.error("用户中断")
        return EXIT_USAGE

    logging.getLogger("od_platform.transform_data").info("处理完成: %s", result)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
