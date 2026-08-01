"""Command line entry for interactive dataset annotation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from od_platform.annotation.canvas import AnnotationCanvas
from od_platform.annotation.session import AnnotationSession
from od_platform.common import paths
from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR

EXIT_OK = 0
EXIT_TOOL_ERROR = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-annotate",
        description="交互式边界框标注，输出 YOLO 格式标注文件",
    )
    parser.add_argument("--dataset", required=True, help="数据集名称，定位 data/raw/<dataset>/")
    parser.add_argument("--classes", nargs="+", required=True, metavar="NAME", help="类别名称列表，index 即类别 ID")
    parser.add_argument("--images-dir", type=Path, help="图片目录（默认 data/raw/<dataset>/images）")
    parser.add_argument("--labels-dir", type=Path, help="标注输出目录（默认 data/raw/<dataset>/annotations）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="继续中断的会话（默认即跳过已标注图片，此参数为显式语义提示）",
    )
    parser.add_argument("--max-display-size", type=int, default=1200, help="显示尺寸上限（默认 1200）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_logger(
        base_path=LOGGING_DIR,
        log_type="annotate",
        log_level=logging.INFO,
        temp_log=False,
        logger_name="od_platform.annotate",
    )
    warn_cli_if_not_expected_environment(logger=logger)

    try:
        raw_root = paths.RAW_DATA_DIR / args.dataset
        images_dir = args.images_dir or raw_root / "images"
        labels_dir = args.labels_dir or raw_root / "annotations"

        session = AnnotationSession(
            images_dir=images_dir,
            labels_dir=labels_dir,
            classes=args.classes,
        )
        if not session.images:
            logger.error("图片目录为空或无图片: %s", images_dir)
            return EXIT_TOOL_ERROR

        annotated, total = session.progress
        logger.info("开始标注数据集 %s: 共 %d 张图片，已完成 %d 张", args.dataset, total, annotated)
        logger.info("类别: %s", " / ".join(f"{index}:{name}" for index, name in enumerate(args.classes)))

        canvas = AnnotationCanvas(classes=args.classes, max_display_size=args.max_display_size)
        processed = 0
        while True:
            image_path = session.next_unannotated()
            if image_path is None:
                logger.info("全部图片标注完成")
                break

            existing = session.load_labels(image_path.stem) if args.resume else []
            result = canvas.annotate_image(image_path, existing=existing)
            if result.is_quit:
                logger.info("用户退出标注会话")
                break
            if result.action == "save":
                session.save_labels(image_path.stem, result.boxes)
            processed += 1

            done, remaining_total = session.progress
            logger.info(
                "进度 %d/%d: %s (%s, %d 框)",
                done,
                remaining_total,
                image_path.name,
                result.action,
                len(result.boxes),
            )

        done, remaining_total = session.progress
        logger.info("标注完成: %d/%d 张已标注，产物目录 %s", done, remaining_total, labels_dir)
        return EXIT_OK
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("未预期异常：CLI 或工具自身失败")
        return EXIT_TOOL_ERROR


if __name__ == "__main__":
    sys.exit(main())
