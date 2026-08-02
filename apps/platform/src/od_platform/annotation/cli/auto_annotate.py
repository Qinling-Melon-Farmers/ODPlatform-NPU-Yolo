"""Command line entry for VLM-based automatic dataset annotation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from od_platform.annotation.vlm import VLMConfig, run_vlm_annotation
from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_TOOL_ERROR = 2

#: 常用视觉模型名提示（Qwen-VL / GLM-4.5V 等，实际以 base-url 服务为准）。
VLM_MODEL_HINTS = "qwen-vl-max / glm-4.5v-turbo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-auto-annotate",
        description="用 VLM 按自然语言指令自动标注数据集，输出 YOLO 格式（断点续跑）",
    )
    parser.add_argument("--dataset", required=True, help="数据集名称，定位 data/raw/<dataset>/")
    classes_group = parser.add_mutually_exclusive_group(required=True)
    classes_group.add_argument("--classes", nargs="+", metavar="NAME", help="类别名称列表，index 即类别 ID")
    classes_group.add_argument(
        "--classes-from-yaml",
        action="store_true",
        help="从 configs/datasets/<dataset>.yaml 的 names 自动读取类别",
    )
    parser.add_argument("--prompt", default="框出所有目标", help="自然语言标注指令（默认: 框出所有目标）")
    parser.add_argument(
        "--base-url",
        help="OpenAI 兼容 API 基地址（如 DashScope/Qwen 兼容端点）；缺省读环境变量 OPENAI_BASE_URL",
    )
    parser.add_argument("--model", required=True, help=f"视觉模型名（如 {VLM_MODEL_HINTS}）")
    parser.add_argument("--api-key", help="API 密钥；缺省读环境变量 OPENAI_API_KEY")
    parser.add_argument("--images-dir", type=Path, help="图片目录（默认 data/raw/<dataset>/images）")
    parser.add_argument("--labels-dir", type=Path, help="标注输出目录（默认 data/raw/<dataset>/annotations）")
    parser.add_argument("--limit", type=int, help="本次处理图片数上限（默认全部未标注）")
    parser.add_argument("--retries", type=int, default=2, help="无效 JSON 重试次数（默认 2）")
    parser.add_argument("--dry-run", action="store_true", help="只统计待标注图片数，不调用 API")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    return parser


def _resolve_classes(args) -> list[str]:
    """解析类别列表：显式 --classes 或从 dataset yaml 的 names 读取。"""
    if not args.classes_from_yaml:
        return list(args.classes)
    from od_platform.common.paths import dataset_yaml_path

    yaml_path = dataset_yaml_path(args.dataset)
    if not yaml_path.exists():
        raise FileNotFoundError(f"数据集配置不存在: {yaml_path}")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, dict):
        classes = [str(name) for name in names.values()]
    elif isinstance(names, list):
        classes = [str(name) for name in names]
    else:
        raise ValueError(f"数据集配置 {yaml_path} 缺少 names 字段")
    if not classes:
        raise ValueError(f"数据集配置 {yaml_path} 的 names 为空")
    logger.info("已从 %s 读取类别: %s", yaml_path, classes)
    return classes


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_logger(
        base_path=LOGGING_DIR,
        log_type="auto_annotate",
        log_level=logging.DEBUG if args.verbose else logging.INFO,
        temp_log=False,
        logger_name="od_platform.auto_annotate",
    )
    warn_cli_if_not_expected_environment(logger=logger)

    try:
        classes = _resolve_classes(args)
        config = VLMConfig(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            prompt=args.prompt,
            retries=args.retries,
        )
        report = run_vlm_annotation(
            dataset=args.dataset,
            classes=classes,
            config=config,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        logger.info(
            "报告: 数据集 %s 共 %d 张，本次新增 %d 张（%d 框），失败 %d 张",
            report.dataset,
            report.total_images,
            report.annotated_new,
            report.box_count,
            len(report.failed),
        )
        return EXIT_OK
    except Exception:
        logger.exception("未预期异常：CLI 或工具自身失败")
        return EXIT_TOOL_ERROR


if __name__ == "__main__":
    sys.exit(main())
