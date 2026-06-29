import logging
from pathlib import Path
from typing import List

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR, RAW_DATA_DIR, ROOT_DIR, get_dirs_to_initialize
from od_platform.common.string_utils import format_table_row, format_table_separator

logger = logging.getLogger("od_platform.cli.init_project")


def _format_relative(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR))


def _check_raw_data_status() -> None:
    if not RAW_DATA_DIR.exists():
        logger.warning(
            "原始数据目录不存在，请创建以数据集名称命名的子文件夹，并包含 images/ 与 annotations/"
        )
        return

    children = [path for path in RAW_DATA_DIR.iterdir() if path.is_dir()]
    if not children:
        logger.warning(
            "原始数据目录为空，请放入至少一个数据集，预期结构为 data/raw/<dataset>/images 与 annotations"
        )
        return

    names = ", ".join(path.name for path in children)
    logger.info("发现原始数据集目录: %s", names)


def initialize_project() -> None:
    """初始化 ODPlatform 项目运行时目录。"""
    get_logger(base_path=LOGGING_DIR, log_type="init_project")

    line_width = 60
    logger.info("开始初始化项目核心目录".center(line_width, "="))
    logger.info("项目根目录: %s", ROOT_DIR)

    created: List[Path] = []
    existed: List[Path] = []

    for directory in get_dirs_to_initialize():
        relative_path = _format_relative(directory)
        try:
            if directory.exists():
                logger.info("目录已存在: %s", relative_path)
                existed.append(directory)
                continue

            directory.mkdir(parents=True, exist_ok=True)
            logger.info("成功创建: %s", relative_path)
            created.append(directory)
        except OSError as exc:
            logger.error("创建目录失败: %s，错误: %s", relative_path, exc)
            raise SystemExit(1) from exc

    _check_raw_data_status()

    logger.info("初始化汇总".center(line_width, "="))
    widths = [25, 10]
    aligns = ["left", "center"]
    logger.info(format_table_row(["目录", "状态"], widths, aligns))
    logger.info(format_table_separator(widths))
    for directory in created:
        logger.info(format_table_row([_format_relative(directory), "新创建"], widths, aligns))
    for directory in existed:
        logger.info(format_table_row([_format_relative(directory), "已存在"], widths, aligns))
    logger.info(format_table_separator(widths))
    logger.info(
        "初始化完成: 新建了 %d 个目录，已经存在了 %d 个目录",
        len(created),
        len(existed),
    )


if __name__ == "__main__":
    initialize_project()
