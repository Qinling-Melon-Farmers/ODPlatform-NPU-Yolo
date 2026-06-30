"""项目重置命令行工具。

安全清理 ODPlatform 运行时产物，使工作区接近 git clone 后状态。

@FileName:   reset_project.py
@Function:   reset_project CLI、dry-run、安全校验与审计日志
"""

import argparse
import logging
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from od_platform.common import paths
from od_platform.common.audit_utils import (
    _audit_context,
    append_audit_result,
    write_audit_record,
)
from od_platform.common.logging_utils import get_logger
from od_platform.common.performance_utils import time_it
from od_platform.common.string_utils import format_table_row, format_table_separator
from od_platform.common.system_utils import _format_size

LINE_WIDTH = 72
LARGE_DIR_BYTES = 1 << 30
PRESERVED_PLACEHOLDERS = {"README.md", ".gitkeep"}

logger = logging.getLogger("od_platform.cli.reset_project")


@dataclass(frozen=True)
class ScanResult:
    """单个 reset 目标目录的扫描结果。"""

    path: Path
    file_count: int
    total_bytes: int
    exists: bool


@dataclass(frozen=True)
class DeleteFailure:
    """单个目录删除失败明细。"""

    path: Path
    error: str


def _format_relative(path: Path) -> str:
    """格式化工作区相对路径。

    Args:
        path: 待格式化路径。

    Returns:
        相对 ROOT_DIR 的路径；越界时返回绝对路径。
    """
    try:
        return str(path.relative_to(paths.ROOT_DIR))
    except ValueError:
        return str(path)


def _is_preserved_placeholder(root: Path, child: Path) -> bool:
    return child.parent == root and child.name in PRESERVED_PLACEHOLDERS


@time_it(iterations=1, name="reset 目录扫描", logger_instance=logger)
def _scan_dir(path: Path) -> ScanResult:
    """扫描目录中将被清理的文件数量与字节数。

    Args:
        path: reset 目标目录。

    Returns:
        扫描结果；目录不存在时文件数与字节数为 0。
    """
    if not path.exists():
        return ScanResult(path=path, file_count=0, total_bytes=0, exists=False)

    file_count = 0
    total_bytes = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)
                    if _is_preserved_placeholder(path, entry_path):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry_path)
                        elif entry.is_file(follow_symlinks=False):
                            file_count += 1
                            total_bytes += entry.stat(follow_symlinks=False).st_size
                    except OSError as exc:
                        logger.warning("扫描跳过异常路径 %s: %s", entry_path, exc)
        except OSError as exc:
            logger.warning("扫描目录失败 %s: %s", current, exc)
    return ScanResult(path=path, file_count=file_count, total_bytes=total_bytes, exists=True)


def _validate_targets(targets: list[Path]) -> bool:
    """执行 reset 目标二次保护校验。

    Args:
        targets: 由 ``paths.get_dirs_to_reset`` 返回的白名单目标。

    Returns:
        全部通过返回 True，任意路径受保护时返回 False。
    """
    for target in targets:
        if paths.is_protected(target):
            logger.error("%s 触发了 PROTECTED_DIRS 校验，已整体中止", target)
            return False
    return True


def _print_plan(results: list[ScanResult], dry_run_mode: bool) -> None:
    """输出 reset 删除计划。

    Args:
        results: 目标目录扫描结果。
        dry_run_mode: 是否 dry-run。

    Returns:
        None。
    """
    header = "📋 [DRY-RUN] 计划如下（未实际删除）" if dry_run_mode else "⚠️ 即将删除以下目录"
    logger.info(header)
    widths = [30, 10, 14, 10]
    aligns = ["left", "right", "right", "center"]
    logger.info(format_table_row(["目录", "文件数", "大小", "状态"], widths, aligns))
    logger.info(format_table_separator(widths))
    for result in results:
        status = "存在" if result.exists else "不存在"
        logger.info(
            format_table_row(
                [
                    _format_relative(result.path),
                    result.file_count,
                    _format_size(result.total_bytes),
                    status,
                ],
                widths,
                aligns,
            )
        )
    total_files = sum(item.file_count for item in results)
    total_bytes = sum(item.total_bytes for item in results)
    logger.info(format_table_separator(widths))
    logger.info(format_table_row(["合计", total_files, _format_size(total_bytes), "-"], widths, aligns))
    logger.info("不会触碰: data/raw/、models/pretrained/、apps/、docs/、scripts/、.git/")


def _confirm(total_targets: int) -> bool:
    """执行交互式二次确认。

    Args:
        total_targets: 将要清理的目录数量。

    Returns:
        用户精确输入 ``RESET`` 时返回 True，否则返回 False。
    """
    logger.warning("⚠️  你正要删除 %d 个目录中的运行产物。这个操作不可撤销。", total_targets)
    logger.warning("请精确输入 RESET 继续，其他任何输入都会取消。")
    try:
        answer = input("输入 RESET 确认: ")
    except KeyboardInterrupt:
        logger.warning("❌ 用户取消，未执行删除")
        return False
    if answer != "RESET":
        logger.warning("❌ 用户取消，未执行删除")
        return False
    return True


def _on_rm_error(func, path: str, exc_info) -> None:
    """处理 Windows 只读文件删除失败。

    Args:
        func: ``shutil.rmtree`` 传入的失败函数。
        path: 删除失败路径。
        exc_info: 异常信息三元组。

    Returns:
        None。
    """
    del exc_info
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _clear_directory_contents(root: Path) -> None:
    """清理目录内容，同时保留 Git 占位文件。

    Args:
        root: reset 目标目录。

    Returns:
        None。
    """
    if not root.exists():
        return
    for child in root.iterdir():
        if _is_preserved_placeholder(root, child):
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, onerror=_on_rm_error)
        else:
            child.unlink()


def _execute_deletion(results: list[ScanResult]) -> list[DeleteFailure]:
    """执行真实清理。

    Args:
        results: 目标目录扫描结果。

    Returns:
        删除失败列表。
    """
    failures: list[DeleteFailure] = []
    total = len(results)
    for index, result in enumerate(results, start=1):
        rel_path = _format_relative(result.path)
        logger.info(
            "[%d/%d] 删除 %s (%s, %d 个文件)",
            index,
            total,
            rel_path,
            _format_size(result.total_bytes),
            result.file_count,
        )
        if result.total_bytes >= LARGE_DIR_BYTES:
            logger.warning("——这可能需要一会...")
        try:
            _clear_directory_contents(result.path)
        except OSError as exc:
            logger.error("[%d/%d] ❌ 删除失败 %s: %s", index, total, rel_path, exc)
            failures.append(DeleteFailure(path=result.path, error=str(exc)))
            continue
        logger.info("[%d/%d] ✅ 已删除: %s", index, total, rel_path)
    return failures


def _print_summary(results: list[ScanResult], failures: list[DeleteFailure]) -> int:
    """输出执行汇总并计算退出码。

    Args:
        results: 全部目标扫描结果。
        failures: 删除失败列表。

    Returns:
        标准化进程退出码。
    """
    success_count = len(results) - len(failures)
    logger.info("=" * LINE_WIDTH)
    logger.info("完成: 成功 %d 个，失败 %d 个", success_count, len(failures))
    for failure in failures:
        logger.error("  - %s: %s", _format_relative(failure.path), failure.error)
    if not failures:
        return 0
    if len(failures) == len(results):
        return 2
    return 1


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Returns:
        argparse 参数解析器。
    """
    parser = argparse.ArgumentParser(
        prog="odp-reset",
        description="安全重置 ODPlatform 运行时产物。默认 dry-run，必须 --yes 才会删除。",
    )
    parser.add_argument("--yes", action="store_true", help="真正执行删除；默认只 dry-run。")
    parser.add_argument("--force", action="store_true", help="跳过确认；仅与 --yes 同时使用时生效。")
    parser.add_argument("--dry-run", action="store_true", help="显式 dry-run；优先级高于 --yes。")
    return parser


def reset_project(yes: bool = False, force: bool = False, dry_run: bool = False) -> int:
    """重置 ODPlatform 运行时目录。

    Args:
        yes: 是否真正执行删除。
        force: 是否跳过交互确认，仅在 ``yes=True`` 时有效。
        dry_run: 是否显式 dry-run；与 ``yes`` 同时出现时优先。

    Returns:
        标准化退出码：0 成功或 dry-run，1 部分失败，2 全部失败或保护拦截。
    """
    get_logger(
        base_path=paths.META_LOGGING_DIR,
        log_type="reset_project",
        logger_name="od_platform.cli.reset_project",
    )
    audit_dir = paths.META_LOGGING_DIR / "reset_project"
    context = _audit_context(
        root_dir=paths.ROOT_DIR,
        argv=[f"yes={yes}", f"force={force}", f"dry_run={dry_run}"],
    )
    audit_file = write_audit_record(audit_dir, context)
    logger.info("[AUDIT] %s", audit_file)

    dry_run_mode = dry_run or not yes
    if dry_run and yes:
        logger.warning("⚠️ 同时给了 --dry-run 和 --yes，以 --dry-run 为准（只打印不删除）")

    targets = paths.get_dirs_to_reset()
    if not _validate_targets(targets):
        append_audit_result(audit_file, {"exit_code": 2, "reason": "protected_target"})
        return 2

    results = [_scan_dir(target) for target in targets]
    _print_plan(results, dry_run_mode=dry_run_mode)

    if dry_run_mode:
        logger.info("💡 这是 dry-run（默认行为）。要真正执行删除，请加 --yes:")
        logger.info("   odp-reset --yes")
        append_audit_result(audit_file, {"exit_code": 0, "mode": "dry_run"})
        return 0

    if not force and not _confirm(len(targets)):
        append_audit_result(audit_file, {"exit_code": 0, "mode": "cancelled"})
        return 0

    failures = _execute_deletion(results)
    exit_code = _print_summary(results, failures)
    append_audit_result(
        audit_file,
        {
            "exit_code": exit_code,
            "mode": "delete",
            "failed": [_format_relative(item.path) for item in failures],
        },
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。

    Args:
        argv: 可选命令行参数列表；为 None 时由 argparse 读取 ``sys.argv``。

    Returns:
        标准化退出码。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return reset_project(yes=args.yes, force=args.force, dry_run=args.dry_run)
    except Exception:
        logger.exception("reset_project 发生未捕获异常")
        return 1


if __name__ == "__main__":
    sys.exit(main())
