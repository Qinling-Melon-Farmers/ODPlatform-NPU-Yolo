"""测试专用迷你 CLI：验证 Agent 工具执行链路，不触碰真实平台 CLI。

模块名下划线前缀，避免被 pytest 收集为测试。
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini", description="迷你测试 CLI")
    parser.add_argument("--epochs", type=int, help="轮数")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--classes", nargs="+", help="类别列表")
    parser.add_argument("--data", help="数据集")
    parser.add_argument("input", help="输入路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger.info("迷你 CLI 成功执行: input=%s", args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
