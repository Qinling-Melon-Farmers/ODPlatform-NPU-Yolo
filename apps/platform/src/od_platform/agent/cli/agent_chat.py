"""Command line entry for the ODPlatform AI assistant."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from od_platform.agent.client import OpenAIClient
from od_platform.agent.orchestrator import AgentConfig, AgentOrchestrator
from od_platform.agent.tools import build_default_registry
from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR

EXIT_OK = 0
EXIT_TOOL_ERROR = 2
EXIT_INTERRUPTED = 130

#: 常用 OpenAI 兼容模型名提示（DeepSeek/Qwen/GLM 等，实际以 base-url 服务为准）。
MODEL_HINTS = "deepseek-chat / deepseek-reasoner / qwen-vl-max / glm-4.5v-turbo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-agent",
        description="ODPlatform AI 助手：用自然语言驱动平台执行目标检测任务",
    )
    parser.add_argument("--base-url", required=True, help="OpenAI 兼容 API 基地址（如 https://api.deepseek.com/v1）")
    parser.add_argument("--model", required=True, help=f"模型名（如 {MODEL_HINTS}）")
    parser.add_argument("--api-key", help="API 密钥；缺省读环境变量 OPENAI_API_KEY")
    parser.add_argument("--task", help="单次执行的自然语言任务；缺省进入交互模式")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不调用 LLM，打印工具 schema JSON 后退出",
    )
    parser.add_argument("--max-iterations", type=int, default=8, help="最大工具调用轮数（默认 8）")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = get_logger(
        base_path=LOGGING_DIR,
        log_type="agent",
        log_level=logging.DEBUG if args.verbose else logging.INFO,
        temp_log=False,
        logger_name="od_platform.agent",
    )
    warn_cli_if_not_expected_environment(logger=logger)

    try:
        registry = build_default_registry()

        if args.dry_run:
            schemas = registry.schemas()
            logger.info(
                json.dumps(
                    [{"name": s["function"]["name"], "description": s["function"]["description"]} for s in schemas],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            logger.info("共 %d 个工具（dry-run 模式，未调用 LLM）", len(schemas))
            return EXIT_OK

        config = AgentConfig(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_iterations=args.max_iterations,
        )
        client = OpenAIClient(api_key=args.api_key, base_url=args.base_url)
        orchestrator = AgentOrchestrator(client=client, registry=registry, config=config)

        if args.task:
            return _run_once(orchestrator, args.task, logger)
        return _run_repl(orchestrator, logger)
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("未预期异常：CLI 或工具自身失败")
        return EXIT_TOOL_ERROR


def _run_once(orchestrator: AgentOrchestrator, task: str, logger: logging.Logger) -> int:
    """单次任务模式：流式渲染事件后退出。"""
    for event in orchestrator.run_stream(task):
        _render_event(event, logger)
    return EXIT_OK


def _run_repl(orchestrator: AgentOrchestrator, logger: logging.Logger) -> int:
    """交互模式：逐条输入自然语言任务。"""
    logger.info("进入交互模式（输入 exit 或 Ctrl+C 退出）")
    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出"):
            break
        for event in orchestrator.run_stream(user_input):
            _render_event(event, logger)
    return EXIT_OK


def _render_event(event, logger: logging.Logger) -> None:
    """将 AgentEvent 渲染为日志输出（桌面端复用事件流做 UI 渲染）。"""
    if event.kind == "message":
        logger.info("%s", event.content)
    elif event.kind == "tool_start":
        logger.info("→ 调用工具 %s(%s)", event.tool_name, event.content or "")
    elif event.kind == "tool_result" and event.tool_result is not None:
        status = "成功" if event.tool_result.ok else "失败"
        logger.info("← 工具 %s %s（退出码 %s）: %s", event.tool_name, status, event.tool_result.exit_code, event.tool_result.summary[:200])
    elif event.kind == "error":
        logger.error("Agent 错误: %s", event.content)
    elif event.kind == "done":
        logger.info("== 完成 ==")


if __name__ == "__main__":
    sys.exit(main())
