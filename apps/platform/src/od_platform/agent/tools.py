"""Agent 工具注册表与执行调度。

两种执行方式：
- **CLI 工具**：进程内调用 ``module.main(argv)``（复用全部既有 CLI 逻辑，
  包括默认值、runtime config 解析、日志、审计、退出码）；argparse 的
  SystemExit 被映射为失败观察回填给 LLM。
- **服务工具**：直接调用服务层函数，返回结构化文本（上下文感知工具，
  如列出数据集/模型）。

@FileName:   tools.py
@Function:   工具注册、argv 构造、执行与结果封装
"""

from __future__ import annotations

import importlib
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from od_platform.agent.schema import ToolSchema, parser_to_tool_schema

logger = logging.getLogger(__name__)

#: 回填给 LLM 的观察文本长度上限。
SUMMARY_LIMIT = 4000
#: 完整日志长度上限。
DETAILS_LIMIT = 20000


@dataclass(frozen=True)
class ToolResult:
    """一次工具执行的结果。

    Attributes:
        name:      工具名。
        arguments: 实际使用的参数（已脱敏）。
        ok:        是否成功。
        exit_code: CLI 退出码；服务工具为 None。
        summary:   给 LLM 的观察文本（截断）。
        details:   完整日志尾部（供人工排查）。
    """

    name: str
    arguments: dict
    ok: bool
    exit_code: int | None
    summary: str
    details: str


ServiceHandler = Callable[[dict], ToolResult]


@dataclass
class _CliEntry:
    """一个 CLI 工具的注册信息。"""

    module: str
    entry: str
    schema: ToolSchema
    requires_confirmation: bool = False
    excluded_args: tuple[str, ...] = ()
    dry_run_flag: bool = False


@dataclass
class _ServiceEntry:
    """一个服务层工具的注册信息。"""

    name: str
    description: str
    parameters: dict
    handler: ServiceHandler


class ToolRegistry:
    """工具注册表与执行调度器。

    Args:
        dry_run: 安全模式；True 时对注册了 ``dry_run_flag`` 的工具自动追加
                 ``--dry-run``（训练/推理等耗时工具默认不真实执行）。
        executor: 执行人标识，写入工具执行的审计日志。
    """

    def __init__(self, *, dry_run: bool = False, executor: str = "agent") -> None:
        self.dry_run = dry_run
        self.executor = executor
        self._cli_entries: dict[str, _CliEntry] = {}
        self._service_entries: dict[str, _ServiceEntry] = {}
        self.confirmed_tools: set[str] = set()

    # ---- 注册 ----

    def register_cli(
        self,
        module: str,
        *,
        entry: str = "main",
        requires_confirmation: bool = False,
        excluded_args: tuple[str, ...] = (),
        dry_run_flag: bool = False,
    ) -> None:
        """注册一个 CLI 模块为工具（走 ``module.main(argv)`` 进程内调用）。"""
        try:
            module_obj = importlib.import_module(module)
        except ImportError as exc:
            logger.warning("注册工具失败（模块不可导入）: %s: %s", module, exc)
            return
        parser = getattr(module_obj, "build_parser", lambda: None)()
        schema = parser_to_tool_schema(parser) if parser is not None else None
        if schema is None:
            logger.warning("注册工具失败（无有效 parser）: %s", module)
            return

        if excluded_args:
            # 支持选项字符串（--api-key）或属性名（api_key）两种写法
            excluded = {arg.lstrip("-").replace("-", "_") for arg in excluded_args}
            properties = dict(schema.parameters["properties"])
            for key in excluded:
                properties.pop(key, None)
            parameters = {"type": "object", "properties": properties}
            required = [key for key in schema.parameters.get("required", []) if key not in excluded]
            if required:
                parameters["required"] = required
            schema = ToolSchema(
                name=schema.name,
                description=schema.description,
                parameters=parameters,
                arg_specs=[spec for spec in schema.arg_specs if spec.key not in excluded],
            )

        self._cli_entries[schema.name] = _CliEntry(
            module=module,
            entry=entry,
            schema=schema,
            requires_confirmation=requires_confirmation,
            excluded_args=excluded_args,
            dry_run_flag=dry_run_flag,
        )

    def register_service(
        self,
        name: str,
        *,
        description: str,
        parameters: dict | None = None,
    ) -> Callable[[ServiceHandler], ServiceHandler]:
        """注册一个服务层工具（装饰器）。"""

        def decorator(handler: ServiceHandler) -> ServiceHandler:
            self._service_entries[name] = _ServiceEntry(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                handler=handler,
            )
            return handler

        return decorator

    # ---- 查询 ----

    def names(self) -> list[str]:
        return [*self._cli_entries, *self._service_entries]

    def schemas(self) -> list[dict]:
        """返回 OpenAI 格式的工具定义列表。"""
        result: list[dict] = []
        for name, entry in self._cli_entries.items():
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": entry.schema.description,
                        "parameters": entry.schema.parameters,
                    },
                }
            )
        for name, entry in self._service_entries.items():
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": entry.description,
                        "parameters": entry.parameters,
                    },
                }
            )
        return result

    def get_schema(self, name: str) -> ToolSchema | None:
        entry = self._cli_entries.get(name)
        return entry.schema if entry is not None else None

    # ---- 执行 ----

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """执行一个工具并返回结果。

        Args:
            name: 工具名。
            arguments: LLM 提供的参数 dict。

        Returns:
            执行结果（失败也返回 ToolResult，不抛异常）。
        """
        safe_arguments = _redact_arguments(arguments)
        if name in self._service_entries:
            entry = self._service_entries[name]
            try:
                return entry.handler(dict(arguments))
            except Exception as exc:  # noqa: BLE001 - 边界层统一转为失败结果
                logger.exception("服务工具 %s 执行异常", name)
                return ToolResult(
                    name=name,
                    arguments=safe_arguments,
                    ok=False,
                    exit_code=None,
                    summary=f"工具 {name} 执行异常: {type(exc).__name__}: {exc}",
                    details="",
                )

        entry = self._cli_entries.get(name)
        if entry is None:
            return ToolResult(
                name=name,
                arguments=safe_arguments,
                ok=False,
                exit_code=None,
                summary=f"未知工具: {name}",
                details="",
            )

        if entry.requires_confirmation and name not in self.confirmed_tools:
            return ToolResult(
                name=name,
                arguments=safe_arguments,
                ok=False,
                exit_code=None,
                summary=f"工具 {name} 需要用户确认后才可执行，请先征得用户同意",
                details="",
            )

        argv = _argv_from_arguments(entry.schema, arguments)
        if self.dry_run and entry.dry_run_flag:
            argv.append("--dry-run")

        output = _run_cli(entry.module, entry.entry, argv, executor=self.executor)
        summary = output.output.strip()
        if len(summary) > SUMMARY_LIMIT:
            summary = summary[:SUMMARY_LIMIT] + "\n...(已截断)"
        return ToolResult(
            name=name,
            arguments=safe_arguments,
            ok=output.exit_code == 0,
            exit_code=output.exit_code,
            summary=summary or f"工具 {name} 无输出，退出码 {output.exit_code}",
            details=output.details[:DETAILS_LIMIT],
        )

    def confirm(self, name: str) -> None:
        """确认执行某工具（交互 REPL 下用户二次确认）。"""
        self.confirmed_tools.add(name)


@dataclass(frozen=True)
class _CliOutput:
    exit_code: int
    output: str
    details: str


def _run_cli(module: str, entry: str, argv: list[str], *, executor: str) -> _CliOutput:
    """进程内调用 CLI 模块入口，捕获日志输出与退出码。"""
    module_obj = importlib.import_module(module)
    main_func = getattr(module_obj, entry)

    capture = io.StringIO()
    handler = logging.StreamHandler(capture)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    # pytest 等宿主可能把 root level 设为 WARNING，临时降级以捕获 CLI 日志
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        try:
            code = main_func(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
        except KeyboardInterrupt:
            code = 130
        except Exception as exc:  # noqa: BLE001 - 边界层统一映射为失败
            logger.exception("CLI 工具 %s 执行异常", module)
            code = 2
            logging.getLogger("agent.tools").error("工具异常: %s: %s", type(exc).__name__, exc)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
    output = capture.getvalue()
    return _CliOutput(exit_code=code, output=output, details=output)


def _argv_from_arguments(schema: ToolSchema, arguments: dict) -> list[str]:
    """按 ArgSpec 元信息从 LLM arguments 构造 argv。"""
    argv: list[str] = []
    positionals: list[tuple[int, str]] = []
    for spec in schema.arg_specs:
        if spec.key not in arguments or arguments[spec.key] is None:
            continue
        value = arguments[spec.key]
        if spec.is_positional:
            if isinstance(value, list):
                for item in value:
                    positionals.append((spec.positional_index or 0, str(item)))
            else:
                positionals.append((spec.positional_index or 0, str(value)))
            continue
        option = spec.option
        if option is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(option)
            continue
        if isinstance(value, list):
            argv.append(option)
            argv.extend(str(item) for item in value)
            continue
        argv.extend([option, str(value)])

    positionals.sort(key=lambda pair: pair[0])
    argv.extend(item for _index, item in positionals)
    return argv


def _redact_arguments(arguments: dict) -> dict:
    """脱敏参数中的密钥字段（key/api_key 等）。"""
    redacted: dict[str, Any] = {}
    for key, value in arguments.items():
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def build_default_registry(*, dry_run: bool = False, executor: str = "agent") -> ToolRegistry:
    """构建默认工具注册表（服务工具 + 平台 CLI 工具）。"""
    registry = ToolRegistry(dry_run=dry_run, executor=executor)

    _register_service_tools(registry)
    for module, flags in _DEFAULT_CLI_TOOLS:
        registry.register_cli(module, **flags)
    return registry


def _register_service_tools(registry: ToolRegistry) -> None:
    """注册上下文感知的服务层工具。"""
    from od_platform.common import paths
    from od_platform.common.constants import IMAGE_EXTENSIONS

    @registry.register_service(
        "list_datasets",
        description="列出 data/raw 下所有原始数据集，含图片数与已标注数（标注前先确认现状）",
    )
    def _list_datasets(_arguments: dict) -> ToolResult:
        lines: list[str] = []
        raw_root = paths.RAW_DATA_DIR
        if not raw_root.exists():
            return ToolResult("list_datasets", {}, True, None, "data/raw 目录不存在", "")
        suffixes = {suffix.lower() for suffix in IMAGE_EXTENSIONS}
        for dataset_dir in sorted(raw_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            images_dir = dataset_dir / "images"
            labels_dir = dataset_dir / "annotations"
            images = sum(
                1
                for path in images_dir.glob("*")
                if path.is_file() and path.suffix.lower() in suffixes
            )
            annotated = sum(1 for path in labels_dir.glob("*.txt")) if labels_dir.exists() else 0
            lines.append(f"{dataset_dir.name}: 图片 {images}，已标注 {annotated}")
        return ToolResult(
            "list_datasets",
            {},
            True,
            None,
            "\n".join(lines) if lines else "data/raw 下没有数据集",
            "",
        )

    @registry.register_service(
        "list_available_models",
        description="列出可用模型（目录内置 YOLO 系列 + 本地权重），训练前确认模型名",
    )
    def _list_available_models(_arguments: dict) -> ToolResult:
        from od_platform.common.refs import list_available_models

        names = list_available_models()
        return ToolResult(
            "list_available_models",
            {},
            True,
            None,
            "可用模型: " + ", ".join(names) if names else "无可用模型",
            "",
        )

    @registry.register_service(
        "odp-list-models",
        description="列出或推荐 YOLO 模型：传 preference 用自然语言（如最快/最准），family 按系列过滤",
        parameters={
            "type": "object",
            "properties": {
                "preference": {"type": "string", "description": "自然语言偏好，如 '最快'、'yolov8 最准'"},
                "family": {"type": "string", "description": "系列过滤，如 yolov8/yolo11"},
                "limit": {"type": "integer", "description": "返回数量上限（默认 5）"},
            },
        },
    )
    def _list_models(arguments: dict) -> ToolResult:
        from od_platform.common.constants import Task
        from od_platform.model_catalog import recommend_model

        preference = arguments.get("preference") or ""
        family = arguments.get("family") or None
        limit = int(arguments.get("limit") or 5)
        models = recommend_model(str(preference), task=Task.DETECT, family=family, limit=limit)
        lines = [
            f"{info.name} [{info.family} {info.variant}] mAP50-95={info.primary_metric} "
            f"CPU={info.speed_cpu_ms}ms: {info.description}"
            for info in models
        ]
        return ToolResult(
            "odp-list-models",
            {"preference": preference, "family": family, "limit": limit},
            True,
            None,
            "\n".join(lines) if lines else "没有匹配的模型",
            "",
        )


#: (模块路径, 注册标志)。auto-annotate 的凭据只走环境变量。
_DEFAULT_CLI_TOOLS: list[tuple[str, dict]] = [
    ("od_platform.cli.import_dataset", {}),
    ("od_platform.cli.transform_data", {}),
    ("od_platform.cli.validate_data", {}),
    ("od_platform.runtime_config.generator", {}),
    ("od_platform.cli.train_model", {"dry_run_flag": True}),
    ("od_platform.cli.evaluate_model", {}),
    ("od_platform.cli.infer_model", {"dry_run_flag": True}),
    ("od_platform.cli.plot_training", {}),
    (
        "od_platform.annotation.cli.auto_annotate",
        {"excluded_args": ("--api-key", "--base-url")},
    ),
]
