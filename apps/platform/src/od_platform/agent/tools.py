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
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from od_platform.agent.schema import ToolSchema, parser_to_tool_schema

logger = logging.getLogger(__name__)

#: 回填给 LLM 的观察文本长度上限。
SUMMARY_LIMIT = 4000
#: 完整日志长度上限。
DETAILS_LIMIT = 20000
#: ANSI 颜色码（控制台 formatter 输出会携带）。
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: 工具风险等级（评审 P1-3）。read_only 自动执行，其余需用户确认。
RISK_READ_ONLY = "read_only"
RISK_WRITE_FILES = "write_files"
RISK_COST_API = "cost_api"
RISK_GPU_LONG_RUN = "gpu_long_run"
RISK_DESTRUCTIVE = "destructive"

#: 风险等级中文说明（确认提示用）。
RISK_LABELS: dict[str, str] = {
    RISK_READ_ONLY: "只读查询",
    RISK_WRITE_FILES: "写入文件",
    RISK_COST_API: "消耗 API 额度",
    RISK_GPU_LONG_RUN: "GPU 长任务",
    RISK_DESTRUCTIVE: "破坏性操作",
}


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
        requires_user_action: 是否因风险确认/授权被拒（TUI/GUI 据此提示用户）。
        duration_ms: 执行耗时（毫秒）。
    """

    name: str
    arguments: dict
    ok: bool
    exit_code: int | None
    summary: str
    details: str
    requires_user_action: bool = False
    duration_ms: int | None = None


ServiceHandler = Callable[[dict], ToolResult]


@dataclass
class _CliEntry:
    """一个 CLI 工具的注册信息。"""

    module: str
    entry: str
    schema: ToolSchema
    risk_level: str = RISK_READ_ONLY
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
        execution: CLI 工具执行方式——``subprocess``（默认，隔离性好，
                 评审 P2-1 推荐）或 ``inproc``（进程内快速路径，测试用）。
    """

    def __init__(
        self,
        *,
        dry_run: bool = False,
        executor: str = "agent",
        execution: str = "subprocess",
    ) -> None:
        self.dry_run = dry_run
        self.executor = executor
        self.execution = execution
        self._cli_entries: dict[str, _CliEntry] = {}
        self._service_entries: dict[str, _ServiceEntry] = {}
        self.confirmed_tools: set[str] = set()

    # ---- 注册 ----

    def register_cli(
        self,
        module: str,
        *,
        entry: str = "main",
        risk_level: str = RISK_READ_ONLY,
        excluded_args: tuple[str, ...] = (),
        dry_run_flag: bool = False,
    ) -> None:
        """注册一个 CLI 模块为工具（走 ``module.main(argv)`` 进程内调用）。

        Args:
            module:         CLI 模块路径。
            entry:          入口函数名（默认 main）。
            risk_level:     风险等级（read_only 自动执行，其余需确认）。
            excluded_args:  从 schema 排除的参数。
            dry_run_flag:   安全模式下自动追加 --dry-run。
        """
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
            risk_level=risk_level,
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

    def risk_level(self, name: str) -> str:
        """返回工具的风险等级（未注册返回 read_only）。"""
        entry = self._cli_entries.get(name)
        return entry.risk_level if entry is not None else RISK_READ_ONLY

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
        started = time.perf_counter()
        if name in self._service_entries:
            entry = self._service_entries[name]
            try:
                result = entry.handler(dict(arguments))
            except Exception as exc:  # noqa: BLE001 - 边界层统一转为失败结果
                logger.exception("服务工具 %s 执行异常", name)
                result = ToolResult(
                    name=name,
                    arguments=safe_arguments,
                    ok=False,
                    exit_code=None,
                    summary=f"工具 {name} 执行异常: {type(exc).__name__}: {exc}",
                    details="",
                )
            return _with_duration(result, started)

        entry = self._cli_entries.get(name)
        if entry is None:
            return _with_duration(
                ToolResult(
                    name=name,
                    arguments=safe_arguments,
                    ok=False,
                    exit_code=None,
                    summary=f"未知工具: {name}",
                    details="",
                ),
                started,
            )

        # 安全模式下带 dry-run 标志的工具自动放行（dry-run 无副作用）
        auto_approved = self.dry_run and entry.dry_run_flag
        if entry.risk_level != RISK_READ_ONLY and name not in self.confirmed_tools and not auto_approved:
            label = RISK_LABELS.get(entry.risk_level, entry.risk_level)
            return _with_duration(
                ToolResult(
                    name=name,
                    arguments=safe_arguments,
                    ok=False,
                    exit_code=None,
                    summary=f"工具 {name} 属于「{label}」风险等级，需要用户确认后才可执行，请先征得用户同意",
                    details="",
                    requires_user_action=True,
                ),
                started,
            )

        argv = _argv_from_arguments(entry.schema, arguments)
        if self.dry_run and entry.dry_run_flag:
            argv.append("--dry-run")

        if self.execution == "subprocess":
            output = _run_cli_subprocess(entry.module, argv)
        else:
            output = _run_cli(entry.module, entry.entry, argv, executor=self.executor)
        summary = _summarize_output(output.output)
        return _with_duration(
            ToolResult(
                name=name,
                arguments=safe_arguments,
                ok=output.exit_code == 0,
                exit_code=output.exit_code,
                summary=summary or f"工具 {name} 无输出，退出码 {output.exit_code}",
                details=output.details[:DETAILS_LIMIT],
            ),
            started,
        )

    def confirm(self, name: str) -> None:
        """确认执行某工具（交互 REPL 下用户二次确认）。"""
        self.confirmed_tools.add(name)


@dataclass(frozen=True)
class _CliOutput:
    exit_code: int
    output: str
    details: str


def _run_cli_subprocess(module: str, argv: list[str], *, timeout: float | None = None) -> _CliOutput:
    """子进程执行 CLI 模块（``<当前解释器> -m <module> <argv>``）。

    子进程继承当前环境的 PYTHONPATH 与 KMP_DUPLICATE_LIB_OK，
    与训练/推理等重状态模块完全隔离（评审 P2-1 推荐路径）。

    Args:
        module:  CLI 模块路径。
        argv:    命令行参数。
        timeout: 超时秒数；超时返回退出码 124。

    Returns:
        合并 stdout+stderr 的输出与退出码。
    """
    from od_platform.common import paths

    command = [sys.executable, "-m", module, *argv]
    # 继承当前进程的平台 src 路径（开发/测试环境经 sys.path 注入时子进程同样可导入）
    env = os.environ.copy()
    platform_src = [path for path in sys.path if path.replace("\\", "/").endswith("apps/platform/src")]
    existing_pythonpath = env.get("PYTHONPATH", "")
    additions = [path for path in platform_src if path not in existing_pythonpath]
    if additions:
        env["PYTHONPATH"] = os.pathsep.join([*additions, existing_pythonpath] if existing_pythonpath else additions)

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=paths.ROOT_DIR,
            env=env,
        )
        exit_code = proc.returncode
        output = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        output = f"子进程超时（>{timeout}s）: {exc.stdout or ''}"
    except OSError as exc:
        exit_code = 2
        output = f"子进程启动失败: {type(exc).__name__}: {exc}"
    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.debug("子进程 %s 执行 %d ms，退出码 %d", module, duration_ms, exit_code)
    return _CliOutput(exit_code=exit_code, output=output, details=output)


def _run_cli(module: str, entry: str, argv: list[str], *, executor: str) -> _CliOutput:
    """进程内调用 CLI 模块入口，捕获日志输出与退出码。

    捕获采用组合方案（覆盖全部日志路径）:
    1. 临时替换 ``sys.stdout`` 为 capture —— 调用后才创建的
       ``StreamHandler(sys.stdout)``（如 configure_run_logger 的 run logger，
       propagate=False）会绑定新 stdout，日志进 capture；
    2. 遍历 ``loggerDict`` 给已存在的 propagate=False logger 补挂 capture
       handler —— 覆盖会话早期创建、handler 已绑定旧 stdout 的命名 logger；
    3. finally 恢复 sys.stdout 并移除全部补挂 handler。

    进程内多库（torch/matplotlib）同载 libiomp5md.dll 会触发 OMP Error #15，
    与 CLI 入口一致设置 KMP_DUPLICATE_LIB_OK（无害）。
    """
    from od_platform.common.environment import set_kmp_duplicate_lib_ok

    set_kmp_duplicate_lib_ok()
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

    previous_stdout = sys.stdout
    sys.stdout = capture  # type: ignore[assignment] - 覆盖运行期创建的 StreamHandler(sys.stdout)
    attached: list[logging.Logger] = []
    try:
        for candidate in logging.Logger.manager.loggerDict.values():
            if not isinstance(candidate, logging.Logger) or candidate is root_logger:
                continue
            if not candidate.propagate and handler not in candidate.handlers:
                candidate.addHandler(handler)
                attached.append(candidate)
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
        sys.stdout = previous_stdout
        for candidate in attached:
            candidate.removeHandler(handler)
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
    output = capture.getvalue()
    return _CliOutput(exit_code=code, output=output, details=output)


def _summarize_output(output: str, *, limit: int = SUMMARY_LIMIT) -> str:
    """将捕获输出整理为 Agent 观察文本。

    处理策略（失败归因优先）:
    1. ERROR/WARNING 行提取置顶；
    2. 过滤 get_logger 启动 banner 行（Logging Ready/runtime:/log type: 等）；
    3. 剩余行取尾部 N 行（保留最新进展）后截断到 limit。
    """
    if not output:
        return ""
    lines = output.splitlines()
    banner_markers = (
        "Logging Ready",
        "runtime:",
        "log type:",
        "log file:",
        "log level:",
        "model name:",
        "环境信息快照",
    )
    errors: list[str] = []
    body: list[str] = []
    for line in lines:
        # 剥 ANSI 颜色码（控制台 formatter 输出）
        stripped = _ANSI_RE.sub("", line).strip()
        if any(marker in stripped for marker in banner_markers) or stripped.startswith("=" * 10):
            continue
        if stripped.startswith(("ERROR", "WARNING", "错误", "警告")) or " [ERROR" in stripped or " [WARNING" in stripped:
            errors.append(stripped)
        else:
            body.append(stripped)

    tail = body[-150:]
    parts: list[str] = []
    if errors:
        parts.append("关键错误/警告:")
        parts.extend(errors[:30])
        parts.append("---")
    parts.extend(tail)
    summary = "\n".join(parts).strip()
    if len(summary) > limit:
        summary = summary[:limit] + "\n...(已截断)"
    return summary


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


def _artifact_summary(run_dir: Path) -> str:
    """提取一个运行目录的关键指标摘要（训练/评估/质检）。"""
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        from od_platform.training.metrics import summarize_results_csv

        try:
            summary = summarize_results_csv(results_csv)
            last = summary.get("last") or {}
            epochs = summary.get("epochs", "?")
            map50 = last.get("map50")
            map50_95 = last.get("map50_95")
            if isinstance(map50, (int, float)):
                if isinstance(map50_95, (int, float)):
                    return f"训练 {epochs} 轮, mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
                return f"训练 {epochs} 轮, mAP50={map50:.4f}"
            return f"训练 {epochs} 轮"
        except (OSError, ValueError, IndexError, TypeError):
            return "results.csv 读取失败"

    audit = run_dir / "odp_audit.json"
    if audit.exists():
        try:
            payload = json.loads(audit.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            map50 = metrics.get("map50") or metrics.get("map50_95")
            return f"评估 mAP={map50:.4f}" if isinstance(map50, (int, float)) else "评估完成"
        except (OSError, ValueError, json.JSONDecodeError):
            return "评估审计读取失败"

    report = run_dir / "report.json"
    if report.exists():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            return f"质检 {payload.get('overall_severity', '?')}"
        except (OSError, ValueError, json.JSONDecodeError):
            return "质检报告读取失败"

    return "无指标文件"


def _with_duration(result: ToolResult, started: float) -> ToolResult:
    """填充执行耗时（毫秒）。"""
    duration_ms = int((time.perf_counter() - started) * 1000)
    return ToolResult(
        name=result.name,
        arguments=result.arguments,
        ok=result.ok,
        exit_code=result.exit_code,
        summary=result.summary,
        details=result.details,
        requires_user_action=result.requires_user_action,
        duration_ms=duration_ms,
    )


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
        "list_run_artifacts",
        description="列出最近运行产物（训练/评估/质检）的目录与关键指标，训练或评估后用于检查结果",
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "enum": ["train", "evaluate", "validate"],
                    "description": "产物类型过滤：train/evaluate/validate（默认全部）",
                },
                "limit": {"type": "integer", "description": "每类最近 N 个（默认 5）"},
            },
        },
    )
    def _list_run_artifacts(arguments: dict) -> ToolResult:
        from od_platform.common import paths

        task = arguments.get("task") or None
        limit = int(arguments.get("limit") or 5)
        scan_roots: list[tuple[str, Path]] = []
        if task in (None, "train"):
            scan_roots.append(("train", paths.RUNS_DIR / "detect"))
        if task in (None, "evaluate"):
            scan_roots.append(("evaluate", paths.RUNS_DIR / "evaluation"))
        if task in (None, "validate"):
            scan_roots.append(("validate", paths.RUNS_DIR / "data_validation"))

        lines: list[str] = []
        for label, root in scan_roots:
            if not root.exists():
                continue
            candidates = [path for path in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if path.is_dir()][:limit]
            if not candidates:
                continue
            lines.append(f"[{label}]")
            for run_dir in candidates:
                lines.append(f"  {run_dir.name}: {_artifact_summary(run_dir)}")
        return ToolResult(
            "list_run_artifacts",
            {"task": task, "limit": limit},
            True,
            None,
            "\n".join(lines) if lines else "没有找到运行产物",
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
    ("od_platform.cli.import_dataset", {"risk_level": RISK_WRITE_FILES}),
    ("od_platform.cli.transform_data", {"risk_level": RISK_WRITE_FILES}),
    ("od_platform.cli.validate_data", {"risk_level": RISK_WRITE_FILES}),
    ("od_platform.runtime_config.generator", {"risk_level": RISK_WRITE_FILES}),
    ("od_platform.cli.train_model", {"risk_level": RISK_GPU_LONG_RUN, "dry_run_flag": True}),
    ("od_platform.cli.evaluate_model", {"risk_level": RISK_GPU_LONG_RUN}),
    ("od_platform.cli.infer_model", {"risk_level": RISK_GPU_LONG_RUN, "dry_run_flag": True}),
    ("od_platform.cli.plot_training", {"risk_level": RISK_WRITE_FILES}),
    (
        "od_platform.annotation.cli.auto_annotate",
        {"risk_level": RISK_COST_API, "excluded_args": ("--api-key", "--base-url")},
    ),
]
