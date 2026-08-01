"""CLI argparse → OpenAI function-calling tool schema 自省。

通过 ``build_parser()`` 返回的 ArgumentParser 自省生成 tool schema，
保证 schema 与 CLI 参数定义同步；``ArgSpec`` 记录参数元信息，
供 tools.py 从 LLM 返回的 arguments 反向构造 argv。

排除规则：
- prog 以 ``odp-agent`` 开头（防递归暴露自身）
- 控制字段：--config / --executor / --dry-run / --log-level
- help / version 系统动作

@FileName:   schema.py
@Function:   argparse 自省与 tool schema 生成
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass, field
from typing import Any

#: 排除的控制字段 dest（agent 自行管理，不暴露给 LLM）。
CONTROL_DESTS: frozenset[str] = frozenset({"config", "executor", "dry_run", "log_level", "verbose"})

#: 类型 → JSON Schema type 映射。
_TYPE_MAP: dict[str, str] = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "Path": "string",
}


@dataclass(frozen=True)
class ArgSpec:
    """一个 argparse 参数的元信息（供反向构造 argv）。"""

    key: str
    option_strings: tuple[str, ...]
    action: str
    nargs: str | int | None
    required: bool
    is_positional: bool
    positional_index: int | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""

    @property
    def option(self) -> str | None:
        """首选长选项名（如 ``--epochs``），无则返回 None。"""
        for option in self.option_strings:
            if option.startswith("--"):
                return option
        return self.option_strings[0] if self.option_strings else None


@dataclass(frozen=True)
class ToolSchema:
    """一个 CLI 对应的工具定义。"""

    name: str
    description: str
    parameters: dict
    arg_specs: list[ArgSpec] = field(default_factory=list)


def parser_to_tool_schema(parser: argparse.ArgumentParser) -> ToolSchema | None:
    """将 argparse 解析器自省为 ToolSchema。

    Args:
        parser: 目标 CLI 的解析器（``build_parser()`` 返回值）。

    Returns:
        ToolSchema；prog 以 ``odp-agent`` 开头时返回 None。
    """
    prog = parser.prog or ""
    if prog.startswith("odp-agent"):
        return None

    properties: dict[str, dict] = {}
    required: list[str] = []
    arg_specs: list[ArgSpec] = []
    positional_index = 0

    for action in parser._actions:  # noqa: SLF001 - argparse 自省需要内部结构
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)):  # noqa: SLF001
            continue
        dest = action.dest
        if dest in CONTROL_DESTS:
            continue

        is_positional = not action.option_strings
        if is_positional:
            key = dest
        else:
            key = _preferred_key(action, dest)

        spec = _build_arg_spec(action, key, is_positional, positional_index)
        if spec is None:
            continue
        if is_positional:
            positional_index += 1

        property_schema = _property_schema(action)
        if not property_schema:
            continue

        properties[key] = property_schema
        arg_specs.append(spec)
        if spec.required:
            required.append(key)

    if not properties:
        return None

    parameters: dict = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return ToolSchema(
        name=prog,
        description=parser.description or f"执行 {prog}",
        parameters=parameters,
        arg_specs=arg_specs,
    )


def build_all_tool_schemas(modules: list[str] | None = None) -> list[ToolSchema]:
    """构建一批 CLI 模块的 tool schema。

    Args:
        modules: 模块路径列表（如 ``od_platform.cli.train_model``）；
                缺省使用默认 CLI 清单。

    Returns:
        ToolSchema 列表（跳过无法导入或返回 None 的模块）。
    """
    schemas: list[ToolSchema] = []
    for module_name in modules or DEFAULT_CLI_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        build_parser = getattr(module, "build_parser", None)
        if build_parser is None:
            continue
        schema = parser_to_tool_schema(build_parser())
        if schema is not None:
            schemas.append(schema)
    return schemas


def _preferred_key(action: argparse.Action, dest: str) -> str:
    """确定参数在 schema 中的属性名。

    优先用长选项名（去 ``--``、横线转下划线），使 schema 属性名与
    CLI 参数一致（如 ``--format`` 的 dest 是 ``annotation_format``，
    但属性名应为 ``format``）；无长选项时回退 dest。
    """
    for option in action.option_strings:
        if option.startswith("--"):
            return option[2:].replace("-", "_")
    return dest


def _build_arg_spec(
    action: argparse.Action,
    key: str,
    is_positional: bool,
    positional_index: int,
) -> ArgSpec | None:
    """从 action 构造 ArgSpec，无法映射时返回 None。"""
    if action.required or (is_positional and action.nargs != argparse.OPTIONAL):
        required = True
    else:
        required = False

    choices: tuple[Any, ...] | None = None
    if action.choices is not None:
        choices = tuple(action.choices)

    description = action.help or ""
    if description and description != argparse.SUPPRESS:
        if action.default is not None and action.default is not argparse.SUPPRESS:
            description += f"（默认: {action.default}）"
    else:
        description = ""

    return ArgSpec(
        key=key,
        option_strings=tuple(action.option_strings),
        action=type(action).__name__,
        nargs=action.nargs,
        required=required,
        is_positional=is_positional,
        positional_index=positional_index if is_positional else None,
        choices=choices,
        description=description,
    )


def _property_schema(action: argparse.Action) -> dict | None:
    """构造单个参数的 JSON Schema 属性。"""
    type_name = _type_name(action.type)
    json_type = _TYPE_MAP.get(type_name, "string")

    property_schema: dict[str, Any] = {}
    if action.nargs == "+" or action.nargs == "*":
        property_schema["type"] = "array"
        property_schema["items"] = {"type": json_type}
    else:
        property_schema["type"] = json_type

    if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
        property_schema["type"] = "boolean"
        property_schema["description"] = (action.help or "").replace("--", "")
    if isinstance(action, argparse._StoreFalseAction):  # noqa: SLF001
        property_schema["type"] = "boolean"

    if action.choices is not None:
        property_schema["enum"] = [str(choice) for choice in action.choices]

    if action.help and action.help != argparse.SUPPRESS:
        property_schema["description"] = action.help
    return property_schema


def _type_name(type_func: Any) -> str:
    """解析 argparse type callable 的类型名。"""
    if type_func is None:
        return "str"
    name = getattr(type_func, "__name__", None)
    if name:
        return name
    return type(type_func).__name__


#: 默认注册的 CLI 模块清单（tools.py 的默认注册表与此一致）。
DEFAULT_CLI_MODULES: list[str] = [
    "od_platform.cli.init_project",
    "od_platform.cli.import_dataset",
    "od_platform.cli.transform_data",
    "od_platform.cli.validate_data",
    "od_platform.runtime_config.generator",
    "od_platform.cli.train_model",
    "od_platform.cli.evaluate_model",
    "od_platform.cli.infer_model",
    "od_platform.cli.plot_training",
    "od_platform.model_catalog.cli.list_models",
    "od_platform.annotation.cli.auto_annotate",
]
