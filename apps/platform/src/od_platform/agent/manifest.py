"""Agent 工具单一清单（评审 P3：消除双清单维护）。

新增 CLI 工具只需在此登记一条；``schema.build_all_tool_schemas`` 与
``tools.build_default_registry`` 均从本清单派生，避免"schema 能生成
但 Agent 没注册"或"注册了但文档没同步"的漂移。

服务层工具（无 CLI 对应，如 list_datasets）仍在 tools.py 注册，
不在此清单。
"""

from __future__ import annotations

from dataclasses import dataclass

from od_platform.agent.tools import (
    RISK_COST_API,
    RISK_GPU_LONG_RUN,
    RISK_WRITE_FILES,
)


@dataclass(frozen=True)
class ToolManifestEntry:
    """一个 CLI 工具的登记条目。

    Attributes:
        module:        CLI 模块路径（如 ``od_platform.cli.train_model``）。
        risk_level:    风险等级（tools 常量）。
        excluded_args: 从 schema 排除的参数（凭据等）。
        dry_run_flag:  安全模式下自动追加 --dry-run。
        entry:         入口函数名（默认 main）。
    """

    module: str
    risk_level: str
    excluded_args: tuple[str, ...] = ()
    dry_run_flag: bool = False
    entry: str = "main"


#: 平台 CLI 工具单一清单（新增 CLI 在此登记）。
TOOL_MANIFEST: tuple[ToolManifestEntry, ...] = (
    ToolManifestEntry("od_platform.cli.import_dataset", RISK_WRITE_FILES),
    ToolManifestEntry("od_platform.cli.transform_data", RISK_WRITE_FILES),
    ToolManifestEntry("od_platform.cli.validate_data", RISK_WRITE_FILES),
    ToolManifestEntry("od_platform.runtime_config.generator", RISK_WRITE_FILES),
    ToolManifestEntry("od_platform.cli.train_model", RISK_GPU_LONG_RUN, dry_run_flag=True),
    ToolManifestEntry("od_platform.cli.evaluate_model", RISK_GPU_LONG_RUN),
    ToolManifestEntry("od_platform.cli.infer_model", RISK_GPU_LONG_RUN, dry_run_flag=True),
    ToolManifestEntry("od_platform.cli.plot_training", RISK_WRITE_FILES),
    ToolManifestEntry(
        "od_platform.annotation.cli.auto_annotate",
        RISK_COST_API,
        excluded_args=("--api-key", "--base-url"),
    ),
)


def manifest_modules() -> list[str]:
    """返回清单中全部 CLI 模块路径（供 schema 批量自省）。"""
    return [entry.module for entry in TOOL_MANIFEST]
