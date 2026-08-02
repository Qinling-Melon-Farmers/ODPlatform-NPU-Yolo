"""任务命令构建（纯函数，无 Qt 依赖，可直接单测）。

从桌面端任务启动页提取：接收任务名与参数字典，返回
``(CLI 模块路径, 参数列表)``；缺必填参数抛 ValueError（由 UI 层展示）。
"""

from __future__ import annotations

from typing import Any

#: 任务名 → 中文名（UI 下拉条目）。
TASK_NAMES: tuple[str, ...] = (
    "导入数据集",
    "数据转换",
    "数据质检",
    "模型评估",
    "模型训练",
    "项目重置",
    "训练曲线生成",
    "列出模型",
    "数据标注",
    "自动标注",
    "AI 任务",
)


def build_task_command(task_name: str, params: dict[str, Any]) -> tuple[str, list[str]]:
    """按任务名与参数构造 CLI 命令。

    Args:
        task_name: 任务中文名（与 UI 下拉一致）。
        params:    参数字典（键见各分支；缺省值由 CLI 处理，None 跳过）。

    Returns:
        (CLI 模块路径, argv)。

    Raises:
        ValueError: 缺必填参数或未知任务。
    """
    if task_name == "导入数据集":
        zip_path = _req(params, "zip_path", "导入数据集需要填写数据 zip")
        args = [zip_path]
        dataset = params.get("dataset")
        if dataset:
            args.extend(["--name", dataset])
        args.extend(["--format", params.get("format") or "voc"])
        if params.get("overwrite"):
            args.append("--overwrite")
        return "od_platform.cli.import_dataset", [*args, *_extra(params)]

    if task_name == "数据转换":
        dataset = _req(params, "dataset", "数据转换需要填写数据集名称")
        annotation_format = params.get("format") or "pascal_voc"
        if annotation_format == "voc":
            annotation_format = "pascal_voc"
        args = [
            "--dataset",
            dataset,
            "--format",
            annotation_format,
            "--task",
            params.get("task") or "detect",
        ]
        return "od_platform.cli.transform_data", [*args, *_extra(params)]

    if task_name == "数据质检":
        dataset = _req(params, "dataset", "数据质检需要填写数据集名称")
        args = ["--dataset", dataset, "--task", params.get("task") or "detect"]
        executor = params.get("executor")
        if executor:
            args.extend(["--executor", executor])
        return "od_platform.cli.validate_data", [*args, *_extra(params)]

    if task_name == "模型评估":
        model = _req(params, "model", "模型评估需要填写模型权重")
        dataset = _req(params, "dataset", "模型评估需要填写数据集名称")
        args = [
            "--config",
            params.get("config") or "val",
            "--model",
            model,
            "--data",
            dataset,
        ]
        _extend_optional(args, params, ("device", "executor", "name"))
        return "od_platform.cli.evaluate_model", [*args, *_extra(params)]

    if task_name == "模型训练":
        model = _req(params, "model", "模型训练需要填写模型权重或模型名")
        dataset = _req(params, "dataset", "模型训练需要填写数据集名称")
        args = [
            "--config",
            params.get("config") or "train",
            "--model",
            model,
            "--data",
            dataset,
            "--epochs",
            str(params.get("epochs") or 1),
            "--batch",
            str(params.get("batch") or 16),
            "--workers",
            str(params.get("workers") or 0),
        ]
        _extend_optional(args, params, ("device", "executor", "name"))
        if params.get("dry_run"):
            args.append("--dry-run")
        return "od_platform.cli.train_model", [*args, *_extra(params)]

    if task_name == "项目重置":
        args: list[str] = []
        for flag, option in (("dry_run", "--dry-run"), ("backup", "--backup"), ("yes", "--yes"), ("force", "--force")):
            if params.get(flag):
                args.append(option)
        return "od_platform.cli.reset_project", [*args, *_extra(params)]

    if task_name == "训练曲线生成":
        csv_path = _req(params, "csv_path", "训练曲线生成需要填写 results.csv")
        args = [csv_path]
        _extend_optional(args, params, ("output", "summary"))
        if params.get("matplotx"):
            args.append("--matplotx")
        return "od_platform.cli.plot_training", [*args, *_extra(params)]

    if task_name == "列出模型":
        args: list[str] = []
        family = params.get("family")
        if family:
            args.extend(["--family", family])
        recommend = params.get("recommend")
        if recommend:
            args.extend(["--recommend", recommend])
        args.extend(["--limit", str(params.get("limit") or 10)])
        if params.get("json"):
            args.append("--json")
        return "od_platform.model_catalog.cli.list_models", [*args, *_extra(params)]

    if task_name == "数据标注":
        dataset = _req(params, "dataset", "数据标注需要填写数据集名称")
        classes = params.get("classes") or []
        if not classes:
            raise ValueError("数据标注需要填写至少一个类别名")
        args = ["--dataset", dataset, "--classes", *classes]
        if params.get("resume"):
            args.append("--resume")
        if params.get("edit"):
            args.append("--edit")
        return "od_platform.annotation.cli.annotate", [*args, *_extra(params)]

    if task_name == "自动标注":
        dataset = _req(params, "dataset", "自动标注需要填写数据集名称")
        classes = params.get("classes") or []
        if not classes:
            raise ValueError("自动标注需要填写至少一个类别名")
        base_url = _req(params, "base_url", "自动标注需要填写 API 地址")
        model = _req(params, "model", "自动标注需要填写视觉模型名")
        args = [
            "--dataset",
            dataset,
            "--classes",
            *classes,
            "--prompt",
            params.get("prompt") or "框出所有目标",
            "--base-url",
            base_url,
            "--model",
            model,
        ]
        api_key = params.get("api_key")
        if api_key:
            args.extend(["--api-key", api_key])
        limit = params.get("limit")
        if limit:
            args.extend(["--limit", str(limit)])
        if params.get("dry_run"):
            args.append("--dry-run")
        return "od_platform.annotation.cli.auto_annotate", [*args, *_extra(params)]

    if task_name == "AI 任务":
        prompt = _req(params, "prompt", "AI 任务需要填写自然语言任务描述")
        base_url = _req(params, "base_url", "AI 任务需要填写 API 地址")
        model = _req(params, "model", "AI 任务需要填写模型名")
        args = ["--base-url", base_url, "--model", model, "--task", prompt]
        api_key = params.get("api_key")
        if api_key:
            args.extend(["--api-key", api_key])
        return "od_platform.agent.cli.agent_chat", [*args, *_extra(params)]

    raise ValueError(f"未知任务: {task_name}")


def _req(params: dict[str, Any], key: str, message: str) -> str:
    value = params.get(key)
    if not value:
        raise ValueError(message)
    return str(value)


def _extend_optional(args: list[str], params: dict[str, Any], keys: tuple[str, ...]) -> None:
    """追加可选单值参数（非空才加）。"""
    for key in keys:
        value = params.get(key)
        if value:
            args.extend([f"--{key}", str(value)])


def _extra(params: dict[str, Any]) -> list[str]:
    """追加 CLI 参数（用户自由输入）。"""
    return list(params.get("extra_args") or [])
