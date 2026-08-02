"""自然语言任务模板（评审路线图阶段五）。

将常见流程沉淀为固定步骤，避免 LLM 每次自由发挥；
``odp-agent --template <name>`` 把模板步骤附加到用户消息。

@FileName:   templates.py
@Function:   任务模板定义与查询
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskTemplate:
    """一个任务模板。

    Attributes:
        name:        模板名。
        description: 一句话描述（/templates 列表展示）。
        steps:       执行步骤说明（附加给 LLM 的指导）。
    """

    name: str
    description: str
    steps: list[str]


TEMPLATES: dict[str, TaskTemplate] = {
    "prepare_dataset": TaskTemplate(
        name="prepare_dataset",
        description="导入数据集 zip 并转换为 YOLO，然后跑质检",
        steps=[
            "1. 用 list_datasets 确认现状，不重复导入",
            "2. odp-import-dataset 导入 zip（格式 yolo 或 voc）",
            "3. odp-transform 转换为 YOLO 并划分数据集",
            "4. odp-validate 检查数据集质量",
            "5. 返回质检报告路径与问题摘要",
        ],
    ),
    "train_and_evaluate": TaskTemplate(
        name="train_and_evaluate",
        description="用指定模型训练数据集，完成后评估并汇总指标",
        steps=[
            "1. 用 list_available_models / odp-list-models 确认模型",
            "2. 检查数据集配置（configs/datasets/<name>.yaml）存在",
            "3. 确认 GPU 长任务（用户确认或安全模式 dry-run）",
            "4. odp-train 训练（显式给出 epochs/batch/workers/imgsz）",
            "5. odp-val 评估",
            "6. list_run_artifacts 检查产物与指标",
            "7. 汇总 best.pt、mAP50/mAP50-95、曲线图路径",
        ],
    ),
    "annotate_and_review": TaskTemplate(
        name="annotate_and_review",
        description="用 VLM 自动标注未标注图片，并生成复核清单",
        steps=[
            "1. 确认 API 配置（环境变量 OPENAI_BASE_URL/OPENAI_API_KEY）",
            "2. 从 dataset yaml 读取类别（--classes-from-yaml）",
            "3. 确认 API 成本（cost_api 风险等级）",
            "4. odp-auto-annotate 自动标注（dry-run 预览后确认执行）",
            "5. 说明审计产物位置（runs/annotation/<run_id>/）与复核清单",
        ],
    ),
}


def list_templates() -> list[str]:
    """返回全部模板名。"""
    return list(TEMPLATES)


def get_template(name: str) -> TaskTemplate | None:
    """按名获取模板；不存在返回 None。"""
    return TEMPLATES.get(name)


def template_instruction(name: str) -> str:
    """生成附加到用户消息的模板步骤说明（无此模板返回空串）。"""
    template = TEMPLATES.get(name)
    if template is None:
        return ""
    lines = [f"请严格按以下步骤执行任务（模板: {template.name}）:", *template.steps]
    return "\n".join(lines)
