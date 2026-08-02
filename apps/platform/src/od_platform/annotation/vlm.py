"""VLM 自动标注：自然语言指令批量生成 YOLO 标注。

数据流：图片 → base64 多模态消息 → VLM 返回 JSON → 校验转 BBox →
``AnnotationSession.save_labels`` 落盘（断点续跑）。

与手动标注形成"VLM 预标注 + 人工精修"闭环：VLM 产物与手动产物
格式完全一致（data/raw/<dataset>/annotations/ 下 YOLO txt），
精修时用 canvas 编辑模式 ``annotate_image(existing=...)``。

VLM 客户端复用 ``agent.client.OpenAIClient``（OpenAI 兼容端点，
支持 Qwen-VL / GLM-4.5V 等视觉模型）。

@FileName:   vlm.py
@Function:   提示词构造、JSON 容错解析、单图标注与批量运行
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from od_platform.agent.client import APIError, OpenAIClient, image_url_content_part
from od_platform.annotation.session import AnnotationSession
from od_platform.annotation.writer import BBox
from od_platform.common import paths

logger = logging.getLogger(__name__)


class VLMError(Exception):
    """VLM 标注错误（无效 JSON 重试耗尽等）。"""


@dataclass(frozen=True)
class VLMConfig:
    """VLM 标注配置。

    Attributes:
        model:    视觉模型名（调用方强制传入，如 qwen-vl-max / glm-4.5v-turbo）。
        api_key:  API 密钥；缺省读环境变量 OPENAI_API_KEY。
        base_url: OpenAI 兼容基地址（调用方强制传入）。
        prompt:   自然语言标注指令，如 "框出所有飞机"。
        max_boxes: 单图最大框数上限（防异常输出）。
        retries:  无效 JSON 时的重试次数。
        timeout:  单次请求超时（秒）。
    """

    model: str
    api_key: str | None = None
    base_url: str | None = None
    prompt: str = "框出所有目标"
    max_boxes: int = 100
    retries: int = 2
    timeout: float = 120.0


@dataclass(frozen=True)
class VLMAnnotationReport:
    """一次批量标注的运行报告。

    Attributes:
        dataset:       数据集名称。
        total_images:  总图片数。
        annotated_new: 本次新增标注数。
        failed:        失败图片文件名列表。
        box_count:     本次标注框总数。
        audit_dir:     本次审计产物目录（runs/annotation/<run_id>/）。
    """

    dataset: str
    total_images: int
    annotated_new: int
    failed: list[str] = field(default_factory=list)
    box_count: int = 0
    audit_dir: Path | None = None


def build_annotation_prompt(*, classes: list[str], user_prompt: str) -> str:
    """构造 VLM 标注系统提示词。

    类别清单嵌入（索引即 class_id），要求严格 JSON 输出与归一化坐标，
    与训练流水线的 YOLO 格式契约一致。
    """
    class_lines = "\n".join(f"{index}: {name}" for index, name in enumerate(classes))
    return (
        "你是目标检测数据集标注助手。只输出机器可解析的 JSON，不要输出任何其他文字。\n"
        "\n"
        f"类别清单（索引即类别 ID，只能使用这些类别）:\n{class_lines}\n"
        "\n"
        f"任务描述:\n{user_prompt}\n"
        "\n"
        "输出 JSON 格式（必须严格符合）:\n"
        '{"boxes": [{"class_id": 0, "x_center": 0.50, "y_center": 0.40, "width": 0.20, "height": 0.30}]}\n'
        "\n"
        "约束:\n"
        "- boxes 为空数组表示图片中没有目标。\n"
        "- class_id 必须是整数且来自上面的类别清单。\n"
        "- x_center / y_center 是框中心点，width / height 是框的宽高，均为 0 到 1 的归一化值。\n"
        "- 每个目标一个框；同一类别多个目标输出多个框；不要输出清单之外的类别。\n"
        "- 只输出一个 JSON 对象，禁止 Markdown 代码块、解释或额外字段。"
    )


def extract_json_object(text: str) -> dict | None:
    """从 VLM 回复中容错提取 JSON 对象。

    剥 `` ```json `` 围栏 → 定位第一个平衡的 ``{...}`` → json.loads。

    Args:
        text: VLM 返回文本。

    Returns:
        解析出的 dict；失败返回 None。
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def parse_vlm_boxes(
    payload: dict,
    *,
    classes: list[str],
    max_boxes: int = 100,
) -> tuple[list[BBox], int]:
    """校验并转换 VLM 输出的 boxes 为 BBox 列表。

    Args:
        payload:    VLM 返回的 JSON dict。
        classes:    类别名称列表（索引即 class_id）。
        max_boxes:  单图最大框数上限（防异常输出）。

    Returns:
        (有效 BBox 列表, 被丢弃条目数)。非法条目（class_id 越界/非数值/
        宽高 <= 0）被丢弃，坐标裁剪到 [0, 1]。
    """
    raw_boxes = payload.get("boxes")
    if not isinstance(raw_boxes, list):
        return [], 0

    boxes: list[BBox] = []
    discarded = 0
    for raw in raw_boxes[:max_boxes]:
        if not isinstance(raw, dict):
            discarded += 1
            continue
        try:
            class_id = int(raw["class_id"])
            x_center = _clamp01(float(raw["x_center"]))
            y_center = _clamp01(float(raw["y_center"]))
            width = _clamp01(float(raw["width"]))
            height = _clamp01(float(raw["height"]))
        except (KeyError, TypeError, ValueError):
            discarded += 1
            continue
        box = BBox(class_id=class_id, x_center=x_center, y_center=y_center, width=width, height=height)
        if not box.is_valid:
            discarded += 1
            continue
        if class_id < 0 or class_id >= len(classes):
            discarded += 1
            continue
        boxes.append(box)
    return boxes, discarded


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class VLMAnnotationResult:
    """单图 VLM 标注结果（含审计信息）。

    Attributes:
        boxes:        有效 BBox 列表（空列表表示图片无目标）。
        raw_response: VLM 原始响应文本（审计保存用）。
        discarded:    被丢弃的非法框数量。
        latency_ms:   单图请求耗时（毫秒）。
    """

    boxes: list[BBox]
    raw_response: str
    discarded: int = 0
    latency_ms: int = 0


def annotate_image_with_vlm(
    client: OpenAIClient,
    image_path: Path,
    *,
    config: VLMConfig,
    classes: list[str],
) -> VLMAnnotationResult:
    """单图 VLM 标注。

    Args:
        client:     OpenAI 兼容客户端。
        image_path: 图片路径。
        config:     VLM 配置。
        classes:    类别名称列表。

    Returns:
        标注结果（含原始响应与丢弃计数）。

    Raises:
        VLMError: 无效 JSON 重试耗尽。
        APIError: API 调用失败（由上层跳过该图）。
    """
    system_prompt = build_annotation_prompt(classes=classes, user_prompt=config.prompt)
    retry_hint = "你上次的输出不是合法 JSON。请只输出一个 JSON 对象，不要包含任何其他内容。"

    last_text = ""
    started = time.perf_counter()
    for attempt in range(config.retries + 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    image_url_content_part(image_path),
                    {"type": "text", "text": config.prompt},
                ],
            },
        ]
        if attempt > 0:
            messages.append({"role": "user", "content": retry_hint})

        response = client.chat(messages, model=config.model, temperature=0.0)
        last_text = response["choices"][0]["message"].get("content") or ""
        payload = extract_json_object(last_text)
        if payload is not None:
            boxes, discarded = parse_vlm_boxes(payload, classes=classes, max_boxes=config.max_boxes)
            if discarded:
                logger.debug("图片 %s 丢弃 %d 条非法框", image_path.name, discarded)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return VLMAnnotationResult(
                boxes=boxes,
                raw_response=last_text,
                discarded=discarded,
                latency_ms=latency_ms,
            )
        logger.warning("图片 %s 第 %d 次返回非 JSON: %s", image_path.name, attempt + 1, last_text[:100])

    raise VLMError(f"图片 {image_path.name} 返回无效 JSON（重试 {config.retries} 次后仍失败）")


def run_vlm_annotation(
    *,
    dataset: str,
    classes: list[str],
    config: VLMConfig,
    images_dir: Path | None = None,
    labels_dir: Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> VLMAnnotationReport:
    """批量 VLM 标注数据集（断点续跑）。

    Args:
        dataset:    数据集名称（定位 data/raw/<dataset>/）。
        classes:    类别名称列表。
        config:     VLM 配置。
        images_dir: 图片目录；缺省 data/raw/<dataset>/images。
        labels_dir: 标注输出目录；缺省 data/raw/<dataset>/annotations。
        limit:      本次处理上限（None 表示全部未标注）。
        dry_run:    只统计剩余图片数，不调用 API。

    Returns:
        运行报告。单图失败（API/VLM 错误）记录到 failed 并继续。
    """
    raw_root = paths.RAW_DATA_DIR / dataset
    session = AnnotationSession(
        images_dir=images_dir or raw_root / "images",
        labels_dir=labels_dir or raw_root / "annotations",
        classes=classes,
    )

    pending = [image for image in session.images if not session.is_annotated(image.stem)]
    if limit is not None:
        pending = pending[:limit]

    if dry_run:
        logger.info("dry-run: 数据集 %s 共 %d 张，剩余 %d 张待标注", dataset, session.total_images, len(pending))
        return VLMAnnotationReport(dataset=dataset, total_images=session.total_images, annotated_new=0)

    if not pending:
        logger.info("数据集 %s 已全部标注（%d 张），无待处理图片", dataset, session.total_images)
        return VLMAnnotationReport(dataset=dataset, total_images=session.total_images, annotated_new=0)

    client = OpenAIClient(api_key=config.api_key, base_url=config.base_url)
    failed: list[str] = []
    annotated_new = 0
    box_count = 0
    discarded_total = 0
    latency_sum = 0
    review_rows: list[dict[str, str]] = []

    # 审计产物目录：runs/annotation/<run_id>/
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir = paths.RUNS_DIR / "annotation" / run_id
    raw_dir = audit_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始 VLM 标注数据集 %s: %d 张待标注，模型 %s", dataset, len(pending), config.model)
    for index, image_path in enumerate(pending, start=1):
        try:
            result = annotate_image_with_vlm(client, image_path, config=config, classes=classes)
            session.save_labels(image_path.stem, result.boxes)
            annotated_new += 1
            box_count += len(result.boxes)
            discarded_total += result.discarded
            latency_sum += result.latency_ms
            # 保存原始响应（审计）
            (raw_dir / f"{image_path.stem}.txt").write_text(result.raw_response, encoding="utf-8")
            # 复核队列：空标注（可能是背景图或漏检）与有丢弃框的样本
            if not result.boxes:
                review_rows.append(
                    {
                        "image": image_path.name,
                        "boxes": "0",
                        "discarded": str(result.discarded),
                        "reason": "empty",
                        "label_path": str(session.label_path_for(image_path.stem)),
                    }
                )
            elif result.discarded:
                review_rows.append(
                    {
                        "image": image_path.name,
                        "boxes": str(len(result.boxes)),
                        "discarded": str(result.discarded),
                        "reason": "discarded",
                        "label_path": str(session.label_path_for(image_path.stem)),
                    }
                )
            logger.info("[%d/%d] %s: %d 框", index, len(pending), image_path.name, len(result.boxes))
        except (APIError, VLMError) as exc:
            logger.warning("[%d/%d] %s 标注失败: %s", index, len(pending), image_path.name, exc)
            failed.append(image_path.name)
            review_rows.append(
                {
                    "image": image_path.name,
                    "boxes": "-",
                    "discarded": "-",
                    "reason": "failed",
                    "label_path": str(session.label_path_for(image_path.stem)),
                }
            )

    _write_audit_report(
        audit_dir=audit_dir,
        run_id=run_id,
        dataset=dataset,
        config=config,
        classes=classes,
        total_images=session.total_images,
        pending_count=len(pending),
        annotated_new=annotated_new,
        failed=failed,
        box_count=box_count,
        discarded_total=discarded_total,
        latency_sum=latency_sum,
        review_rows=review_rows,
    )

    logger.info(
        "VLM 标注完成: 新增 %d 张（共 %d 框，丢弃 %d），失败 %d 张；审计目录 %s",
        annotated_new,
        box_count,
        discarded_total,
        len(failed),
        audit_dir,
    )
    return VLMAnnotationReport(
        dataset=dataset,
        total_images=session.total_images,
        annotated_new=annotated_new,
        failed=failed,
        box_count=box_count,
        audit_dir=audit_dir,
    )


def _write_audit_report(
    *,
    audit_dir: Path,
    run_id: str,
    dataset: str,
    config: VLMConfig,
    classes: list[str],
    total_images: int,
    pending_count: int,
    annotated_new: int,
    failed: list[str],
    box_count: int,
    discarded_total: int,
    latency_sum: int,
    review_rows: list[dict[str, str]],
) -> None:
    """写入 annotation_report.json 与 review_queue.csv（审计产物）。"""
    report = {
        "run_id": run_id,
        "dataset": dataset,
        "model": config.model,
        "prompt": config.prompt,
        "class_names": classes,
        "image_count": total_images,
        "pending_count": pending_count,
        "annotated_count": annotated_new,
        "skipped_count": pending_count - annotated_new - len(failed),
        "failed_count": len(failed),
        "total_boxes": box_count,
        "discarded_boxes": discarded_total,
        "avg_latency_ms": round(latency_sum / annotated_new, 1) if annotated_new else 0,
        "review_queue": len(review_rows),
        "failed_images": failed,
        "audit_dir": str(audit_dir),
    }
    (audit_dir / "annotation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_path = audit_dir / "review_queue.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "boxes", "discarded", "reason", "label_path"])
        writer.writeheader()
        writer.writerows(review_rows)
    logger.debug("审计产物已写入: %s", audit_dir)
