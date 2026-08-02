# ODPlatform（目标检测开发平台）

ODPlatform 是通用目标检测开发平台（桌面工作台 ODPlatform Studio），覆盖从数据导入、质检、标注（手动/VLM 自动）、模型训练（YOLO v5/v7/v8/v9/v10/11/12）到推理评估的端到端工作流，并支持 AI 助手自然语言驱动（Agent 工具编排）。当前端到端主线只支持 YOLO `detect` 任务，`segment` 保留为未来扩展，不作为本阶段可交付链路。

内部 Python 包名保持为 `od_platform`，仓库根目录为 `ODPlatform/`。

## 快速安装

```powershell
conda activate odplat
python -c "import sys; print(sys.executable)"
pip install -r requirements.txt
pip install -e ./apps/platform
```

`requirements.txt` 是 GitHub 快速复现入口，由 `pipreqs` 扫描候选并合并 `apps/platform/pyproject.toml`、`apps/desktop/requirements.txt` 得到。平台包依赖的权威声明仍是 `apps/platform/pyproject.toml`。

主要 CLI 会在启动时自检当前解释器是否来自 `odplat`，如果不是只给 warning，不阻断运行。

## 示例数据集流程（Steel）

```powershell
# 导入 Roboflow VOC zip
odp-import-dataset "..\steel surface defect.v1i.voc.zip" --name steel-surface-defect --format voc --overwrite

# 转换、划分、落盘并生成 dataset yaml
odp-transform --dataset steel-surface-defect --format pascal_voc --task detect

# 数据质检
odp-validate --dataset steel-surface-defect --executor your-name

# 训练
odp-train --yaml train --data steel-surface-defect --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --device 0 --workers 4 --name steel-defect-yolo11n

# 模型评估
odp-val --config val --model models/trained/<best-dir>/best.pt --data steel-surface-defect --device 0

# 图片目录推理
odp-infer --model models/trained/<best-dir>/best.pt --source data/processed/steel-surface-defect/test/images --conf 0.25 --device 0 --name steel-defect-test --threaded

# 摄像头实时推理
odp-infer --model models/trained/<best-dir>/best.pt --source 0 --show --conf 0.25 --device 0 --name steel-camera-demo --threaded
```

`apps/platform/configs/runtime/infer_pipeline.yaml` 是本地 runtime 配置，默认包含 steel 六类中文标签映射：裂纹、夹杂、斑块、点蚀表面、轧制氧化皮、划痕。

## YOLO 数据集流程：liftrace

`liftrace.zip` 是 YOLO 格式数据集，结构包含 `data.yaml`、`images/<split>/` 和 `labels/<split>/`。导入时应使用 `--format yolo`，不要按 VOC 导入。

```powershell
odp-import-dataset "..\liftrace.zip" --name liftrace --format yolo --overwrite
odp-transform --dataset liftrace --format yolo --task detect
odp-validate --dataset liftrace --executor your-name
```

YOLO zip 导入会复制 `data.yaml`，并把 `images/*` 与 `labels/*` 展平成 `data/raw/<dataset>/images` 和 `data/raw/<dataset>/annotations`。转换时如果没有传 `--classes`，会优先从 raw 数据集根目录的 `data.yaml` 读取类别。

## CLI 一览

| 命令 | 用途 |
| --- | --- |
| `odp-init` | 初始化运行目录 |
| `odp-reset` | 清理运行产物，默认 dry-run；可加 `--backup` 先备份到 `runs/reset_backup/<timestamp>/` |
| `odp-import-dataset` | 导入真实数据集 zip，当前支持 VOC zip 和 YOLO zip |
| `odp-transform` | 数据格式转换、划分、落盘、生成 YOLO dataset yaml，并写入 `odp_meta.fingerprint` |
| `odp-validate` | 数据质检，输出 JSON、Markdown、HTML、CSV/Excel 返工清单 |
| `odp-gen-config` | 生成 train / val / infer runtime 配置 |
| `odp-train` | YOLO 训练、日志、manifest、训练曲线、权重归档 |
| `odp-val` | YOLO 模型评估，不归档权重 |
| `odp-infer` | 图片、目录、视频、摄像头推理，支持 D8 多级流水线 |
| `odp-plot-training` | 绘制训练曲线并导出指标摘要 |
| `odp-list-models` | 列出内置 YOLO 全系列模型（v5/v7/v8/v9/v10/11/12），支持自然语言推荐 |
| `odp-annotate` | 交互式边界框标注（OpenCV），`--edit` 编辑已有标注 |
| `odp-auto-annotate` | VLM 自动标注（Qwen-VL / GLM-4.5V 等 OpenAI 兼容端点），输出审计报告与复核队列 |
| `odp-agent` | AI 助手：自然语言驱动平台任务，支持 REPL / TUI / 会话恢复 / 任务模板 |

## AI 助手（Agent）

```powershell
# 交互模式（REPL，多轮记忆 + 斜杠命令）
odp-agent --base-url https://api.deepseek.com/v1 --model deepseek-chat

# TUI 界面（需 pip install textual）
odp-agent --tui --base-url https://api.deepseek.com/v1 --model deepseek-chat

# 按任务模板执行（准备数据集 / 训练并评估 / 自动标注复核）
odp-agent --template train_and_evaluate --task "用 yolo11n 训练 liftrace 50 轮" \
  --base-url https://api.deepseek.com/v1 --model deepseek-chat
```

Agent 以自然语言控制层调用既有 CLI/服务（不复制底层逻辑），支持：多轮会话（`runs/agent_sessions/`）、历史恢复（`--session`）、工具风险分级（写文件/消耗 API/GPU 长任务需 `/confirm`）、子进程隔离执行、任务模板、VLM 自动标注闭环（`runs/annotation/<run_id>/` 审计报告 + 复核队列 + 桌面复核页）。

## 模块边界

- `common/`：路径、日志、环境自检、计时、注册表、资源引用解析、系统信息。
- `data_pipeline/`：VOC / COCO / YOLO 转换、数据划分、落盘、dataset yaml 与指纹。
- `data_validation/`：snapshot 缓存、检查项注册、调度、报告和返工清单。
- `runtime_config/`：训练、评估、推理配置模型，支持默认值、YAML、CLI 合并。
- `training/`：训练编排、审计、模型归档、训练曲线。
- `evaluation/`：D7 模型评估服务与 `odp-val`。
- `frame_source/`：图片、图片目录、视频、摄像头统一帧源。
- `visualization/`：中文标签、美化框、Pillow 文本渲染。
- `inference/`：D8 推理服务、hook、sink、pause/cancel、多级流水线。
- `model_catalog/`：YOLO 全系列模型元数据、自然语言推荐、用户扩展（configs/models.yaml）。
- `annotation/`：手动标注画布、YOLO 读写、VLM 自动标注（提示词/JSON 容错/审计/复核队列）。
- `agent/`：OpenAI 兼容客户端、工具注册（单一清单 manifest）、风险分级、会话持久化、ReAct 编排、TUI。
- `apps/desktop/`：PySide6 工作台（推理/评估/质检/训练/模型目录/AI 助手/标注复核/任务启动），只调用 platform 服务层和 CLI，不复制业务逻辑。

## 桌面端

```powershell
conda activate odplat
python apps/desktop/main.py
```

桌面端（ODPlatform Studio）当前支持：图片/视频/摄像头推理、模型评估/数据质检/训练结果浏览、模型目录（34 模型推荐与一键应用）、AI 助手对话页（多轮会话/历史恢复）、标注复核页（VLM 审计队列精修）、任务启动页（11 个入口，含 VLM 自动标注与 AI 任务）。训练默认 dry-run，避免误触发长任务。

## QA 与非 Web 收尾

`docs/qa/` 记录 edge case 审查和扩展项要求。本阶段暂不推进 Web，已收敛的非 Web 事项包括：

- 根目录 `requirements.txt` 快速安装入口。
- 非 `odplat` 环境自检 warning。
- 当前产品边界明确为 `detect`。
- `odp-reset --backup` 可选备份。
- dataset yaml 指纹输出。
- YAML schema、pipeline config、frame source、desktop task worker 等补充测试。

## 参考资产说明

根目录的 `inference.zip`、`visualization.zip`、`data_pipeline*.zip`、单个 `.py` 脚本和 HTML/Markdown 文档是带教参考资产，用于对照实现。`odp-import-dataset` 只用于导入真实数据集压缩包，例如 `steel surface defect.v1i.voc.zip`、`liftrace.zip`。

## 验证

```powershell
conda activate odplat
python -m ruff check apps/platform/src/od_platform apps/desktop scripts
python -m pytest apps/platform/src/od_platform/tests -q
python -m compileall apps/desktop
```
