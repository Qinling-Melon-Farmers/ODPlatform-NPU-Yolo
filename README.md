# ODPlatform

ODPlatform 是生产实习阶段构建的目标检测开发平台。当前主线围绕 steel surface defect 数据集完成数据导入、格式转换、数据质检、训练、模型评估、推理、训练曲线、推理结果美化和桌面端推理演示。

## 当前状态

- 仓库根目录：`ODPlatform`
- Python 包：`apps/platform/src/od_platform`
- 开发环境：统一使用 `odplat` Conda 环境
- 安装方式：`pip install -e ./apps/platform`
- 当前数据集：`steel-surface-defect`
- 当前任务类型：YOLO detect

## 快速开始

```powershell
conda activate odplat
python -c "import sys; print(sys.executable)"
pip install -e ./apps/platform
odp-init
```

Python 路径应包含 `envs\odplat`。不要提交虚拟环境、缓存、日志、数据集、模型权重和运行产物。

## Steel 数据集流程

```powershell
# 导入 Roboflow VOC zip
odp-import-dataset "..\steel surface defect.v1i.voc.zip" --name steel-surface-defect --format voc --overwrite

# 转换、划分、生成 dataset yaml
odp-transform --dataset steel-surface-defect --format pascal_voc --task detect

# 数据质量检查
odp-validate --dataset steel-surface-defect --executor your-name

# 训练
odp-train --yaml train --data steel-surface-defect --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --device 0 --workers 4 --name steel-defect-yolo11n

# 模型评估，区别于 odp-validate 数据质检
odp-val --config val --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --data steel-surface-defect --device 0

# 推理；默认会读取 apps/platform/configs/runtime/infer_pipeline.yaml 的 steel 中文标签映射
odp-infer --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --source data/processed/steel-surface-defect/test/images --conf 0.25 --device 0 --name steel-defect-test

# 摄像头实时推理
odp-infer --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --source 0 --show --conf 0.25 --device 0 --name steel-camera-demo
```

## CLI 一览

| 命令 | 用途 |
| --- | --- |
| `odp-init` | 初始化运行目录 |
| `odp-reset` | 安全清理运行产物，默认 dry-run |
| `odp-import-dataset` | 导入数据集 zip，目前支持 VOC zip |
| `odp-transform` | 数据格式转换、划分和 YOLO yaml 生成 |
| `odp-validate` | 数据质量检查，生成 JSON/Markdown/HTML/CSV 报告 |
| `odp-gen-config` | 生成 train / val / infer 运行配置 |
| `odp-train` | YOLO 训练、训练日志、manifest、权重归档 |
| `odp-val` | YOLO 模型评估，不归档权重 |
| `odp-infer` | 图片、目录、视频、摄像头推理 |
| `odp-plot-training` | 绘制训练曲线和指标摘要 |

## 桌面端推理演示

桌面端位于 `apps/desktop`，当前提供 PySide6 最小推理演示窗口，复用 platform 的 `infer_yolo()`、`InferHooks` 和 `OutputSink`，不复制 YOLO 推理逻辑。

```powershell
conda activate odplat
pip install -r apps/desktop/requirements.txt
python apps/desktop/main.py
```

界面默认优先查找 `models/trained/**/*best*.pt`，并读取 `apps/platform/configs/runtime/infer_pipeline.yaml` 中的 steel 中文类别映射。`odplat` 当前若未安装 PySide6，运行时会给出明确安装提示。

## 模块边界

- `common/`：路径、日志、计时、注册表、资源引用解析、系统信息。
- `data_pipeline/`：VOC / COCO / YOLO 转换、划分、物化、dataset yaml 生成。
- `data_validation/`：数据质检注册表、快照缓存、检查项、报告和返工清单。
- `runtime_config/`：训练、评估、推理配置模型，支持默认值、YAML、CLI 合并。
- `training/`：训练编排、日志、manifest、权重归档、训练图表。
- `evaluation/`：D7 模型评估，提供 `ValService`、`ValResult`、`odp-val`。
- `frame_source/`：图片、图片文件夹、视频、摄像头统一帧输入源。
- `visualization/`：中文标签、颜色映射、圆角框、Pillow 文本渲染和尺寸缓存。
- `inference/`：D8 推理服务、逐帧流水线、hook、sink、HUD、审计输出。
- `apps/desktop/`：PySide6 桌面端推理演示，只调用 platform 服务层。

## 根目录参考资产说明

根目录的 `inference.zip`、`visualization.zip`、`data_pipeline*.zip`、单个 `.py` 脚本和 HTML/Markdown 文档是带教发布的参考资产，用于对照实现。`odp-import-dataset` 只用于导入真实数据集压缩包，例如 `steel surface defect.v1i.voc.zip`，不要把参考代码 zip 当成数据集导入。

## 后续重点

- D8 推理继续增强 pipeline / hook / sink 的扩展点，服务桌面端复用。
- 桌面端后续增加评估结果、数据质检报告和训练曲线浏览。
- Web 后端仍是预留目录，尚未进入当前阶段主线。

## 验证

```powershell
conda activate odplat
python -m ruff check apps/platform/src/od_platform
python -m pytest apps/platform/src/od_platform/tests -q
python -m compileall apps/desktop
```
