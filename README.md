# ODPlatform

ODPlatform 是生产实习阶段构建的目标检测开发平台，当前重点覆盖数据集导入、格式转换、数据质检、训练、推理、训练结果可视化、推理结果美化绘制，以及图片/视频/摄像头统一输入源。

## 当前状态

- 仓库根目录：`ODPlatform`
- Python 包：`apps/platform/src/od_platform`
- 安装方式：`pip install -e ./apps/platform`
- 开发环境：使用 `odplat` Conda 环境
- 日志：终端使用彩色输出，文件日志保持纯文本

## 快速开始

```powershell
conda activate odplat
python -c "import sys; print(sys.executable)"
pip install -e ./apps/platform
odp-init
```

Python 路径应包含 `envs\odplat`。不要在项目内提交虚拟环境、缓存、日志、数据集、模型权重和运行产物。

## 常用命令

```powershell
# 初始化运行目录
odp-init

# 安全重置运行产物，默认 dry-run
odp-reset
odp-reset --yes --force

# 导入 VOC zip 数据集到 data/raw
odp-import-dataset "C:\path\dataset.voc.zip" --name steel-surface-defect

# 转换并划分数据集
odp-transform --dataset steel-surface-defect --format pascal_voc --task detect

# 数据质量检查
odp-validate --dataset steel-surface-defect --executor your-name

# 生成运行配置
odp-gen-config train --force
odp-gen-config val --force
odp-gen-config infer --force

# 训练、推理、训练曲线
odp-train --yaml train --data steel-surface-defect --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --device 0
odp-infer --config infer
odp-plot-training runs/detect/train/results.csv --output runs/detect/train/training_results.png
```

## 主要模块

- `common/`：路径、日志、性能计时、注册表、资源引用解析。
- `data_pipeline/`：数据格式转换、YOLO 数据划分、数据集 YAML 生成。
- `data_validation/`：数据质检注册表、快照、检查项、JSON/Markdown/HTML/CSV 报告。
- `runtime_config/`：训练、验证、推理配置模型，支持默认值、YAML、CLI 合并。
- `training/`：YOLO 训练、训练产物审计、权重归档、结果图表。
- `inference/`：YOLO 推理、审计清单和结果摘要。
- `frame_source/`：图片、图片文件夹、视频、摄像头四类输入源，支持同步、线程和异步包装。
- `visualization/`：YOLO 检测框美化绘制，支持中文标签、颜色映射、圆角框和文本尺寸缓存。

## 兼容说明

保留少量带教脚本兼容入口，例如 `od_platform.cli.model_train`、`od_platform.validate_dateset`、`CameraFrameSource` 等旧名称。新代码应使用正式入口：

- `od_platform.cli.train_model`
- `od_platform.data_validation`
- `CameraSource` / `ImageSource` / `ImageFolderSource` / `VideoSource`

## 验证

```powershell
conda activate odplat
python -m ruff check apps/platform/src/od_platform
python -m pytest apps/platform/src/od_platform/tests -q
```

## 文档

- [Platform README](apps/platform/README.md)
- [Desktop README](apps/desktop/README.md)
- [CLI README](apps/platform/src/od_platform/cli/README.md)
- [ADR-001 Monorepo](docs/architecture/ADR-001-monorepo.md)
- [ADR-002 Reset 安全设计](docs/architecture/ADR-002-reset-safety-design.md)
- [ADR-003 数据转换注册表](docs/architecture/ADR-003-data-converter-registry.md)
