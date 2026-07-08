# Platform 端

`apps/platform` 是 ODPlatform 的核心 Python 包，包名为 `od_platform`。它承载命令行工具、数据流水线、数据质检、运行配置、训练、模型评估、推理、美化可视化和统一帧输入源。

桌面端位于 `apps/desktop`，只调用本包提供的服务层和 hook/sink 接口，不在 UI 层重复实现 platform 业务逻辑。

## 安装

```powershell
conda activate odplat
pip install -e ./apps/platform
```

## 命令

| 命令 | 用途 |
| --- | --- |
| `odp-init` | 初始化运行目录 |
| `odp-reset` | 安全清理运行产物，默认 dry-run |
| `odp-import-dataset` | 导入原始数据集 zip |
| `odp-transform` | 数据转换、划分和 YOLO 数据集落盘 |
| `odp-validate` | 数据质量检查和报告生成 |
| `odp-gen-config` | 生成运行配置 |
| `odp-train` | YOLO 训练 |
| `odp-val` | YOLO 模型评估 |
| `odp-infer` | YOLO 推理，支持 D8 逐帧流水线 |
| `odp-plot-training` | 训练结果图表 |

## Steel 常用命令

```powershell
odp-transform --dataset steel-surface-defect --format pascal_voc --task detect
odp-validate --dataset steel-surface-defect --executor your-name
odp-train --yaml train --data steel-surface-defect --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --device 0 --workers 4 --name steel-defect-yolo11n
odp-val --config val --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --data steel-surface-defect --device 0
odp-infer --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --source data/processed/steel-surface-defect/test/images --conf 0.25 --device 0 --name steel-defect-test
```

## 模块边界

- `common`：公共路径、日志、计时、注册表、引用解析。
- `runtime_config`：Pydantic 运行配置，合并默认值、YAML 和 CLI。
- `data_pipeline`：VOC / COCO / YOLO 转换、划分、YOLO yaml 生成。
- `data_validation`：数据质检检查项、快照缓存、报告输出。
- `training`：训练执行、日志、审计、权重归档、结果图表。
- `evaluation`：模型评估，提供 `odp-val`，只产出指标和审计，不归档权重。
- `frame_source`：图片、图片文件夹、视频、摄像头输入源。
- `visualization`：检测框美化、中文标签映射、颜色映射。
- `inference`：推理执行、逐帧 pipeline、hook、sink、HUD 和审计。

## 验证

```powershell
conda activate odplat
python -m ruff check apps/platform/src/od_platform
python -m pytest apps/platform/src/od_platform/tests -q
```
