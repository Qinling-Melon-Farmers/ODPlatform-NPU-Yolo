# ODPlatform CLI

正式脚本入口定义在 `apps/platform/pyproject.toml` 的 `[project.scripts]` 中。

## 命令一览

| 命令 | 入口 | 用途 |
| --- | --- | --- |
| `odp-init` | `od_platform.cli.init_project:initialize_project` | 初始化运行目录 |
| `odp-reset` | `od_platform.cli.reset_project:main` | 安全清理运行产物 |
| `odp-import-dataset` | `od_platform.cli.import_dataset:main` | 导入数据集 zip |
| `odp-transform` | `od_platform.cli.transform_data:main` | 数据转换和划分 |
| `odp-validate` | `od_platform.cli.validate_data:main` | 数据质量检查 |
| `odp-gen-config` | `od_platform.runtime_config.generator:main` | 生成运行配置 |
| `odp-train` | `od_platform.cli.train_model:main` | 训练模型 |
| `odp-val` | `od_platform.cli.evaluate_model:main` | 评估模型 |
| `odp-infer` | `od_platform.cli.infer_model:main` | 推理 |
| `odp-plot-training` | `od_platform.cli.plot_training:main` | 绘制训练结果 |

## 常用示例

```powershell
odp-init
odp-reset
odp-reset --yes --force

odp-import-dataset "..\steel surface defect.v1i.voc.zip" --name steel-surface-defect --format voc --overwrite
odp-transform --dataset steel-surface-defect --format pascal_voc --task detect
odp-validate --dataset steel-surface-defect --executor your-name

odp-gen-config train --force
odp-train --yaml train --data steel-surface-defect --model yolo11n.pt --epochs 100 --batch 16 --imgsz 640 --device 0 --workers 4 --name steel-defect-yolo11n

odp-gen-config val --force
odp-val --config val --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --data steel-surface-defect --device 0

odp-infer --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --source data/processed/steel-surface-defect/test/images --conf 0.25 --device 0 --name steel-defect-test --threaded
odp-infer --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --source demo.mp4 --conf 0.25 --device 0 --name steel-video-demo --threaded
odp-infer --model models/trained/steel-defect-yolo11n-2-20260707-122829-yolo11n-best/best.pt --source 0 --show --conf 0.25 --device 0 --name steel-camera-demo --threaded
```

`odp-infer --threaded` 会启用后台读帧。摄像头源使用 latest 缓冲，优先保证实时显示；视频和图片目录使用 bounded 缓冲，优先保证完整处理不丢帧。

## 返回码

- `odp-validate` 用于数据质检，返回 CI 可用状态码。
- `odp-val` 用于模型评估，成功返回 `0`，评估准备或执行失败返回 `1`。
- `odp-train`、`odp-infer` 参数错误返回 `2`，键盘中断返回 `130`。

## 兼容说明

`od_platform.cli.model_train` 仅作为旧带教脚本兼容入口保留。新代码和文档统一使用 `od_platform.cli.train_model`。`odp-val` 不替代 `odp-validate`，前者评估模型，后者检查数据。
