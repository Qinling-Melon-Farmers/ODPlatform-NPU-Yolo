# Platform 端

`apps/platform` 是 ODPlatform 的核心 Python 包，包名为 `od_platform`。当前承载命令行工具、数据流水线、数据质检、运行配置、训练、推理和统一帧输入源。

## 安装

```powershell
conda activate odplat
pip install -e ./apps/platform
```

安装后会提供以下命令：

| 命令 | 用途 |
| --- | --- |
| `odp-init` | 初始化运行目录 |
| `odp-reset` | 安全清理运行产物，默认 dry-run |
| `odp-import-dataset` | 导入原始数据集压缩包 |
| `odp-transform` | 数据格式转换、划分和 YOLO 数据集落盘 |
| `odp-validate` | 数据质量检查并生成报告 |
| `odp-gen-config` | 生成 train / val / infer 运行配置 |
| `odp-train` | 启动 YOLO 训练 |
| `odp-infer` | 启动 YOLO 推理 |
| `odp-plot-training` | 生成训练曲线和指标摘要 |

## 模块边界

- `common`：公共路径、日志、性能计时、注册表工具。
- `runtime_config`：Pydantic 运行配置，支持默认值、YAML、CLI 三源合并。
- `data_pipeline`：VOC / COCO / YOLO 转换、数据划分、数据集 YAML 生成。
- `data_validation`：数据质检检查项、快照缓存、报告输出和返工清单。
- `training`：训练执行、日志、审计、权重归档、结果可视化。
- `inference`：推理执行、结果摘要和审计。
- `frame_source`：图片、图片文件夹、视频、摄像头输入源。

## 日志

`common.logging_utils.get_logger()` 统一配置终端和文件日志：

- 终端日志使用 `colorlog` 彩色输出。
- 文件日志保持纯文本，不包含 ANSI 控制字符。
- 训练和推理会生成独立运行日志、manifest 和结果摘要。

## 兼容入口

项目保留部分兼容路径以便旧带教脚本继续运行，例如 `od_platform.cli.model_train` 和 `od_platform.validate_dateset`。新代码应使用正式模块名：`train_model`、`data_validation`、`frame_source` 的新类名。

## 验证

```powershell
conda activate odplat
python -m ruff check apps/platform/src/od_platform
python -m pytest apps/platform/src/od_platform/tests -q
```
