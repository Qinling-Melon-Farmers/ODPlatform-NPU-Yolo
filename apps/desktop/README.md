# ODPlatform Desktop

`apps/desktop` 是 SteelDefect Studio 的 PySide6 桌面工作台。桌面端只调用 platform 服务层和现有 CLI，不在 UI 层复制训练、推理、质检或数据转换逻辑。

## 启动

```powershell
conda activate odplat
pip install -r requirements.txt
pip install -e ./apps/platform
python apps/desktop/main.py
```

也可以只安装桌面依赖：

```powershell
pip install -r apps/desktop/requirements.txt
```

## 当前能力

- 推理：支持图片、图片文件夹、视频、摄像头 `0` 或其他摄像头编号。
- 推理参数：支持模型、输入源、runtime 配置、pipeline 配置、detect task、conf、iou、imgsz、max_det、classes、device、name、max_frames、视频抽帧间隔。
- D8 多级流水线：默认使用 platform 的 `--threaded` 路径，摄像头采用 latest 缓冲，视频/目录采用 bounded 缓冲。
- 交互控制：启动、暂停、继续、停止推理，显示实时画面、FPS、检测数量、输出目录和日志。
- 结果浏览：模型评估、数据质检、训练结果支持筛选、搜索和摘要查看。
- 训练曲线：支持 `results.csv` 对应图表内嵌预览。
- 模型目录：内置 YOLOv5/v7/v8/v9/v10/11/12 全系列模型元数据浏览，支持系列筛选、自然语言推荐（最快/最准/平衡等），一键应用到训练或推理。
- 任务启动：按任务类型切换独立参数页，通过后台子进程调用 `odp-import-dataset`、`odp-transform`、`odp-validate`、`odp-val`、`odp-train`、`odp-reset`、`odp-plot-training`、`odp-list-models`、`odp-annotate`。
- 数据导入：支持在桌面端选择 `voc` 或 `yolo` zip；liftrace 这类包含 `data.yaml/images/labels` 的数据集应选择 `yolo`。
- 项目重置：桌面端支持 `dry-run`、`--backup`、`--yes`、`--force` 开关；默认 dry-run，避免误删运行产物。
- 训练曲线生成：桌面端可选择 `results.csv`，调用 `odp-plot-training` 生成训练曲线 PNG 和 summary JSON。
- 数据标注：调用 `odp-annotate` 打开 OpenCV 交互式标注窗口，支持类别列表与断点恢复（`--resume`）。
- 自动标注（VLM）：调用 `odp-auto-annotate` 接入视觉大模型（如 Qwen-VL / GLM-4.5V）按自然语言指令批量标注，支持断点续跑；默认 dry-run 避免误触发 API 调用。
- AI 任务：调用 `odp-agent` 用自然语言驱动平台执行目标检测任务（如"用最快的模型训练 rsod"），API 地址与模型必填（如 DeepSeek）。
- AI 助手对话页：内置对话式界面，输入自然语言任务（如"列出可用数据集"），后台线程流式渲染 Agent 事件（工具调用/结果/回答），支持安全模式（训练/推理 dry-run）与停止按钮。

训练任务默认 dry-run，避免在桌面端误触发长时间训练。

## 技术边界

- UI 使用 PySide6、Qt signal、worker thread。
- 推理调用 `od_platform.inference.infer_yolo()`、`InferHooks`、`OutputSink`。
- 帧输入调用 `od_platform.frame_source`。
- 绘制调用 `od_platform.visualization`。
- 任务启动调用 CLI module 子进程。
- platform 包不得 import PySide6、PyQt、FastAPI、Celery 等 UI 或调度框架。

## 后续可做

- 任务模板保存和运行历史复用。
- 多 run 训练曲线对比。
- exe 打包，暂不属于当前阶段。

## 验证

```powershell
conda activate odplat
python -m compileall apps/desktop
python -m pytest apps/platform/src/od_platform/tests/test_desktop_task_worker.py -q
python apps/desktop/main.py
```
