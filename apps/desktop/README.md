# Desktop 端

`apps/desktop` 是 ODPlatform 的桌面端应用目录。桌面端采用 PySide6，定位是调用 platform 后端能力的可视化工作台，不在本目录重复实现训练、推理、数据质检或数据转换逻辑。

## 当前状态

已提供最小推理演示入口：

```powershell
conda activate odplat
pip install -r apps/desktop/requirements.txt
python apps/desktop/main.py
```

当前界面支持：

- 选择模型权重，默认优先查找 `models/trained/**/*best*.pt`。
- 选择输入源：图片、图片目录、视频，或直接使用摄像头 `0`。
- 读取 `apps/platform/configs/runtime/infer_pipeline.yaml` 中的 steel 中文类别映射和美化框配置。
- 启动、停止推理，并显示实时画面、FPS、累计检测数量和输出目录。
- 通过 `QtSignalSink` 和 `InferHooks` 复用 platform 推理服务，不在 UI 层直接实现 YOLO 推理。

## 技术边界

- UI 层使用 PySide6 / Qt signal / worker thread。
- 推理只调用 `od_platform.inference.infer_yolo()` 以及 hook/sink 接口。
- 帧输入只调用 `od_platform.frame_source`。
- 绘制只复用 `od_platform.visualization`。
- platform 后端不得 import PySide6、PyQt、FastAPI、Celery 等前端或调度框架。

## 后续开发

1. 将当前演示窗口扩展为工作台式布局：模型、输入源、运行状态、检测统计分区展示。
2. 增加评估结果浏览，读取 `runs/evaluation/**/odp_audit.json`。
3. 增加数据质检报告浏览，读取 `runs/data_validation/**/report.md` 与 `report.json`。
4. 增加训练结果浏览，展示 `results.csv` 曲线和归档权重信息。

## 验收标准

- 桌面端能使用同一份 steel best 权重完成图片或摄像头推理。
- 关闭窗口或点击停止后能释放摄像头资源。
- UI 代码不 import Ultralytics 训练逻辑，不复制 platform 的业务代码。
