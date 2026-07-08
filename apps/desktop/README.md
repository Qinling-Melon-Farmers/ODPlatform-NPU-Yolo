# Desktop 端

`apps/desktop` 是 ODPlatform 的桌面端应用目录。桌面端采用 PySide6，定位是调用 platform 后端能力的可视化工作台，不在本目录重复实现训练、推理、数据质检或数据转换逻辑。

## 当前状态

已提供工作台第一版入口：

```powershell
conda activate odplat
pip install -r apps/desktop/requirements.txt
python apps/desktop/main.py
```

当前界面支持：

- 选择模型权重，默认优先查找 `models/trained/**/*best*.pt`。
- 选择输入源：图片、图片目录、视频，或直接使用摄像头 `0`。
- 读取 `apps/platform/configs/runtime/infer_pipeline.yaml` 中的 steel 中文类别映射和美化框配置。
- 可选推理运行配置，并可设置 task、conf、iou、imgsz、max_det、classes、device、name、max_frames。
- 启动、暂停/继续、停止推理，并显示实时画面、FPS、累计检测数量和输出目录。
- 默认开启“多级流水线”，复用 D8 `--threaded` 路径；摄像头源使用 latest 缓冲保证实时性，图片目录/视频源使用 bounded 缓冲保证不丢帧。
- 通过 `QtSignalSink` 和 `InferHooks` 复用 platform 推理服务，不在 UI 层直接实现 YOLO 推理。

工作台页面：

- 推理：图片、目录、视频、摄像头推理。
- 模型评估：读取 `runs/evaluation/**/odp_audit.json`。
- 数据质检：读取 `runs/data_validation/**/report.md` 或 `report.json`。
- 训练结果：读取 `runs/**/results.csv` 和 `weights/*.pt` 摘要。

## 输入源

- 图片：点击“图片/视频”选择 `.jpg`、`.png`、`.bmp`、`.webp` 等文件。
- 视频：点击“图片/视频”选择 `.mp4`、`.avi`、`.mkv`、`.mov` 等文件。
- 图片目录：点击“文件夹”选择包含图片的目录。
- 摄像头：点击“摄像头 0”，或手动输入其他摄像头编号。

## 技术边界

- UI 层使用 PySide6 / Qt signal / worker thread。
- 推理只调用 `od_platform.inference.infer_yolo()` 以及 hook/sink 接口。
- 帧输入只调用 `od_platform.frame_source`。
- 绘制只复用 `od_platform.visualization`。
- platform 后端不得 import PySide6、PyQt、FastAPI、Celery 等前端或调度框架。

## 后续开发

1. 增加训练曲线图片/表格可视化。
2. 增加运行历史筛选、搜索和删除本地产物的安全入口。
3. 增加数据导入、数据转换、数据质检、评估的任务启动表单。

## 验收标准

- 桌面端能使用同一份 steel best 权重完成图片、图片目录、视频或摄像头推理。
- 关闭窗口或点击停止后能释放摄像头资源。
- UI 代码不 import Ultralytics 训练逻辑，不复制 platform 的业务代码。
