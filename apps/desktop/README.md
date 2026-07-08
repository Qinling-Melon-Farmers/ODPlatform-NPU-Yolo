# Desktop 端

`apps/desktop` 是 ODPlatform 后续桌面端应用目录。桌面端采用 PySide6，定位是调用 platform 后端能力的可视化工作台，不在本目录重复实现训练、推理、数据质检或数据转换逻辑。

## 首版目标

- 选择模型权重，优先支持 `models/trained/**/best.pt` 和手动路径。
- 选择输入源：图片、图片文件夹、视频、摄像头。
- 读取或编辑推理 pipeline 配置，包括 steel 中文类别映射和颜色映射。
- 启动、暂停、继续、停止推理。
- 显示实时画面、检测框、FPS、推理耗时、帧号、输入源分辨率。
- 将推理结果交给 `od_platform.inference` 的 sink/hook，不直接写散乱输出。

## 技术边界

- UI 层使用 PySide6 / Qt signal / worker thread。
- 推理只调用 `od_platform.inference.InferService` 或其 hook/sink 接口。
- 帧输入只调用 `od_platform.frame_source`。
- 结果绘制只调用 `od_platform.visualization`。
- 训练、评估、数据质检结果查看作为后续功能，不进入首版最小闭环。

## 开发顺序

1. 先完成 platform 后端能力：`odp-val`、steel 推理配置、推理 hook/sink 稳定。
2. 再搭建 PySide6 主窗口、模型选择、输入源选择、启动/停止按钮。
3. 最后加入训练结果、评估结果、数据质检报告的浏览入口。

## 验收标准

- 桌面端能使用同一个 steel best 权重完成图片或摄像头推理。
- 关闭窗口或点击停止后能释放摄像头资源。
- UI 代码不 import Ultralytics 训练逻辑，不复制 platform 的业务代码。
