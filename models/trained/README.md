# 训练产出模型目录

`models/trained/` 存放从训练运行中归档出来、准备长期保留的模型权重。

## 当前归档方式

`odp-train` 训练结束后会从实际 Ultralytics 输出目录读取 `weights/best.pt` 和 `weights/last.pt`，并归档为带运行编号、时间戳和模型名的文件。

示例：

```text
models/trained/
└── train-4-20260707-150257-yolo11n-best.pt
└── train-4-20260707-150257-yolo11n-last.pt
```

## 规范

- 权重文件不进入 Git。
- `best.pt` 通常用于推理和评估。
- `last.pt` 通常用于断点续训或复现实验末态。
- 归档元数据和训练 manifest 以 `runs/` 中的运行目录为准。

## 相关目录

- `runs/`：每次训练的完整现场。
- `models/pretrained/`：外部预训练权重。
