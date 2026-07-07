# 预训练模型目录

`models/pretrained/` 存放外部下载或课程提供的预训练权重，例如 `yolo11n.pt`。

## 规范

- 这里的权重是输入资产，业务代码只读使用。
- `.pt` 等权重文件不进入 Git。
- 如果权重来自带教发布文件，可以手动复制到本目录，或在配置中直接引用其绝对路径。
- 训练时推荐通过 `--model yolo11n.pt` 或运行配置 `model` 字段引用。

## 相关目录

- `models/checkpoints/`：训练中间检查点或临时验证权重。
- `models/trained/`：训练结束后归档的 best / last 权重。
