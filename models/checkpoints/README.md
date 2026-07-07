# 检查点目录

`models/checkpoints/` 用于放置需要被推理、验证或演示命令直接引用的权重文件。

## 使用场景

- 放置带教发布的演示权重。
- 放置从 `models/trained/` 挑选出的当前默认推理权重。
- 临时保存待验证的 `best.pt` 或 `last.pt`。

## 规范

- `.pt` 权重文件不进入 Git。
- 如果权重是训练自动归档的长期产物，优先放在 `models/trained/`。
- 如果权重是外部预训练输入，优先放在 `models/pretrained/`。
- 推理配置可通过 `model` 字段引用本目录权重。
