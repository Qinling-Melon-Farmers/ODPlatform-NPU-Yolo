# 运行产物目录

`runs/` 存放训练、推理、数据质检等命令产生的运行现场。除本 README 外，运行产物不进入 Git。

## 常见结构

```text
runs/
├── detect/                       # Ultralytics 训练输出
│   └── <train-name>/
├── inference/                    # 推理输出、manifest、summary
│   └── detect/
├── data_validation/              # 数据质检报告
│   └── <run_id>/
│       ├── report.json
│       ├── report.md
│       ├── report.html
│       └── rework.csv
└── README.md
```

## 规范

- `runs/` 是过程现场，可以清理后重新生成。
- 训练结束后需要长期保留的权重会归档到 `models/trained/`。
- 数据质检报告可用于提交、复查和返工，但默认不进入 Git。
- `odp-reset` 会安全清理运行产物；默认 dry-run，只有显式 `--yes` 才执行删除。
