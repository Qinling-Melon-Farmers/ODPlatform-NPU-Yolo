# 数据目录

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **所属** | ODPlatform 顶层共享资产 |

---

## 目录结构

```
data/
├── raw/                       ← 原始数据（只读，绝不就地修改）
│   └── yolo_staged_labels/    ← YOLO 格式标注暂存区
├── train/                     ← 训练集（由 raw 经固定种子一次性划分产出）
│   ├── images/
│   └── annotations/
├── val/                       ← 验证集
│   ├── images/
│   └── annotations/
└── test/                      ← 测试集
    ├── images/
    └── annotations/
```

---

## 使用规范

1. **原始数据只读**：`raw/` 下的数据**绝不**就地修改。所有清洗、转换、划分操作产出新文件，写入 `train/` / `val/` / `test/` 或 `processed/`。
2. **固定种子划分**：训练/验证/测试集的划分**一次性**完成并冻结，训练时只读那份固定划分，不每次临时分。
3. **不进 Git**：`data/` 下的图像、标注、大文件均被 `.gitignore` 排除，仅 `README.md` 进入版本库。

---

## 如何放入数据

在 `data/raw/` 下以**数据集名称**创建子文件夹，结构如下：

```
data/raw/<dataset_name>/
├── images/          ← 图片文件
└── annotations/     ← 标注文件
```

放入后运行 `odp-init`，系统会列出已发现的数据集目录。

---

## 参考

- [D0 设计指南 §6.5 "原始数据放哪"](../../D0-设计指南.md)
- [.gitignore](../../.gitignore)
