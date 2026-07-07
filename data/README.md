# 数据目录

`data/` 存放数据集输入和由流水线生成的数据集产物。真实图像、标注和大文件不进入 Git。

## 目录结构

```text
data/
├── raw/                         # 原始数据集，只读输入
│   └── <dataset>/
│       ├── images/
│       └── annotations/
├── processed/                   # 转换、划分后的数据集产物
│   └── <dataset>/
│       ├── train/images/
│       ├── train/labels/
│       ├── val/images/
│       ├── val/labels/
│       ├── test/images/
│       └── test/labels/
└── README.md
```

## 使用方式

```powershell
odp-import-dataset "C:\path\dataset.voc.zip" --name steel-surface-defect
odp-transform --dataset steel-surface-defect --format pascal_voc --task detect
odp-validate --dataset steel-surface-defect --executor your-name
```

`odp-transform` 会把可训练数据集写入 `data/processed/<dataset>/`，并生成对应的 dataset YAML 到 `apps/platform/configs/datasets/`。

## 规范

- `data/raw/` 只读，不在原地清洗或修改。
- `data/processed/` 是可再生产物，可由命令重新生成。
- 图片、标注、大文件不进 Git，仅 README 进入版本库。
- 数据划分应固定随机种子，保证训练、验证、测试集可复现。
