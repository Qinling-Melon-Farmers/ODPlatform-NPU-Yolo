# 训练检查点目录

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **所属** | ODPlatform 顶层共享资产（产物） |

---

## 用途

存放训练过程中的**中间检查点**（checkpoint），用于断点续训和训练过程回溯。

---

## 使用规范

1. 训练框架在每轮（epoch）或定期自动保存检查点到此目录
2. 训练正常结束后，保留最后一个检查点用于断点续训
3. 不再需要的旧检查点可安全删除

---

## 不进 Git

检查点文件被 `.gitignore` 排除，不会进入版本库。

---

## 参考

- [pretrained/README.md](../pretrained/README.md) — 预训练权重（输入）
- [trained/README.md](../trained/README.md) — 最终产出权重（成品）
