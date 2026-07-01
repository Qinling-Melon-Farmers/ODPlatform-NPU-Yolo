# ADR-003 数据标注转换器采用注册表模式

## 状态

Accepted

## 决策日期

2026-06-30

## 背景

数据管线需要支持多种标注格式输入，例如 Pascal VOC、COCO JSON、YOLO txt。后续还会增加数据集划分、校验和统计能力。如果服务层直接写 `if format == ...` 分支，格式越多，代码越难维护。

## 决定

数据标注转换器采用注册表模式，数据集划分策略复用同一模式：

- `registry.register(format_name, supported_tasks=...)` 负责注册具体转换器。
- `registry.get_converter(format_name)` 按格式名查询转换器。
- `service.convert_data_to_yolo(...)` 只依赖注册表，不直接依赖具体格式实现。
- `converters/` 包下每个文件实现一个格式转换器，导入时通过装饰器自动注册。
- `split.registry.register(strategy_name, ...)` 负责注册划分策略。
- `common.registry_utils.import_modules_from_package(...)` 负责包扫描和自动导入，避免 convert 与 split 重复实现同一段扫描逻辑。

## 调用链

```text
调用方
  -> convert_data_to_yolo(...)
    -> get_converter(annotation_format)
      -> lazy import converters/*
      -> ConverterEntry.func(input_path, output_labels_dir, options)
```

## 扩展方式

新增一种格式时，只需要：

1. 在 `common.constants.AnnotationFormat` 增加格式名。
2. 在 `data_pipeline/convert/converters/` 下新增一个模块。
3. 给转换函数加 `@register(...)` 装饰器。
4. 补对应单元测试。

服务层不需要修改。

## 当前实现

- `pascal_voc.py`: Pascal VOC XML -> YOLO txt
- `coco.py`: COCO detection JSON -> YOLO txt
- `yolo.py`: YOLO txt 自身校验与规范化输出
- `split/strategies/random.py`: 按固定随机种子随机划分 train/val/test
- `split/manifest.py`: 记录三组样本和比例、随机种子、策略名
- `split/materializer.py`: 将 manifest 落盘为 YOLO `train/val/test` 目录

## 后果

正面：

- 格式扩展只新增模块，不改服务层。
- 支持能力可通过 `list_capabilities()` 查询。
- 数据集划分策略复用同一思路，后续新增分层划分时只需新增 `split/strategies/stratified.py`。

负面：

- 转换器模块依赖导入副作用完成注册，需要保持 `converters/` 包结构稳定。
