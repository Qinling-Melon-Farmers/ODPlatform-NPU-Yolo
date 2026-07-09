from pathlib import Path

WORKSPACE_MARKER: str = ".odp-workspace"


def _find_workspace_root(
    start: Path,
    markers: tuple[str, ...] = (WORKSPACE_MARKER,),
) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        for marker in markers:
            if (parent / marker).exists():
                return parent

    raise FileNotFoundError(
        f"找不到 workspace marker 文件 ({markers})，"
        f"请确认仓库根目录已存在 {WORKSPACE_MARKER} 文件"
    )


ROOT_DIR: Path = _find_workspace_root(Path(__file__))

APP_DIR: Path = ROOT_DIR / "apps" / "platform"

DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
RUNS_DIR: Path = ROOT_DIR / "runs"
VALIDATION_RUNS_DIR: Path = RUNS_DIR / "data_validation"
INFERENCE_RUNS_DIR: Path = RUNS_DIR / "inference"
RESET_BACKUP_DIR: Path = RUNS_DIR / "reset_backup"

PRETRAINED_MODELS_DIR: Path = MODELS_DIR / "pretrained"
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"

RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
TRAIN_DIR: Path = DATA_DIR / "train"
VAL_DIR: Path = DATA_DIR / "val"
TEST_DIR: Path = DATA_DIR / "test"

CONFIGS_DIR: Path = APP_DIR / "configs"
DATASET_CONFIGS_DIR: Path = CONFIGS_DIR / "datasets"
RUNTIME_CONFIGS_DIR: Path = CONFIGS_DIR / "runtime"
LOGGING_DIR: Path = APP_DIR / "logging"
META_LOGGING_DIR: Path = APP_DIR / "meta_logging"
UNIT_TEST_DIR: Path = APP_DIR / "tests"

DOCS_DIR: Path = ROOT_DIR / "docs"
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"

PROTECTED_DIRS: tuple[Path, ...] = (
    ROOT_DIR,
    ROOT_DIR / ".git",
    ROOT_DIR / "apps",
    APP_DIR / "src",
    APP_DIR / "pyproject.toml",
    APP_DIR / "README.md",
    SCRIPTS_DIR,
    DOCS_DIR,
    RAW_DATA_DIR,
    PRETRAINED_MODELS_DIR,
    ROOT_DIR / WORKSPACE_MARKER,
    CONFIGS_DIR,
    META_LOGGING_DIR,
)


def get_dirs_to_initialize() -> list[Path]:
    return [
        DATA_DIR,
        MODELS_DIR,
        RUNS_DIR,
        INFERENCE_RUNS_DIR,
        RESET_BACKUP_DIR,
        PRETRAINED_MODELS_DIR,
        TRAINED_MODELS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CONFIGS_DIR,
        DATASET_CONFIGS_DIR,
        RUNTIME_CONFIGS_DIR,
        LOGGING_DIR,
        UNIT_TEST_DIR,
        DOCS_DIR,
        SCRIPTS_DIR,
    ]


def get_dirs_to_reset() -> list[Path]:
    """返回 reset_project 允许清理的运行时目录白名单。

    Returns:
        仅包含运行时产物目录的路径列表。业务数据、代码、文档、
        原始数据与预训练权重不在该清单内。
    """
    return [
        RUNS_DIR,
        VALIDATION_RUNS_DIR,
        INFERENCE_RUNS_DIR,
        CHECKPOINTS_DIR,
        LOGGING_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
    ]


def is_protected(path: Path) -> bool:
    """判断给定路径是否处于 reset_project 保护范围内。

    Args:
        path: 待判断的路径。

    Returns:
        命中保护目录、保护目录子路径，或越出工作区时返回 True。

    Raises:
        TypeError: 输入不是 ``Path`` 实例时抛出。
    """
    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path 实例")

    resolved_path = path.resolve()
    resolved_root = ROOT_DIR.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return True

    for protected in PROTECTED_DIRS:
        resolved_protected = protected.resolve()
        if resolved_path == resolved_protected:
            return True
        if resolved_protected in {resolved_root, (ROOT_DIR / "apps").resolve()}:
            continue
        try:
            resolved_path.relative_to(resolved_protected)
        except ValueError:
            continue
        return True

    return False


def dataset_processed_dir(name: str) -> Path:
    """Return the processed dataset root for one dataset name."""
    return PROCESSED_DATA_DIR / name


def dataset_yaml_path(name: str) -> Path:
    """Return the generated Ultralytics yaml path for one dataset name."""
    return DATASET_CONFIGS_DIR / f"{name}.yaml"


def validation_run_dir(run_id: str) -> Path:
    """返回某次验证运行的产出目录: runs/data_validation/<run_id>/

    Args:
        run_id: 形如 "20260516_184523" 的时间戳 ID (由 validate_dataset 生成)

    Returns:
        Path 对象 (尚未创建, 调用方自己 mkdir)
    """
    return VALIDATION_RUNS_DIR / run_id


def runtime_config_path(name: str) -> Path:
    """Return one runtime config path: apps/platform/configs/runtime/<name>.yaml."""
    return RUNTIME_CONFIGS_DIR / f"{name}.yaml"


if __name__ == "__main__":
    print(f"ROOT DIR (workspace) = {ROOT_DIR}")
    print(f"APP DIR = {APP_DIR}")
    print(f"DATA DIR = {DATA_DIR}")
    print(f"MODELS DIR = {MODELS_DIR}")
    print(f"RUNS DIR = {RUNS_DIR}")
    print(f"PRETRAINED MODELS DIR = {PRETRAINED_MODELS_DIR}")
    print(f"TRAINED MODELS DIR = {TRAINED_MODELS_DIR}")
    print(f"CHECKPOINTS DIR = {CHECKPOINTS_DIR}")
    print(f"RAW DATA DIR = {RAW_DATA_DIR}")
    print(f"PROCESSED DATA DIR = {PROCESSED_DATA_DIR}")
    print(f"TRAIN DIR = {TRAIN_DIR}")
    print(f"VAL DIR = {VAL_DIR}")
    print(f"TEST DIR = {TEST_DIR}")
    print(f"CONFIGS DIR = {CONFIGS_DIR}")
    print(f"LOGGING DIR = {LOGGING_DIR}")
    print(f"META LOGGING DIR = {META_LOGGING_DIR}")
    print(f"UNIT TEST DIR = {UNIT_TEST_DIR}")
    for directory in get_dirs_to_initialize():
        print(f"将要初始化的目录有: {directory.relative_to(ROOT_DIR)}")
