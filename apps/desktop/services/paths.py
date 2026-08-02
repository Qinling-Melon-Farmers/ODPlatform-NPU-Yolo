"""桌面端路径与默认值（services 层，可单测）。

注意：本文件位于 apps/desktop/services/，仓库根为 parents[3]。
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[3]
PLATFORM_SRC: Path = ROOT_DIR / "apps" / "platform" / "src"


def default_model() -> Path:
    """默认模型：最近训练 best.pt，回退 checkpoints/best.pt。"""
    candidates = sorted((ROOT_DIR / "models" / "trained").glob("**/*best*.pt"))
    if candidates:
        return candidates[-1]
    return ROOT_DIR / "models" / "checkpoints" / "best.pt"


def default_source() -> str:
    """默认推理源：steel 测试集首图，回退摄像头 0。"""
    test_images = ROOT_DIR / "data" / "processed" / "steel-surface-defect" / "test" / "images"
    if test_images.exists():
        first = next(iter(sorted(test_images.glob("*"))), None)
        if first is not None:
            return str(first)
    return "0"


def default_results_csv() -> Path:
    """默认训练曲线 csv：最近 results.csv，回退标准路径。"""
    candidates = sorted((ROOT_DIR / "runs").glob("**/results.csv"))
    if candidates:
        return candidates[-1]
    return ROOT_DIR / "runs" / "detect" / "train" / "results.csv"


def is_relative_to(path: Path, base: Path) -> bool:
    """判断 path 是否位于 base 之下。"""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
