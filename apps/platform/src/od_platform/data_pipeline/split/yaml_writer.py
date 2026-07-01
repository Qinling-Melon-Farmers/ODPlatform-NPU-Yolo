"""Generate Ultralytics-compatible dataset yaml files."""

from __future__ import annotations

import logging
from pathlib import Path

from od_platform.common.constants import Task
from od_platform.data_pipeline.split.manifest import SplitManifest
from od_platform.data_pipeline.split.materializer import SplitOutputDirs

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def _scalar(value: object) -> str:
    text = str(value)
    if not text or any(char in text for char in ["#", "[", "]", "{", "}"]) or ": " in text:
        return repr(text)
    return text


def write_dataset_yaml(
    yaml_path: Path | None = None,
    *,
    dataset_root: Path,
    classes: list[str],
    dirs: SplitOutputDirs | None = None,
    output_path: Path | None = None,
    manifest: SplitManifest | None = None,
    dataset_name: str | None = None,
    source_format: str | None = None,
    task: str = Task.DETECT,
) -> Path:
    """Write a dataset yaml file and return its path."""
    output = yaml_path or output_path or dataset_root / "dataset.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)

    resolved_dirs = dirs or SplitOutputDirs.for_dataset_root(dataset_root)
    rel_paths = resolved_dirs.yaml_rel_paths()

    lines: list[str] = [
        "# ODPlatform auto-generated dataset config",
        f"path: {_scalar(dataset_root)}",
        "",
        f"train: {_scalar(rel_paths['train'])}",
        f"val: {_scalar(rel_paths['val'])}",
        f"test: {_scalar(rel_paths['test'])}",
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    for idx, name in enumerate(classes):
        lines.append(f"  {idx}: {_scalar(name)}")

    if manifest is not None or dataset_name is not None or source_format is not None:
        lines.extend(
            [
                "",
                "odp_meta:",
                f"  schema_version: {_SCHEMA_VERSION}",
                f"  dataset_name: {_scalar(dataset_name or dataset_root.name)}",
                f"  source_format: {_scalar(source_format or 'unknown')}",
                f"  task: {_scalar(task)}",
            ]
        )
        if manifest is not None:
            counts = manifest.summary()
            lines.extend(
                [
                    "  split:",
                    f"    strategy: {_scalar(manifest.strategy)}",
                    f"    random_state: {manifest.random_state}",
                    "    rates:",
                    f"      train: {manifest.train_rate:.6g}",
                    f"      val: {manifest.val_rate:.6g}",
                    f"      test: {manifest.test_rate:.6g}",
                    "    counts:",
                    f"      train: {counts['train']}",
                    f"      val: {counts['val']}",
                    f"      test: {counts['test']}",
                    f"      total: {counts['total']}",
                ]
            )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("dataset yaml written: %s", output)
    return output
