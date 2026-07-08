"""Text size and font cache for realtime visualization."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from od_platform.common.paths import ROOT_DIR

logger = logging.getLogger(__name__)

DEFAULT_FONT_NAME = "LXGWWenKai-Bold"
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")


def _candidate_font_dirs() -> list[Path]:
    """Return project, workspace, and system font directories."""
    dirs: list[Path] = [
        ROOT_DIR / "apps" / "platform" / "assets" / "fonts",
        ROOT_DIR / "assets" / "fonts",
        ROOT_DIR.parent,
    ]
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(windir) / "Fonts")
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            dirs.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs.extend(
            [
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                Path.home() / "Library" / "Fonts",
            ]
        )
    else:
        dirs.extend(
            [
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path.home() / ".fonts",
                Path.home() / ".local" / "share" / "fonts",
            ]
        )
    return [path for path in dirs if path.is_dir()]


def _match_font_in_dir(directory: Path, name: str, *, recursive: bool) -> Path | None:
    has_ext = Path(name).suffix.lower() in FONT_EXTENSIONS
    if not recursive:
        candidates = [directory / name] if has_ext else [directory / f"{name}{ext}" for ext in FONT_EXTENSIONS]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    name_lower = name.lower()
    try:
        for candidate in directory.rglob("*"):
            if candidate.suffix.lower() not in FONT_EXTENSIONS:
                continue
            if candidate.name.lower() == name_lower or candidate.stem.lower() == name_lower:
                return candidate
    except (OSError, PermissionError):
        return None
    return None


def resolve_font_path(font: str | None) -> str:
    """Resolve a font path/name to a file path where possible.

    The default is intentionally a font *name*, not a vendored binary. If the
    user has installed the font system-wide, the resolver will find it.
    """
    name = font or DEFAULT_FONT_NAME
    path = Path(name)
    if path.is_file():
        return str(path.resolve())

    for directory in _candidate_font_dirs():
        hit = _match_font_in_dir(directory, name, recursive=directory not in {ROOT_DIR.parent})
        if hit is not None:
            return str(hit.resolve())
    return name


class TextSizeCache:
    """Precompute label text sizes and cache loaded fonts."""

    def __init__(
        self,
        labels: list[str],
        label_mapping: dict[str, str] | None = None,
        font_path: str | None = None,
        font_sizes: tuple[int, ...] | None = None,
        confidence_template: str = "99.0%",
    ) -> None:
        self.font_path = resolve_font_path(font_path)
        self.label_mapping = label_mapping or {}
        self.font_sizes = font_sizes or tuple(range(10, 48))
        self.confidence_template = confidence_template
        self._fallback_warned = False
        self._size_cache: dict[tuple[str, int], tuple[int, int]] = {}
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._precompute(labels)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype(self.font_path, size)
        except OSError as exc:
            if not self._fallback_warned:
                logger.warning(
                    "font %r cannot be loaded (%s); fallback PIL default font may not render Chinese correctly",
                    self.font_path,
                    exc,
                )
                self._fallback_warned = True
            return ImageFont.load_default()

    def _load_fonts(self) -> None:
        for size in self.font_sizes:
            self._font_cache[size] = self._load_font(size)

    def _precompute(self, labels: list[str]) -> None:
        self._load_fonts()
        measure_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(measure_img)

        display_labels = set(labels)
        for label in labels:
            mapped = self.label_mapping.get(label)
            if mapped:
                display_labels.add(mapped)

        for display_label in display_labels:
            full_text = f"{display_label} {self.confidence_template}"
            for size, font in self._font_cache.items():
                bbox = draw.textbbox((0, 0), full_text, font=font)
                self._size_cache[(display_label, size)] = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def get_size(self, display_label: str, font_size: int) -> tuple[int, int]:
        """Return cached text size, scaling from the closest size if needed."""
        key = (display_label, font_size)
        if key in self._size_cache:
            return self._size_cache[key]

        nearest_size = min(self.font_sizes, key=lambda size: abs(size - font_size))
        fallback = self._size_cache.get((display_label, nearest_size))
        if fallback is None:
            return (100, 30)
        scale = font_size / nearest_size
        return int(fallback[0] * scale), int(fallback[1] * scale)

    def get_font(self, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Return a cached font object."""
        if font_size not in self._font_cache:
            self._font_cache[font_size] = self._load_font(font_size)
        return self._font_cache[font_size]
