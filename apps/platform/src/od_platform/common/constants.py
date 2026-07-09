"""Project-wide shared constants."""

from __future__ import annotations

DEFAULT_RANDOM_STATE: int = 1210

# Floating-point tolerance for rates such as 1.0 - 0.7 - 0.3.
RATE_EPSILON: float = 1e-6


class AnnotationFormat:
    """Supported annotation format names."""

    PASCAL_VOC = "pascal_voc"
    COCO = "coco"
    YOLO = "yolo"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return cls.PASCAL_VOC, cls.COCO, cls.YOLO


class Task:
    """Supported vision task names."""

    DETECT = "detect"
    SEGMENT = "segment"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return cls.DETECT, cls.SEGMENT

    @classmethod
    def end_to_end(cls) -> tuple[str, ...]:
        """Task types supported by the current full ODPlatform workflow."""
        return (cls.DETECT,)

    @classmethod
    def ensure_end_to_end(cls, value: str) -> str:
        """Validate a task against the current end-to-end product boundary."""
        if value not in cls.end_to_end():
            raise ValueError(
                f"task {value!r} is not supported end-to-end yet; "
                f"supported tasks: {cls.end_to_end()}. Segment is reserved for future work."
            )
        return value


class SplitStrategy:
    """train/val/test split strategy names."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    STRATIFIED_MULTILABEL = "stratified_multilabel"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return cls.RANDOM, cls.STRATIFIED, cls.STRATIFIED_MULTILABEL


DEFAULT_SPLIT_STRATEGY: str = SplitStrategy.RANDOM

# Class balance report thresholds.
CLASS_MIN_IMAGES_HARD: int = 2
CLASS_MIN_BOXES_WARN: int = 20
CLASS_MIN_BOX_SHARE: float = 0.01

# Raw image / annotation coverage thresholds used before conversion.
COVERAGE_HARD_THRESHOLD: float = 0.5
COVERAGE_SOFT_THRESHOLD: float = 0.9

# Dataset validation thresholds for missing image-to-label pairs.
PAIR_MISSING_WARN_RATIO: float = 0.05
PAIR_MISSING_ERROR_RATIO: float = 0.5

# Annotation coverage thresholds — percentage of unannotated images per split.
NO_ANNOTATION_WARN_RATIO: float = 0.3
NO_ANNOTATION_ERROR_RATIO: float = 0.7

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
