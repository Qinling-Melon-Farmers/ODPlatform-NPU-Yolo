"""Shared Pydantic models for runtime configuration."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class BaseRuntimeConfig(BaseModel):
    """Common configuration accepted by Ultralytics runtime commands."""

    model_config = ConfigDict(extra="forbid")

    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = {"extra_args"}

    model: str = Field(default="yolo11n.pt", description="Model path or model name.")
    data: str = Field(default="rsod", description="Dataset yaml name or path.")
    batch: int | float = Field(default=16, description="Batch size.")
    imgsz: int = Field(default=640, ge=32, description="Image size.")
    workers: int = Field(default=8, ge=0, description="Dataloader workers.")
    cache: bool | str = Field(default=False, description="Dataset cache strategy.")
    device: int | str | list[int | str] | None = Field(default=None, description="Training device.")
    project: str | None = Field(default=None, description="Ultralytics project directory.")
    name: str | None = Field(default=None, description="Ultralytics run name.")
    exist_ok: bool = Field(default=False, description="Allow overwriting an existing run directory.")
    save: bool = Field(default=True, description="Save training artifacts.")
    verbose: bool = Field(default=True, description="Verbose Ultralytics output.")
    seed: int = Field(default=0, ge=0, description="Random seed.")
    deterministic: bool = Field(default=True, description="Deterministic training mode.")

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        """Convert framework config into kwargs accepted by Ultralytics."""
        payload = self.model_dump(exclude_none=True)
        for key in self.FRAMEWORK_ONLY_FIELDS:
            payload.pop(key, None)
        return payload

    def audit_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for logs and manifests."""
        return self.model_dump(mode="json", exclude_none=True)
