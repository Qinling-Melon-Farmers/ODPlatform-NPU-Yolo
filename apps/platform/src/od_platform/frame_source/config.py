"""Configuration models for OpenCV frame sources."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CameraBackend = Literal["auto", "msmf", "dshow", "v4l2"]
CameraCodec = Literal["MJPG", "YUYV", "H264", "MP4V"]


class CameraConfig(BaseModel):
    """Camera capture settings requested from OpenCV."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    camera_id: int = Field(default=0, ge=0, description="OpenCV camera id.")
    width: int = Field(default=1280, gt=0, le=7680, description="Requested frame width.")
    height: int = Field(default=720, gt=0, le=4320, description="Requested frame height.")
    backend: CameraBackend = Field(default="auto", description="OpenCV camera backend.")
    codec: CameraCodec = Field(default="MJPG", description="Camera codec.")
    fps: float = Field(default=30.0, gt=0.0, le=1000.0, description="Requested capture FPS.")
    warmup_reads: int = Field(default=1, ge=0, le=30, description="Initial reads used to negotiate camera format.")

    def get_resolution(self) -> tuple[int, int, float]:
        return self.width, self.height, self.fps
