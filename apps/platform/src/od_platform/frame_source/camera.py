"""Camera frame source with OpenCV backend negotiation."""

from __future__ import annotations

import logging
import os
import time

import cv2

from od_platform.frame_source.base import FrameSource
from od_platform.frame_source.config import CameraConfig
from od_platform.frame_source.types import Frame, FrameInfo, SourceType

logger = logging.getLogger(__name__)


class CameraSource(FrameSource):
    """OpenCV camera source with backend, codec, FPS and resolution control."""

    _BACKENDS = {
        "auto": cv2.CAP_ANY,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
        "v4l2": cv2.CAP_V4L2,
    }

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        super().__init__(str(self.config.camera_id))
        self._cap: cv2.VideoCapture | None = None
        self._width = 0
        self._height = 0
        self._fps = 0.0

    def open(self) -> bool:
        if self.config.backend == "msmf":
            os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

        self._cap = cv2.VideoCapture(self.config.camera_id, self._get_backend())
        if not self._cap.isOpened():
            logger.error("Failed to open camera: %s", self.config.camera_id)
            return False

        # Keep this order: resolution -> codec -> FPS. Some Windows camera drivers
        # renegotiate FPS when FOURCC is set before the requested resolution.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.codec))
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        for _ in range(self.config.warmup_reads):
            ret, _ = self._cap.read()
            if not ret:
                logger.warning("Camera negotiation warmup read failed; actual parameters may be inaccurate")
                break

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._warn_if_negotiation_changed()

        self._frame_index = 0
        self._start_time = time.time()
        logger.info(
            "Camera opened: id=%s backend=%s codec=%s target=%sx%s@%.1ffps actual=%sx%s@%.1ffps",
            self.config.camera_id,
            self.config.backend,
            self.config.codec,
            self.config.width,
            self.config.height,
            self.config.fps,
            self._width,
            self._height,
            self._fps,
        )
        return True

    def read(self) -> Frame | None:
        if self._cap is None:
            return None
        ret, image = self._cap.read()
        if not ret:
            return None

        index = self._frame_index
        self._frame_index += 1
        return Frame(
            image=image,
            info=FrameInfo(
                width=self._width,
                height=self._height,
                source_type=SourceType.CAMERA,
                source_path=self.source_path,
                frame_index=index,
                timestamp=time.time() - self._start_time,
                fps=self._fps,
                filename=f"camera:{self.config.camera_id}",
                metadata={
                    "backend": self.config.backend,
                    "codec": self.config.codec,
                    "requested_width": self.config.width,
                    "requested_height": self.config.height,
                    "requested_fps": self.config.fps,
                    "actual_width": self._width,
                    "actual_height": self._height,
                    "actual_fps": self._fps,
                },
            ),
        )

    def set_stride(self, stride: int) -> None:
        if stride != 1:
            logger.warning("Camera source does not support stride; received stride=%s and forced to 1", stride)
        self._stride = 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera closed: %s", self.config.camera_id)

    def get_source_type(self) -> SourceType:
        return SourceType.CAMERA

    def metadata(self) -> dict[str, object]:
        payload = super().metadata()
        payload.update(
            {
                "camera_id": self.config.camera_id,
                "backend": self.config.backend,
                "codec": self.config.codec,
                "requested_resolution": self.config.get_resolution(),
                "actual_width": self._width,
                "actual_height": self._height,
                "actual_fps": self._fps,
            }
        )
        return payload

    def _get_backend(self) -> int:
        return self._BACKENDS[self.config.backend]

    def _warn_if_negotiation_changed(self) -> None:
        if self._width != self.config.width or self._height != self.config.height:
            logger.warning(
                "Camera resolution request changed by backend: expected=%sx%s actual=%sx%s",
                self.config.width,
                self.config.height,
                self._width,
                self._height,
            )
        if self._fps > 0 and self._fps < self.config.fps * 0.9:
            logger.warning(
                "Camera FPS request changed by backend: expected=%.1f actual=%.1f",
                self.config.fps,
                self._fps,
            )


CameraFrameSource = CameraSource
