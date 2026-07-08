"""Inference service package."""

from od_platform.inference.cancel import CancelToken, InferenceCancelled
from od_platform.inference.hooks import FrameEvent, InferHooks, ProgressEvent
from od_platform.inference.pipeline import InferStats
from od_platform.inference.pipeline_config import PipelineConfig, load_pipeline_config
from od_platform.inference.service import (
    InferenceRunResult,
    InferResult,
    InferService,
    infer_yolo,
    run_inference,
)
from od_platform.inference.sinks import LocalFileSink, NullSink, OutputSink

__all__ = [
    "CancelToken",
    "FrameEvent",
    "InferHooks",
    "InferResult",
    "InferService",
    "InferStats",
    "InferenceCancelled",
    "InferenceRunResult",
    "LocalFileSink",
    "NullSink",
    "OutputSink",
    "PipelineConfig",
    "ProgressEvent",
    "infer_yolo",
    "load_pipeline_config",
    "run_inference",
]
