"""Compatibility entry for teaching scripts that refer to cli.model_train."""

from od_platform.cli.train_model import build_parser, main

__all__ = ["build_parser", "main"]
