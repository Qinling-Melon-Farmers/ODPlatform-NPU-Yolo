"""Command line entry for plotting Ultralytics training results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.training.metrics import write_metrics_summary
from od_platform.training.plots import plot_training_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odp-plot-training", description="Plot and summarize Ultralytics results.csv.")
    parser.add_argument("csv_path", type=Path, help="Path to Ultralytics results.csv.")
    parser.add_argument("--output", "-o", type=Path, help="Output image path.")
    parser.add_argument("--summary", type=Path, help="Output summary json path.")
    parser.add_argument("--matplotx", action="store_true", help="Use matplotx style when installed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warn_cli_if_not_expected_environment()
    try:
        plot_training_results(args.csv_path, args.output, use_matplotx=args.matplotx)
        if args.summary:
            write_metrics_summary(args.csv_path, args.summary)
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
