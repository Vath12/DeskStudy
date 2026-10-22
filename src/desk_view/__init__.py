"""Desk View: a calibrated, top-down view of your desk from any webcam."""

import argparse
import sys
from pathlib import Path

CHART_DIR = Path(__file__).resolve().parents[2] / "calibration_chart"


def main() -> None:
    parser = argparse.ArgumentParser(prog="desk-view", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("chart", help="regenerate the printable calibration chart")
    p.add_argument("--out", type=Path, default=CHART_DIR)
    p.add_argument("--square-mm", type=float, default=36.0)
    args = parser.parse_args(sys.argv[1:])

    if args.command == "chart":
        from .board import BoardSpec
        from .chart import generate

        for path in generate(args.out, BoardSpec(square_mm=args.square_mm, marker_mm=args.square_mm * 0.75)):
            print(path)
        return

    from .app import run

    try:
        run()
    except KeyboardInterrupt:
        pass
