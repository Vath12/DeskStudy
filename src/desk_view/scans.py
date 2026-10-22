"""Saved desk scans: `scans/NNNN_YYYY-MM-DD_HH-MM-SS.png` in the project folder."""

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SCANS_DIR = Path(__file__).resolve().parents[2] / "scans"
_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.png$")


def list_scans() -> list[Path]:
    """Scans, newest (highest index) first."""
    if not SCANS_DIR.exists():
        return []
    found = [(int(m.group(1)), p) for p in SCANS_DIR.iterdir() if (m := _NAME.match(p.name))]
    return [p for _, p in sorted(found, reverse=True)]


def describe(path: Path) -> str:
    """Short label, e.g. "#3 · Oct 22, 15:40"."""
    m = _NAME.match(path.name)
    if not m:
        return path.stem
    when = datetime.strptime(f"{m.group(2)} {m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M")
    return f"#{int(m.group(1))} · {when:%b %-d, %H:%M}"


def save_scan(image: np.ndarray) -> Path:
    """Write a BGR image as the next numbered scan and return its path."""
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list_scans()
    index = int(_NAME.match(existing[0].name).group(1)) + 1 if existing else 1
    path = SCANS_DIR / f"{index:04d}_{datetime.now():%Y-%m-%d_%H-%M-%S}.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")
    return path
