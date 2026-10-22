"""Camera discovery and capture. Uses AVFoundation directly on macOS, OpenCV elsewhere."""

import sys

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class OpenCVCamera:
    """Fallback for non-macOS systems: asks for a large frame and takes what it gets."""

    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise CameraError(f"Could not open camera {index}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4096)
        ok, frame = self.cap.read()
        if not ok:
            raise CameraError(f"Camera {index} delivered no frames.")
        self.name = f"Camera {index}"
        self.unique_id = f"cv{index}"
        self.size = (frame.shape[1], frame.shape[0])
        self.description = f"{self.size[0]}x{self.size[1]}"

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


def list_cameras() -> list[tuple[str, str]]:
    """(unique_id, display name) for each camera."""
    if sys.platform == "darwin":
        from .avcapture import list_devices

        return list_devices()
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            found.append((f"cv{i}", f"Camera {i}"))
        cap.release()
    return found


def open_camera(unique_id: str):
    try:
        if unique_id.startswith("cv"):
            return OpenCVCamera(int(unique_id[2:]))
        from .avcapture import AVCamera

        return AVCamera(unique_id)
    except CameraError:
        raise
    except Exception as e:
        raise CameraError(str(e)) from e
