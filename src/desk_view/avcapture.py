"""Direct AVFoundation capture (macOS) configured for the widest possible field of view.

OpenCV's AVFoundation backend uses the session's default preset, which on recent Macs means
a cropped, 1.5x-zoomed 16:9 stream. Here we pick the format that covers the most of the
sensor, reset zoom to its minimum, and turn off Center Stage and geometric distortion
correction (both crop; our own calibration handles lens distortion instead).
"""

import threading

import AVFoundation as AV
import CoreMedia
import dispatch
import numpy as np
import objc
import Quartz
from Foundation import NSObject

_SKIP_TYPES = {getattr(AV, "AVCaptureDeviceTypeDeskViewCamera", None)}


def list_devices() -> list[tuple[str, str]]:
    """(unique_id, name) for every video camera, built-in first."""
    names = ("AVCaptureDeviceTypeBuiltInWideAngleCamera", "AVCaptureDeviceTypeContinuityCamera",
             "AVCaptureDeviceTypeExternal", "AVCaptureDeviceTypeExternalUnknown")
    types = [getattr(AV, n) for n in names if hasattr(AV, n)]
    session = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types, AV.AVMediaTypeVideo, AV.AVCaptureDevicePositionUnspecified)
    return [(str(d.uniqueID()), str(d.localizedName()))
            for d in session.devices() if d.deviceType() not in _SKIP_TYPES]


def _dims(fmt) -> tuple[int, int]:
    d = CoreMedia.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
    return int(d.width), int(d.height)


def _max_fps(fmt) -> float:
    return max((r.maxFrameRate() for r in fmt.videoSupportedFrameRateRanges()), default=0.0)


def widest_format(device, max_width: int = 2600):
    """The format with the largest field of view, then the most pixels (within max_width)."""
    candidates = []
    for fmt in device.formats():
        w, h = _dims(fmt)
        if h > w:  # portrait variants are rotated crops
            continue
        fov = float(fmt.videoFieldOfView() or 0.0)
        candidates.append((round(fov), w <= max_width, _max_fps(fmt) >= 24, w * h, fmt))
    if not candidates:
        return device.activeFormat()
    return max(candidates, key=lambda c: c[:4])[-1]


class _FrameSink(NSObject):
    def init(self):
        self = objc.super(_FrameSink, self).init()
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        pb = CoreMedia.CMSampleBufferGetImageBuffer(sample)
        if pb is None:
            return
        Quartz.CVPixelBufferLockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            w, h = Quartz.CVPixelBufferGetWidth(pb), Quartz.CVPixelBufferGetHeight(pb)
            stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
            buf = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(stride * h)
            frame = np.frombuffer(buf, np.uint8).reshape(h, stride // 4, 4)[:, :w, :3].copy()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
        with self.cond:
            self.frame = frame
            self.seq += 1
            self.cond.notify_all()


class AVCamera:
    """Camera with a cv2.VideoCapture-like `read()`; frames are BGR uint8."""

    def __init__(self, unique_id: str):
        device = AV.AVCaptureDevice.deviceWithUniqueID_(unique_id)
        if device is None:
            raise RuntimeError("Camera not found (was it disconnected?).")
        self.device = device
        self.name = str(device.localizedName())
        self.unique_id = unique_id

        # Center Stage is a per-app setting; take control of it and switch it off.
        if hasattr(AV.AVCaptureDevice, "setCenterStageControlMode_"):
            AV.AVCaptureDevice.setCenterStageControlMode_(AV.AVCaptureCenterStageControlModeApp)
            AV.AVCaptureDevice.setCenterStageEnabled_(False)

        self.session = AV.AVCaptureSession.alloc().init()
        inp, err = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if inp is None:
            raise RuntimeError(f"Could not open {self.name}: {err}")
        out = AV.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        out.setAlwaysDiscardsLateVideoFrames_(True)
        self._sink = _FrameSink.alloc().init()
        self._queue = dispatch.dispatch_queue_create(b"desk_view.capture", None)
        out.setSampleBufferDelegate_queue_(self._sink, self._queue)

        self.session.beginConfiguration()
        self.session.addInput_(inp)
        self.session.addOutput_(out)
        ok, err = device.lockForConfiguration_(None)
        if not ok:
            raise RuntimeError(f"Could not configure {self.name}: {err}")
        fmt = widest_format(device)
        device.setActiveFormat_(fmt)
        if device.respondsToSelector_("setVideoZoomFactor:"):
            device.setVideoZoomFactor_(device.minAvailableVideoZoomFactor())
        if device.respondsToSelector_("isGeometricDistortionCorrectionSupported") \
                and device.isGeometricDistortionCorrectionSupported():
            device.setGeometricDistortionCorrectionEnabled_(False)
        self.session.commitConfiguration()
        # Holding the lock through startRunning stops the session overriding our format.
        self.session.startRunning()
        device.unlockForConfiguration()

        self.size = _dims(fmt)
        zoom = device.videoZoomFactor() if device.respondsToSelector_("videoZoomFactor") else 1.0
        fov = fmt.videoFieldOfView()
        self.description = f"{self.size[0]}x{self.size[1]}, zoom {zoom:g}x" + (f", {fov:.0f}° FOV" if fov else "")
        self._last_seq = 0
        if self.read(timeout=5.0)[0] is False:
            self.release()
            raise RuntimeError(f"{self.name} delivered no frames. Check camera permissions for your terminal.")

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        """Wait for a frame newer than the last one returned."""
        with self._sink.cond:
            if not self._sink.cond.wait_for(lambda: self._sink.seq > self._last_seq, timeout):
                return False, None
            self._last_seq = self._sink.seq
            return True, self._sink.frame

    def latest(self) -> np.ndarray | None:
        """The newest frame without waiting (may repeat)."""
        with self._sink.cond:
            self._last_seq = self._sink.seq
            return self._sink.frame

    def release(self) -> None:
        if self.session.isRunning():
            self.session.stopRunning()
