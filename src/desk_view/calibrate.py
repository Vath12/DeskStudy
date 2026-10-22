"""Calibration steps, driven one frame at a time by the app.

1. Lens: varied views of the chart give focal length and distortion (optional).
2. Desk: the chart lying flat gives the pose of the desk plane, averaged over frames.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .board import BoardDetector, Detection
from .geometry import Calibration, Viewport, fit_viewport

SQUARE_COLOR = (40, 40, 230)  # BGR red
MARKER_COLOR = (230, 120, 30)  # BGR blue
COVERAGE_COLOR = np.array([0, 160, 0])


@dataclass
class Intrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float
    views: int

    @classmethod
    def from_calibration(cls, calib: Calibration | None) -> "Intrinsics | None":
        """Multi-view intrinsics stored in an existing calibration, if it has any."""
        if calib is None or calib.intrinsics_method != "multi-view":
            return None
        return cls(calib.camera_matrix, calib.dist_coeffs,
                   calib.stats.get("intrinsics_rms_px", 0.0), calib.stats.get("intrinsics_views", 0))


@dataclass
class DeskPose:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    rms_px: float
    rms_mm: float
    corners: int
    intrinsics_method: str


def draw_squares(img: np.ndarray, detector: BoardDetector, det: Detection) -> None:
    """Paint the chart's detected black squares red and cover its markers in solid blue.

    Covering the markers matters: this image is shown on screen, and if the camera can see
    the screen, an on-screen copy of readable markers would be detected as a second chart.
    """
    squares = detector.square_outlines(det, black=True)
    markers = detector.square_outlines(det, black=False)
    if not squares and not markers:
        return
    poly = lambda outlines: [np.round(o * 16).astype(np.int32) for o in outlines]
    overlay = img.copy()
    cv2.fillPoly(overlay, poly(squares), SQUARE_COLOR, cv2.LINE_AA, shift=4)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, dst=img)
    cv2.fillPoly(img, poly(markers), MARKER_COLOR, cv2.LINE_AA, shift=4)


class _HeldDetection:
    """Keeps showing the last detection through a few dropped frames, so the overlay
    doesn't flicker when a single frame fails (motion blur, a hand passing over)."""

    HOLD_FRAMES = 3

    def __init__(self):
        self.last: Detection | None = None
        self.age = 0

    def update(self, det: Detection | None) -> Detection | None:
        if det is not None:
            self.last, self.age = det, 0
        elif self.last is not None:
            self.age += 1
            if self.age > self.HOLD_FRAMES:
                self.last = None
        return self.last


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _pose_feature(det: Detection, size: tuple[int, int]) -> np.ndarray:
    """A small descriptor of where/how the board sits in the image, for novelty checks."""
    w, h = size
    pts = det.corners / np.array([w, h])
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    # Board tilt shows up in the perspective terms of the board-to-image homography.
    H, _ = cv2.findHomography(det.object_points[:, :2] / 100.0, pts)
    persp = H[2, :2] / H[2, 2] if H is not None else np.zeros(2)
    return np.concatenate([mean, np.sqrt(np.abs(np.diag(cov))), [cov[0, 1] * 4], persp * 0.5])


# --- step 1: lens ---------------------------------------------------------------------


class LensStep:
    """Collects varied views of the chart; captures automatically when the pose is new."""

    def __init__(self, detector: BoardDetector, size: tuple[int, int], target_views: int = 15):
        self.detector = detector
        self.size = size
        self.target_views = target_views
        self.coverage = np.zeros((size[1], size[0]), np.uint8)
        self.obj_pts, self.img_pts, self.feats = [], [], []
        self.min_corners = max(10, detector.num_corners // 2)
        self.status = ""
        self._last_capture = 0.0
        self._force = False
        self._flash = 0
        self._shown = _HeldDetection()

    @property
    def count(self) -> int:
        return len(self.obj_pts)

    @property
    def done(self) -> bool:
        return self.count >= self.target_views

    def capture_now(self) -> None:
        self._force = True

    def process(self, frame: np.ndarray) -> np.ndarray:
        det = self.detector.detect(_gray(frame))
        vis = frame.copy()
        mask = self.coverage > 0
        vis[mask] = (0.6 * vis[mask] + 0.4 * COVERAGE_COLOR).astype(np.uint8)

        shown = self._shown.update(det)
        if shown is not None:
            draw_squares(vis, self.detector, shown)
        capture = False
        if det is None:
            self.status = "Chart not found."
        else:
            enough = len(det.ids) >= self.min_corners
            f = _pose_feature(det, self.size)
            novel = all(np.linalg.norm(f - g) > 0.06 for g in self.feats)
            if not enough:
                self.status = f"{len(det.ids)} corners: show more of the chart."
            elif not novel:
                self.status = "Move or tilt the chart to a new position."
            else:
                self.status = f"{len(det.ids)} corners."
                capture = time.time() - self._last_capture > 0.4
            capture = capture or self._force
        self._force = False

        if capture:
            self.obj_pts.append(det.object_points)
            self.img_pts.append(det.corners)
            self.feats.append(_pose_feature(det, self.size))
            cv2.fillConvexPoly(self.coverage, cv2.convexHull(det.corners.astype(np.int32)), 255)
            self._last_capture = time.time()
            self._flash = 3
        if self._flash:
            cv2.rectangle(vis, (0, 0), (self.size[0] - 1, self.size[1] - 1), (255, 255, 255), 16)
            self._flash -= 1
        return vis

    def solve(self) -> Intrinsics:
        return solve_intrinsics(self.obj_pts, self.img_pts, self.size)


def solve_intrinsics(obj_pts, img_pts, size) -> Intrinsics:
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    # Drop views that disagree with the rest (motion blur, a bent chart...) and re-solve.
    errs = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i) ** 2, axis=1))))
    errs = np.array(errs)
    keep = errs <= max(1.0, 3 * np.median(errs))
    if not keep.all() and keep.sum() >= 6:
        obj_pts = [o for o, k in zip(obj_pts, keep) if k]
        img_pts = [i for i, k in zip(img_pts, keep) if k]
        rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return Intrinsics(K, dist.ravel(), float(rms), len(obj_pts))


# --- step 2: desk plane ---------------------------------------------------------------


class DeskStep:
    """Locks in automatically once the chart is seen still (or on `lock()`), then
    averages its corners over several frames."""

    STILL_PX = 1.5  # median corner movement between frames that counts as "still"
    STILL_FRAMES = 12
    # Straight after the lens step the chart is usually still in the user's hand, so
    # auto-lock only arms once the chart has left the view or this many seconds have passed.
    ARM_AFTER_S = 3.0

    def __init__(self, detector: BoardDetector, size: tuple[int, int],
                 intrinsics: Intrinsics | None, frames: int = 20, timeout: float = 5.0):
        self.detector = detector
        self.size = size
        self.intrinsics = intrinsics
        self.frames = frames
        self.timeout = timeout
        self.status = ""
        self.result: DeskPose | None = None
        self.error: str | None = None
        self._locking_since: float | None = None
        self._seen: dict[int, list[np.ndarray]] = {}
        self._got = 0
        self._prev: dict[int, np.ndarray] = {}
        self._still = 0
        self._missed = 0
        self._shown = _HeldDetection()
        self._started = time.time()
        self._armed = False

    @property
    def locking(self) -> bool:
        return self._locking_since is not None

    def lock(self) -> None:
        self._locking_since = time.time()
        self._seen, self._got, self.error = {}, 0, None

    MAX_DROPOUT = 3  # missed frames tolerated without restarting the stillness count

    def _track_stillness(self, det: Detection | None) -> None:
        if det is None or len(det.ids) < max(6, self.detector.num_corners // 3):
            self._missed += 1
            if self._missed > self.MAX_DROPOUT:
                self._prev, self._still = {}, 0
            return
        self._missed = 0
        current = {int(i): c for i, c in zip(det.ids, det.corners)}
        common = current.keys() & self._prev.keys()
        moved = (np.median([np.linalg.norm(current[i] - self._prev[i]) for i in common])
                 if len(common) >= 4 else np.inf)
        self._still = self._still + 1 if moved < self.STILL_PX else 0
        self._prev = current

    def process(self, frame: np.ndarray) -> np.ndarray:
        det = self.detector.detect(_gray(frame))
        vis = frame.copy()
        shown = self._shown.update(det)
        if shown is not None:
            draw_squares(vis, self.detector, shown)
        if not self.locking:
            self._track_stillness(det)
            self._armed = self._armed or det is None or time.time() - self._started > self.ARM_AFTER_S
            if det is None:
                self.status = "Chart not found."
            elif len(det.ids) < max(6, self.detector.num_corners // 3):
                self.status = f"Only {len(det.ids)} corners found: move the chart fully into view."
            elif not self._armed:
                self.status = "Now lay the chart flat on the desk."
            elif self._still >= self.STILL_FRAMES:
                self.lock()
            else:
                self.status = "Chart found. Hold still…" if self._still else "Chart found. Keep it still to lock in."
            if not self.locking:
                return vis

        if det is not None:
            self._got += 1
            for i, c in zip(det.ids, det.corners):
                self._seen.setdefault(int(i), []).append(c)
        self.status = f"Locking in… {self._got}/{self.frames}"
        timed_out = time.time() - self._locking_since > self.timeout
        if self._got >= self.frames or timed_out:
            self._locking_since = None
            ids = np.array(sorted(i for i, v in self._seen.items() if len(v) >= self._got / 2))
            if self._got < self.frames // 2 or len(ids) < 6:
                self.error = "Lost sight of the chart. Keep it flat, fully visible and still."
            else:
                corners = np.array([np.median(self._seen[i], axis=0) for i in ids], np.float32)
                try:
                    self.result = solve_desk_pose(corners, ids, self.detector, self.size, self.intrinsics)
                except (RuntimeError, cv2.error):
                    self.error = "Couldn't solve the desk position. Make sure the chart is flat."
            self._prev, self._still = {}, 0  # a failed attempt retries once the chart is still again
        return vis


def solve_desk_pose(corners, ids, detector: BoardDetector, size,
                    intrinsics: Intrinsics | None) -> DeskPose:
    obj = detector.object_points(ids)
    w, h = size
    if intrinsics is not None:
        K, dist, method = intrinsics.camera_matrix, intrinsics.dist_coeffs, "multi-view"
    else:
        # A typical webcam/phone focal length (about 65° across), used when it can't be measured.
        K0 = np.array([[0.8 * w, 0, w / 2], [0, 0.8 * w, h / 2], [0, 0, 1]], np.float64)
        dist = np.zeros(5)
        # One oblique view of a plane pins down the focal length well (given a centred
        # principal point and no distortion); a chart seen nearly face-on does not, as its
        # image then looks the same at any focal length. Overhead cameras hit this case.
        ok, rvec0, _ = cv2.solvePnP(obj, corners, K0, dist, flags=cv2.SOLVEPNP_IPPE)
        tilt = np.degrees(np.arccos(min(1.0, abs(cv2.Rodrigues(rvec0)[0][2, 2])))) if ok else 0.0
        K, method = K0, "assumed"
        if tilt >= 25:
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
                     | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
                     | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
            try:
                # calibrateCamera writes into the matrix it is given, so pass a copy.
                _, K1, _, _, _ = cv2.calibrateCamera([obj], [corners], size, K0.copy(), np.zeros(5), flags=flags)
                if 0.25 * w < K1[0, 0] < 6 * w:
                    K, method = K1, "single-view"
            except cv2.error:
                pass

    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("Could not solve the desk pose.")
    rvec, tvec = cv2.solvePnPRefineLM(obj, corners, K, dist, rvec, tvec)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms_px = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    tmp = Calibration((w, h), K, dist, rvec, tvec, detector.spec.square_mm, Viewport((0, 0), 1, 0))
    desk = tmp.image_to_desk(corners)
    rms_mm = float(np.sqrt(np.mean(np.sum((desk - obj[:, :2]) ** 2, axis=1))))
    return DeskPose(K, dist, rvec, tvec, rms_px, rms_mm, len(ids), method)


def build_calibration(pose: DeskPose, detector: BoardDetector, size, out_size,
                      intrinsics: Intrinsics | None, old: Calibration | None = None) -> Calibration:
    spec = detector.spec
    calib = Calibration(
        image_size=tuple(size), camera_matrix=pose.camera_matrix, dist_coeffs=pose.dist_coeffs,
        rvec=pose.rvec, tvec=pose.tvec, square_mm=spec.square_mm,
        viewport=Viewport(tuple(spec.center_mm), 400.0, 0.0),
        intrinsics_method=pose.intrinsics_method,
        stats={
            "desk_rms_px": round(pose.rms_px, 3),
            "desk_rms_mm": round(pose.rms_mm, 3),
            "desk_corners": pose.corners,
        },
    )
    if intrinsics is not None:
        calib.stats.update(intrinsics_rms_px=round(intrinsics.rms, 3), intrinsics_views=intrinsics.views)
    calib.viewport.angle_deg = calib.default_angle()
    if old is not None:  # keep the user's orientation choices across recalibration
        calib.viewport.mirror = old.viewport.mirror
        calib.viewport.rotation_deg = old.viewport.rotation_deg
    center, width = fit_viewport(calib, spec.center_mm, spec.width_mm, out_size[0] / out_size[1], out_size[0])
    calib.viewport.center, calib.viewport.width_mm = center, width
    return calib


def summary(calib: Calibration) -> str:
    s = calib.stats
    fov = np.degrees(2 * np.arctan(calib.image_size[0] / 2 / calib.camera_matrix[0, 0]))
    fit = f"desk fit {s.get('desk_rms_mm', 0):.2f} mm"
    height = abs(calib.camera_position()[2]) / 10
    if calib.intrinsics_method == "assumed":
        return (f"Calibrated: {fit}. The camera faces the desk almost straight on, so its focal length "
                f"can't be measured from the desk alone (assumed ~{fov:.0f}°, camera ~{height:.0f} cm up); "
                "the lens step removes lens distortion for best accuracy.")
    lens = (f"lens from {s['intrinsics_views']} views ({s['intrinsics_rms_px']:.2f} px)"
            if "intrinsics_rms_px" in s else "lens estimated from desk view")
    return f"Calibrated: {lens}, {fov:.0f}° horizontal FOV, {fit}, camera {height:.0f} cm above the desk."
