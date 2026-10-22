"""Camera model, desk-plane pose and the top-down viewport, plus persistence."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

CONFIG_DIR = Path.home() / ".config" / "desk_view"


def calibration_path(camera_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in camera_id)
    return CONFIG_DIR / "cameras" / f"{safe}.json"


@dataclass
class Viewport:
    """A rectangle on the desk, in board millimetres, that becomes the output image.

    `angle_deg` is the direction (in the board plane) that appears at the top of the
    output. By default it points from the board towards the camera, so the view is
    upright for the person sitting at the desk.
    """

    center: tuple[float, float]
    width_mm: float
    angle_deg: float
    mirror: bool = False

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors on the desk for output +x (right) and +y (down)."""
        a = math.radians(self.angle_deg)
        down = -np.array([math.cos(a), math.sin(a)])
        right = np.array([down[1], -down[0]])
        return (-right if self.mirror else right), down

    def desk_points(self, out_w: int, out_h: int, xs=None, ys=None) -> np.ndarray:
        """Desk coordinates (mm) of output pixel centres, shape (len(ys), len(xs), 2).

        `xs`/`ys` are output pixel indices (may be fractional); default is every pixel.
        """
        right, down = self.axes()
        height_mm = self.width_mm * out_h / out_w
        xs = np.arange(out_w) if xs is None else np.asarray(xs, np.float64)
        ys = np.arange(out_h) if ys is None else np.asarray(ys, np.float64)
        u = ((xs + 0.5) / out_w - 0.5) * self.width_mm
        v = ((ys + 0.5) / out_h - 0.5) * height_mm
        uu, vv = np.meshgrid(u, v)
        return (
            np.asarray(self.center)
            + uu[..., None] * right
            + vv[..., None] * down
        )

    def outline(self, out_w: int, out_h: int, samples: int = 40) -> np.ndarray:
        """Points along the viewport border on the desk, shape (4 * samples, 2)."""
        right, down = self.axes()
        hw, hh = self.width_mm / 2, self.width_mm * out_h / out_w / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts = []
        for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
            for t in np.linspace(0, 1, samples, endpoint=False):
                x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                pts.append(np.asarray(self.center) + x * right + y * down)
        return np.array(pts)


@dataclass
class Calibration:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    square_mm: float
    viewport: Viewport
    intrinsics_method: str = "single-view"
    stats: dict = field(default_factory=dict)

    # --- projection -----------------------------------------------------------------

    @property
    def rotation(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    def camera_position(self) -> np.ndarray:
        """Camera centre in board coordinates (mm)."""
        return (-self.rotation.T @ self.tvec.reshape(3)).ravel()

    def project_desk(self, desk_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project desk points (..., 2) into the image. Returns (pixels, in_front_mask)."""
        shape = desk_xy.shape[:-1]
        pts = np.zeros((int(np.prod(shape)), 3), np.float64)
        pts[:, :2] = desk_xy.reshape(-1, 2)
        cam_z = (pts @ self.rotation.T + self.tvec.reshape(1, 3))[:, 2]
        img, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        return img.reshape(*shape, 2), (cam_z > 1e-6).reshape(shape)

    def image_to_desk(self, pixels: np.ndarray) -> np.ndarray:
        """Back-project image points (N, 2) onto the desk plane, returning (N, 2) mm."""
        norm = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2).astype(np.float64), self.camera_matrix, self.dist_coeffs
        ).reshape(-1, 2)
        rays = np.column_stack([norm, np.ones(len(norm))]) @ self.rotation  # R^T @ ray
        origin = self.camera_position()
        s = -origin[2] / rays[:, 2]
        return (origin + s[:, None] * rays)[:, :2]

    def scaled_to(self, image_size: tuple[int, int]) -> "Calibration":
        """Adapt intrinsics to a different capture resolution with the same aspect ratio."""
        if tuple(image_size) == tuple(self.image_size):
            return self
        sx, sy = image_size[0] / self.image_size[0], image_size[1] / self.image_size[1]
        if abs(sx - sy) > 0.01:
            raise ValueError(
                f"Camera delivers {image_size[0]}x{image_size[1]} but calibration was made at "
                f"{self.image_size[0]}x{self.image_size[1]}; recalibrate at this resolution."
            )
        k = self.camera_matrix.copy()
        k[0] *= sx
        k[1] *= sy
        return Calibration(**{**self.__dict__, "image_size": tuple(image_size), "camera_matrix": k})

    def rescaled(self, square_mm: float) -> "Calibration":
        """The same calibration for a chart printed at a different scale.

        Scaling the board by s scales the desk and the camera translation by s while leaving
        every image projection unchanged, so this is exact.
        """
        s = square_mm / self.square_mm
        vp = self.viewport
        stats = dict(self.stats)
        if "desk_rms_mm" in stats:
            stats["desk_rms_mm"] = round(stats["desk_rms_mm"] * s, 3)
        return Calibration(**{
            **self.__dict__, "tvec": self.tvec * s, "square_mm": square_mm, "stats": stats,
            "viewport": Viewport(tuple(np.asarray(vp.center) * s), vp.width_mm * s, vp.angle_deg, vp.mirror),
        })

    def default_angle(self) -> float:
        """Viewport angle that keeps the view level with the camera (and so the screen).

        The camera faces the user, so its image-right is the user's left: the output's
        right axis is the camera's x axis, flipped, laid onto the desk.
        """
        right = -self.rotation[0, :2]
        return math.degrees(math.atan2(-right[0], right[1]))

    # --- persistence ----------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image_size": list(self.image_size),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.ravel().tolist(),
            "rvec": self.rvec.ravel().tolist(),
            "tvec": self.tvec.ravel().tolist(),
            "square_mm": self.square_mm,
            "viewport": asdict(self.viewport),
            "intrinsics_method": self.intrinsics_method,
            "stats": self.stats,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        d = json.loads(path.read_text())
        vp = d["viewport"]
        return cls(
            image_size=tuple(d["image_size"]),
            camera_matrix=np.array(d["camera_matrix"], np.float64),
            dist_coeffs=np.array(d["dist_coeffs"], np.float64),
            rvec=np.array(d["rvec"], np.float64).reshape(3, 1),
            tvec=np.array(d["tvec"], np.float64).reshape(3, 1),
            square_mm=d["square_mm"],
            viewport=Viewport(tuple(vp["center"]), vp["width_mm"], vp["angle_deg"], vp["mirror"]),
            intrinsics_method=d.get("intrinsics_method", "single-view"),
            stats=d.get("stats", {}),
        )


def _upsample(grid: np.ndarray, step: int, out_w: int, out_h: int) -> np.ndarray:
    """Bilinearly interpolate a grid sampled every `step` pixels up to full size."""
    fx, fy = np.arange(out_w) / step, np.arange(out_h) / step
    x0 = np.minimum(fx.astype(int), grid.shape[1] - 2)
    y0 = np.minimum(fy.astype(int), grid.shape[0] - 2)
    tx, ty = (fx - x0)[None, :, None], (fy - y0)[:, None, None]
    rows = grid[:, x0] * (1 - tx) + grid[:, x0 + 1] * tx
    return rows[y0] * (1 - ty) + rows[y0 + 1] * ty


def build_maps(calib: Calibration, out_size: tuple[int, int], step: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Remap tables that undistort and rectify the desk in one step.

    The mapping is smooth, so it is projected on a coarse grid and interpolated, which
    keeps panning/zooming responsive at well under 0.1 px error.
    """
    out_w, out_h = out_size
    xs = np.arange(0, out_w - 1 + step, step)
    ys = np.arange(0, out_h - 1 + step, step)
    pixels, in_front = calib.project_desk(calib.viewport.desk_points(out_w, out_h, xs, ys))
    pixels[~in_front] = np.nan
    full = _upsample(pixels, step, out_w, out_h).astype(np.float32)
    # Points near the horizon project absurdly far away; outside the frame they are just
    # black, and leaving them would overflow the 16-bit fixed-point maps.
    w, h = calib.image_size
    outside = (np.isnan(full).any(axis=-1) | (full[..., 0] < -2) | (full[..., 0] > w + 1)
               | (full[..., 1] < -2) | (full[..., 1] > h + 1))
    full[outside] = -1
    return cv2.convertMaps(full[..., 0], full[..., 1], cv2.CV_16SC2)


def fit_viewport(
    calib: Calibration,
    board_center: np.ndarray,
    board_width_mm: float,
    aspect: float,
    out_width_px: int,
    max_width_mm: float = 900.0,
    min_sharpness: float = 0.35,
) -> tuple[tuple[float, float], float]:
    """Largest viewport (at the current angle) that is fully visible and not too blurry.

    A desk point counts as usable when the camera samples it at least `min_sharpness`
    source pixels per output pixel. Returns (center, width_mm).
    """
    step = 4.0  # mm per grid cell
    half = 1200.0
    vp = calib.viewport
    right, down = vp.axes()
    coords = np.arange(-half, half + step, step)
    uu, vv = np.meshgrid(coords, coords)
    desk = board_center + uu[..., None] * right + vv[..., None] * down
    pix, in_front = calib.project_desk(desk)
    w, h = calib.image_size
    inside = in_front & (pix[..., 0] >= 0) & (pix[..., 0] < w) & (pix[..., 1] >= 0) & (pix[..., 1] < h)

    # Source pixels per mm along both output axes (worst case).
    du = np.linalg.norm(np.diff(pix, axis=1), axis=-1) / step
    dv = np.linalg.norm(np.diff(pix, axis=0), axis=-1) / step
    density = np.minimum(np.pad(du, ((0, 0), (0, 1)), mode="edge"), np.pad(dv, ((0, 1), (0, 0)), mode="edge"))
    density[~inside] = 0

    origin = len(coords) // 2

    def feasible_centers(width):
        needed = min_sharpness * out_width_px / width  # source px per mm
        mask = (density >= needed).astype(np.uint8)
        kw = max(1, int(round(width / step))) | 1
        kh = max(1, int(round(width / aspect / step))) | 1
        if kw >= mask.shape[1] or kh >= mask.shape[0]:
            return np.zeros_like(mask)
        return cv2.erode(mask, np.ones((kh, kw), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)

    widths = [max_width_mm * 0.95**i for i in range(60) if max_width_mm * 0.95**i >= 60]
    # Prefer a view centred on the chart, as long as that is not much smaller than the chart.
    for width in widths:
        if width < board_width_mm * 1.1:
            break
        if feasible_centers(width)[origin, origin]:
            return (float(board_center[0]), float(board_center[1])), float(width)
    # Otherwise take the largest view anywhere, as close to the chart as possible.
    for width in widths:
        ys, xs = np.nonzero(feasible_centers(width))
        if len(xs):
            i = np.argmin((xs - origin) ** 2 + (ys - origin) ** 2)
            c = desk[ys[i], xs[i]]
            return (float(c[0]), float(c[1])), float(width)
    return (float(board_center[0]), float(board_center[1])), 300.0
