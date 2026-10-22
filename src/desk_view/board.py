"""The ChArUco calibration board: its physical layout and a detector for it."""

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    """Physical layout of the printed chart. All lengths are in millimetres."""

    squares_x: int = 7
    squares_y: int = 5
    square_mm: float = 36.0
    marker_mm: float = 27.0
    dictionary: str = "DICT_4X4_50"

    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.square_mm

    @property
    def center_mm(self) -> np.ndarray:
        return np.array([self.width_mm / 2, self.height_mm / 2])

    def with_square_mm(self, square_mm: float) -> "BoardSpec":
        """The same board printed at a different scale (e.g. measured after printing)."""
        ratio = square_mm / self.square_mm
        return replace(self, square_mm=square_mm, marker_mm=self.marker_mm * ratio)

    def make(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_mm, self.marker_mm, dictionary
        )


@dataclass
class Detection:
    corners: np.ndarray  # (N, 2) float32 image points
    ids: np.ndarray  # (N,) int corner ids
    object_points: np.ndarray  # (N, 3) float32 board points in mm


class BoardDetector:
    def __init__(self, spec: BoardSpec, min_corners: int = 6):
        self.spec = spec
        self.board = spec.make()
        self.min_corners = min_corners
        self._all_object_points = self.board.getChessboardCorners().astype(np.float32)
        params = cv2.aruco.DetectorParameters()
        # On the desk the board is seen at a grazing angle, so markers are small and
        # strongly foreshortened.
        params.adaptiveThreshWinSizeMax = 53
        params.minMarkerPerimeterRate = 0.01
        params.polygonalApproxAccuracyRate = 0.05
        params.perspectiveRemovePixelPerCell = 8
        charuco = cv2.aruco.CharucoParameters()
        charuco.tryRefineMarkers = True  # recover missed markers from the known layout
        charuco.minMarkers = 1
        self.detector = cv2.aruco.CharucoDetector(
            self.board, charuco, params, cv2.aruco.RefineParameters()
        )

    @property
    def num_corners(self) -> int:
        return len(self._all_object_points)

    def detect(self, gray: np.ndarray) -> Detection | None:
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < self.min_corners:
            return None
        ids = ids.ravel()
        return Detection(
            corners=corners.reshape(-1, 2).astype(np.float32),
            ids=ids,
            object_points=self._all_object_points[ids],
        )

    def object_points(self, ids: np.ndarray) -> np.ndarray:
        return self._all_object_points[ids]

    def square_outlines(self, det: Detection) -> list[np.ndarray]:
        """Image outlines (4x2) of the black squares touching a detected corner.

        Detected inner corners are used as-is; the board's outer corners, which have no
        detection of their own, are placed with a board-to-image homography.
        """
        if len(det.ids) < 4:
            return []
        H, _ = cv2.findHomography(det.object_points[:, :2], det.corners)
        if H is None:
            return []
        sx, sy, sq = self.spec.squares_x, self.spec.squares_y, self.spec.square_mm
        gx, gy = np.meshgrid(np.arange(sx + 1), np.arange(sy + 1))
        grid = cv2.perspectiveTransform(
            (np.stack([gx, gy], -1).reshape(-1, 1, 2) * sq).astype(np.float64), H
        ).reshape(sy + 1, sx + 1, 2)
        found = np.zeros((sy + 1, sx + 1), bool)
        for i, c in zip(det.ids, det.corners):
            x, y = i % (sx - 1) + 1, i // (sx - 1) + 1
            grid[y, x] = c
            found[y, x] = True
        outlines = []
        for y in range(sy):
            for x in range(sx):
                if (x + y) % 2 == 0 and found[y:y + 2, x:x + 2].any():
                    outlines.append(np.array([grid[y, x], grid[y, x + 1], grid[y + 1, x + 1], grid[y + 1, x]]))
        return outlines
