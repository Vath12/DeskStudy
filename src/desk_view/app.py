"""The Desk View window: live view, camera picker and in-app calibration (Qt)."""

import json
import math
import sys
import threading

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QDoubleValidator, QFont, QImage, QPainter
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QDoubleSpinBox, QMainWindow, QMenu, QProgressBar, QPushButton, QRadioButton, QSizePolicy, QVBoxLayout,
    QWidget,
)

from .board import BoardDetector, BoardSpec
from .calibrate import DeskStep, Intrinsics, LensStep, build_calibration, summary
from .camera import CameraError, list_cameras, open_camera
from .geometry import CONFIG_DIR, Calibration, build_maps, calibration_path, fit_viewport, wrap_degrees
from .scans import save_scan
from .sidebar import ScanSidebar
from .web import AI_SITES, MORE_AI_SITES, BrowserLauncher, default_browser

SETTINGS_PATH = CONFIG_DIR / "settings.json"
VCAM_SIZE = (1280, 720)  # the virtual camera needs a fixed, standard size
MAX_VIEW_PIXELS = 2_400_000  # about what the camera can resolve; more just costs time
SCAN_PIXELS = 6_000_000  # scans are rendered sharper than the live view
OUTLINE_COLOR = (0, 200, 255)

LENS_HELP = ("Hold the chart flat (tape it to cardboard) and move it around the frame: near and far, "
             "into every corner, tilted left, right, up and down. Detected squares turn red, views are "
             "captured automatically, and green shows the parts of the image covered so far.")
DESK_HELP = ("Lay the chart flat on the desk, in the middle of the area you want to show, and keep "
             "your hands clear. The chart's top edge (with the printed title) becomes the top of the view. "
             "It locks in by itself once the chart is still; Lock in forces it.")
UNCALIBRATED_HELP = ("Run ./generate_calibration_charts.sh and print a chart from the calibration_chart "
                     "folder at 100% scale, measure one black square, enter its size above, then calibrate.")


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}


class CameraCombo(QComboBox):
    """Re-scans cameras each time it opens, so newly connected ones show up."""

    def __init__(self, on_open):
        super().__init__()
        self._on_open = on_open

    def showPopup(self):
        self._on_open()
        super().showPopup()


class VideoView(QWidget):
    """Draws frames scaled to fit, sharp on Retina, and turns mouse/keys into view edits."""

    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self.image: QImage | None = None
        self.scale = 1.0  # displayed (logical) pixels per frame pixel
        self.flash = 0.0  # white flash after a scan, fades out
        self._drag: QPointF | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)

    def set_frame(self, img: np.ndarray) -> None:
        dpr = self.devicePixelRatioF()
        h, w = img.shape[:2]
        self.scale = min(self.width() / w, self.height() / h)
        size = (max(1, round(w * self.scale * dpr)), max(1, round(h * self.scale * dpr)))
        interp = cv2.INTER_AREA if size[0] < w else cv2.INTER_LINEAR
        rgb = np.ascontiguousarray(cv2.cvtColor(cv2.resize(img, size, interpolation=interp), cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, size[0], size[1], 3 * size[0], QImage.Format.Format_RGB888).copy()
        qimg.setDevicePixelRatio(dpr)
        self.image = qimg
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        if self.image is not None:
            w = self.image.width() / self.image.devicePixelRatio()
            h = self.image.height() / self.image.devicePixelRatio()
            p.drawImage(QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), self.image)
        if self.flash > 0:
            p.fillRect(self.rect(), QColor(255, 255, 255, int(200 * self.flash)))
            self.flash = max(0.0, self.flash - 0.15)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.app.view_resized(e.oldSize().width(), e.size().width())

    def mousePressEvent(self, e):
        self._drag = e.position()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.app.rotate_drag(self._drag, e.position())
            else:
                self.app.drag_view(self._drag, e.position())
            self._drag = e.position()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        notches = e.angleDelta().y() / 120
        if notches:
            self.app.zoom(1.1 ** float(np.clip(-notches, -3, 3)), e.position())

    def keyPressEvent(self, e):
        k = e.key()
        pans = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        if k in pans:
            self.app.pan(*pans[k])
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.app.zoom(1 / 1.1)
        elif k == Qt.Key.Key_Minus:
            self.app.zoom(1.1)
        elif k == Qt.Key.Key_BracketLeft:
            self.app.rotate_by(-1)
        elif k == Qt.Key.Key_BracketRight:
            self.app.rotate_by(1)
        elif k == Qt.Key.Key_Space:
            self.app.scan()
        else:
            super().keyPressEvent(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = _load_settings()
        self.cameras: list[tuple[str, str]] = []
        self.camera = None
        self.calib: Calibration | None = None
        self.maps = None
        self.maps_size = None
        self.vcam_maps = None
        self.maps_dirty = False
        self.last_frame = None
        self.step: LensStep | DeskStep | None = None
        self.vcam = None
        self.launcher = BrowserLauncher()
        self.panel_buttons: dict[str, QPushButton] = {}

        self.setWindowTitle("Desk View")
        self.resize(1200, 860)
        self._build_ui()

        self._save_timer = QTimer(self, singleShot=True, interval=800)
        self._save_timer.timeout.connect(self._save_calibration)
        self._tick = QTimer(self, interval=5)
        self._tick.timeout.connect(self.tick)
        self._tick.start()

        self.refresh_cameras()
        ids = [cid for cid, _ in self.cameras]
        if ids:
            wanted = self.settings.get("camera")
            self.camera_combo.setCurrentIndex(ids.index(wanted) if wanted in ids else 0)
            QTimer.singleShot(50, self.on_camera_selected)
        else:
            self.set_status("No cameras found.")
        self.update_controls()

    # --- layout ---------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(8)
        self.setCentralWidget(central)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Camera"))
        self.camera_combo = CameraCombo(self.refresh_cameras)
        self.camera_combo.setMinimumWidth(240)
        self.camera_combo.activated.connect(lambda _: self.on_camera_selected())
        bar.addWidget(self.camera_combo)
        bar.addSpacing(16)
        bar.addWidget(QLabel("Square size"))
        self.square_edit = QLineEdit(f"{self.settings.get('square_mm', BoardSpec().square_mm):g}")
        self.square_edit.setValidator(QDoubleValidator(5.0, 200.0, 2))
        self.square_edit.setFixedWidth(64)
        self.square_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.square_edit.editingFinished.connect(self.on_square_changed)
        bar.addWidget(self.square_edit)
        bar.addWidget(QLabel("mm"))
        bar.addSpacing(8)
        self.calibrate_btn = QPushButton("Calibrate…")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        bar.addWidget(self.calibrate_btn)
        bar.addStretch(1)

        root.addLayout(bar)

        bar = QHBoxLayout()  # second row: view controls
        self.desk_radio = QRadioButton("Desk")
        self.camera_radio = QRadioButton("Camera")
        self.view_group = QButtonGroup(self)
        self.view_group.addButton(self.desk_radio)
        self.view_group.addButton(self.camera_radio)
        self.desk_radio.setChecked(True)
        bar.addWidget(self.desk_radio)
        bar.addWidget(self.camera_radio)
        bar.addSpacing(8)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self.fit)
        bar.addWidget(self.fit_btn)
        bar.addSpacing(8)
        bar.addWidget(QLabel("Rotation"))
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setWrapping(True)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.setSingleStep(1.0)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setFixedWidth(84)
        self.rotation_spin.setToolTip("Turns the desk view clockwise to choose which way is up.\n"
                                      "0° = the chart's top edge up, as it lay during calibration. Shift+drag or [ ] on the view also rotate.")
        self.rotation_spin.setKeyboardTracking(False)  # apply on Enter / arrows, not every keystroke
        self.rotation_spin.valueChanged.connect(self.set_rotation)
        bar.addWidget(self.rotation_spin)
        self.rotate_btn = QPushButton("Rotate 180°")
        self.rotate_btn.clicked.connect(lambda: self.rotate_by(180))
        self.mirror_check = QCheckBox("Mirror")
        self.mirror_check.toggled.connect(self.toggle_mirror)
        self.vcam_check = QCheckBox("Virtual camera")
        self.vcam_check.toggled.connect(self.toggle_vcam)
        for w in (self.rotate_btn, self.mirror_check, self.vcam_check):
            bar.addWidget(w)
        bar.addStretch(1)
        root.addLayout(bar)

        # Third row: scanning, plus opening the real browser or an AI site to drop scans
        # into, on the left half of the screen (Desk View moves to the right half).
        bar = QHBoxLayout()
        browser = default_browser()
        browser_name = browser[1] if browser else "your browser"
        self.web_buttons: list[QPushButton] = []

        self.scans_btn = QPushButton("Scans")
        self.scans_btn.setCheckable(True)
        self.scans_btn.setToolTip("Show or hide the saved scans. Drag a scan into your browser to upload it.")
        self.scans_btn.toggled.connect(self.toggle_sidebar)
        bar.addWidget(self.scans_btn)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.setToolTip("Save a picture of the visible desk area (Space on the view).")
        self.scan_btn.clicked.connect(self.scan)
        bar.addWidget(self.scan_btn)
        bar.addSpacing(12)

        def web_button(label: str, url: str | None, tip: str) -> QPushButton:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda: self.open_web_page(url, label))
            bar.addWidget(b)
            self.web_buttons.append(b)
            return b

        web_button("Browser", None, f"Open a new {browser_name} window on the left half of the screen.")
        for label, url in AI_SITES:
            web_button(label, url, f"Open {label} on the left; drag scans from the sidebar into it.")
        more = web_button("More ▾", None, "Other AI sites that accept images.")
        more.clicked.disconnect()
        menu = QMenu(more)
        for label, url in MORE_AI_SITES:
            menu.addAction(label, lambda checked=False, u=url, n=label: self.open_web_page(u, n))
        more.setMenu(menu)
        bar.addStretch(1)
        root.addLayout(bar)

        # Calibration / onboarding panel, shown only when needed.
        self.panel = QFrame()
        panel = QVBoxLayout(self.panel)
        panel.setContentsMargins(0, 2, 0, 4)
        top = QHBoxLayout()
        self.panel_title = QLabel()
        f = QFont(self.panel_title.font())
        f.setBold(True)
        self.panel_title.setFont(f)
        top.addWidget(self.panel_title)
        self.panel_progress = QProgressBar()
        self.panel_progress.setFixedWidth(180)
        self.panel_progress.setTextVisible(False)
        top.addWidget(self.panel_progress)
        top.addStretch(1)
        self.panel_status = QLabel()
        self.panel_status.setStyleSheet("color: #0a7d2c;")
        top.addWidget(self.panel_status)
        panel.addLayout(top)
        self.panel_text = QLabel()
        self.panel_text.setWordWrap(True)
        panel.addWidget(self.panel_text)
        self.panel_button_row = QHBoxLayout()
        panel.addLayout(self.panel_button_row)
        root.addWidget(self.panel)
        self.panel.hide()

        # Scans sidebar on the left of the view; the view fills everything else.
        body = QHBoxLayout()
        body.setSpacing(0)
        self.sidebar = ScanSidebar()
        body.addWidget(self.sidebar)
        self.view = VideoView(self)
        body.addWidget(self.view, 1)
        root.addLayout(body, 1)
        show = bool(self.settings.get("sidebar", True))
        self.sidebar.setVisible(show)
        self.scans_btn.setChecked(show)

    def set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def show_panel(self, title: str, text: str, buttons: list[tuple[str, callable]],
                   progress: int | None = None) -> None:
        self.panel_title.setText(title)
        self.panel_text.setText(text)
        self.panel_status.setText("")
        while self.panel_button_row.count():
            item = self.panel_button_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.panel_buttons = {}
        for label, command in buttons:
            b = QPushButton(label)
            b.clicked.connect(command)
            self.panel_button_row.addWidget(b)
            self.panel_buttons[label] = b
        self.panel_button_row.addStretch(1)
        self.panel_progress.setVisible(progress is not None)
        if progress is not None:
            self.panel_progress.setRange(0, progress)
            self.panel_progress.setValue(0)
        self.panel.show()

    def update_controls(self) -> None:
        calibrating = self.step is not None
        has_view = self.calib is not None and not calibrating
        if not has_view:
            self.camera_radio.setChecked(True)
        for w in (self.desk_radio, self.fit_btn, self.rotation_spin, self.rotate_btn, self.mirror_check,
                  self.scan_btn):
            w.setEnabled(has_view)
        self.camera_radio.setEnabled(not calibrating)
        self.calibrate_btn.setEnabled(not calibrating and self.camera is not None)
        self.camera_combo.setEnabled(not calibrating)
        if self.calib is not None:
            self.mirror_check.blockSignals(True)
            self.mirror_check.setChecked(self.calib.viewport.mirror)
            self.mirror_check.blockSignals(False)
            self._show_rotation()
        if not calibrating:
            if self.calib is None and self.camera is not None:
                self.show_panel("Not calibrated", UNCALIBRATED_HELP, [("Start calibration", self.start_calibration)])
            else:
                self.panel.hide()

    # --- camera ---------------------------------------------------------------------

    def refresh_cameras(self) -> None:
        current = self.camera.unique_id if self.camera is not None else self.camera_combo.currentData()
        self.cameras = list_cameras()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for cid, name in self.cameras:
            self.camera_combo.addItem(name, cid)
        ids = [cid for cid, _ in self.cameras]
        if current in ids:
            self.camera_combo.setCurrentIndex(ids.index(current))
        self.camera_combo.blockSignals(False)

    def on_camera_selected(self) -> None:
        cid = self.camera_combo.currentData()
        if cid is None:
            return
        name = self.camera_combo.currentText()
        if self.camera is not None:
            if self.camera.unique_id == cid:
                return
            self.camera.release()
            self.camera = None
        self.set_status(f"Opening {name}…")
        QApplication.processEvents()
        try:
            self.camera = open_camera(cid)
        except CameraError as e:
            self.calib = None
            self.set_status(str(e))
            self.update_controls()
            return
        self.settings["camera"] = cid
        self._save_settings()
        self.calib = None
        path = calibration_path(cid)
        status = f"{self.camera.name}: {self.camera.description}."
        if path.exists():
            try:
                self.calib = Calibration.load(path).scaled_to(self.camera.size)
                self.square_edit.setText(f"{self.calib.square_mm:g}")
                status += " " + summary(self.calib)
                self.desk_radio.setChecked(True)
            except (ValueError, KeyError) as e:
                status += f" Saved calibration can't be used ({e}). Please recalibrate."
        self.maps_dirty = True
        self.set_status(status)
        self.update_controls()

    # --- square size ----------------------------------------------------------------

    def parse_square(self) -> float | None:
        try:
            value = float(self.square_edit.text().replace(",", "."))
        except ValueError:
            return None
        return value if 5 <= value <= 200 else None

    def on_square_changed(self) -> None:
        value = self.parse_square()
        if value is None:
            self.set_status("Square size must be a number of millimetres between 5 and 200.")
            return
        self.settings["square_mm"] = value
        self._save_settings()
        # The geometry scales exactly with the chart, so no need to recalibrate.
        if self.calib is not None and abs(value - self.calib.square_mm) > 1e-6:
            self.calib = self.calib.rescaled(value)
            self.maps_dirty = True
            self.schedule_save()
            self.set_status(f"Rescaled to {value:g} mm squares. " + summary(self.calib))

    # --- calibration flow -----------------------------------------------------------

    def start_calibration(self) -> None:
        if self.camera is None:
            return
        square = self.parse_square()
        if square is None:
            self.set_status("Enter the measured size of one printed square (in mm) first.")
            self.square_edit.setFocus()
            return
        self.on_square_changed()
        self.detector = BoardDetector(BoardSpec().with_square_mm(square))
        self.step = LensStep(self.detector, self.camera.size)
        self.camera_radio.setChecked(True)
        buttons = [("Capture now", self.step.capture_now),
                   ("Finish lens step", self.finish_lens),
                   ("Skip lens step", lambda: self.start_desk(None))]
        previous = Intrinsics.from_calibration(self.calib)
        if previous is not None:
            buttons.append(("Reuse previous lens calibration", lambda: self.start_desk(previous)))
        buttons.append(("Cancel", self.end_calibration))
        self.show_panel("Step 1 of 2: Lens", LENS_HELP, buttons, progress=self.step.target_views)
        self.update_controls()

    def finish_lens(self) -> None:
        if not isinstance(self.step, LensStep) or self.step.count < 6:
            return
        self.panel_status.setText("Solving…")
        QApplication.processEvents()
        self.start_desk(self.step.solve())

    def start_desk(self, intrinsics: Intrinsics | None) -> None:
        self.intrinsics = intrinsics
        self.step = DeskStep(self.detector, self.camera.size, intrinsics)
        note = (f" Lens: {intrinsics.views} views, {intrinsics.rms:.2f} px error." if intrinsics
                else " Lens step skipped: focal length will be estimated from this view.")
        self.show_panel("Step 2 of 2: Desk", DESK_HELP + note,
                        [("Lock in", self.step.lock), ("Cancel", self.end_calibration)])

    def finish_desk(self) -> None:
        pose = self.step.result
        self.calib = build_calibration(pose, self.detector, self.camera.size, self.desk_size(),
                                       self.intrinsics, old=self.calib)
        self.calib.save(calibration_path(self.camera.unique_id))
        self.end_calibration()
        self.desk_radio.setChecked(True)
        msg = summary(self.calib)
        if pose.rms_mm > 2.0:
            msg += "  The fit is poor: check that the chart is flat and the square size is right."
        self.set_status(msg)

    def end_calibration(self) -> None:
        self.step = None
        self.maps_dirty = True
        self.update_controls()
        if self.calib is not None:
            self.desk_radio.setChecked(True)

    def update_panel(self) -> None:
        step = self.step
        self.panel_status.setText(step.status)
        if isinstance(step, LensStep):
            self.panel_progress.setValue(step.count)
            self.panel_buttons["Finish lens step"].setEnabled(step.count >= 6)
            if step.done:
                self.finish_lens()
        elif isinstance(step, DeskStep):
            self.panel_buttons["Lock in"].setEnabled(not step.locking)
            if step.error and not step.locking:
                self.panel_status.setText(step.error)
            if step.result is not None:
                self.finish_desk()

    # --- view controls --------------------------------------------------------------

    def desk_shown(self) -> bool:
        return self.calib is not None and self.step is None and self.desk_radio.isChecked()

    def view_changed(self) -> None:
        self.maps_dirty = True
        self.schedule_save()

    def fit(self) -> None:
        if self.calib is None or self.step is not None:
            return
        spec = BoardSpec().with_square_mm(self.calib.square_mm)
        vp = self.calib.viewport
        out_w, out_h = self.desk_size()
        vp.center, vp.width_mm = fit_viewport(self.calib, spec.center_mm, spec.width_mm, out_w / out_h, out_w)
        self.view_changed()

    def _show_rotation(self) -> None:
        self.rotation_spin.blockSignals(True)
        self.rotation_spin.setValue(self.calib.viewport.rotation_deg)
        self.rotation_spin.blockSignals(False)

    def set_rotation(self, deg: float) -> None:
        """Rotate the view clockwise by `deg` from the camera-aligned "up"."""
        if self.calib is None or self.step is not None:
            return
        self.calib.viewport.rotation_deg = wrap_degrees(deg)
        self._show_rotation()
        self.view_changed()

    def rotate_by(self, deg: float) -> None:
        if self.calib is not None and self.step is None:
            self.set_rotation(self.calib.viewport.rotation_deg + deg)

    def rotate_drag(self, start: QPointF, now: QPointF) -> None:
        """Shift+drag: turn the view by the angle the cursor sweeps around its centre."""
        if not self.desk_shown():
            return
        cx, cy = self.view.width() / 2, self.view.height() / 2
        a0 = np.degrees(np.arctan2(start.y() - cy, start.x() - cx))
        a1 = np.degrees(np.arctan2(now.y() - cy, now.x() - cx))
        # Screen y points down, so a growing angle is a clockwise sweep.
        self.rotate_by(wrap_degrees(a1 - a0))

    def toggle_mirror(self, checked: bool) -> None:
        if self.calib is not None and self.step is None:
            self.calib.viewport.mirror = checked
            self.view_changed()

    def _view_to_desk(self, pos: QPointF) -> np.ndarray:
        """Desk point (mm) under a position in the video widget."""
        mm_per_px = self.calib.viewport.width_mm / self.desk_size()[0] / self.view.scale
        u = (pos.x() - self.view.width() / 2) * mm_per_px
        v = (pos.y() - self.view.height() / 2) * mm_per_px
        vp = self.calib.viewport
        right, down = vp.axes()
        return np.asarray(vp.center) + u * right + v * down

    def drag_view(self, start: QPointF, now: QPointF) -> None:
        if not self.desk_shown():
            return
        vp = self.calib.viewport
        vp.center = tuple(np.asarray(vp.center) - (self._view_to_desk(now) - self._view_to_desk(start)))
        self.view_changed()

    def zoom(self, factor: float, at: QPointF | None = None) -> None:
        if not self.desk_shown():
            return
        vp = self.calib.viewport
        new_width = float(np.clip(vp.width_mm * factor, 40.0, 3000.0))
        if at is not None:  # keep the point under the cursor fixed
            p = self._view_to_desk(at)
            vp.center = tuple(p - (p - np.asarray(vp.center)) * (new_width / vp.width_mm))
        vp.width_mm = new_width
        self.view_changed()

    def pan(self, dx: int, dy: int) -> None:
        if not self.desk_shown():
            return
        vp = self.calib.viewport
        right, down = vp.axes()
        vp.center = tuple(np.asarray(vp.center) + vp.width_mm * 0.04 * (dx * right + dy * down))
        self.view_changed()

    def toggle_vcam(self, on: bool) -> None:
        if not on:
            if self.vcam:
                self.vcam.close()
                self.vcam = None
            return
        try:
            import pyvirtualcam

            self.vcam = pyvirtualcam.Camera(*VCAM_SIZE, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            self.set_status(f"Virtual camera on: {self.vcam.device}")
        except ImportError:
            self._uncheck_vcam("Virtual camera needs the extra package:  uv sync --extra virtualcam")
        except RuntimeError as e:
            self._uncheck_vcam(f"Virtual camera unavailable: {e} (on macOS, install OBS and start its "
                               "virtual camera once).")

    def _uncheck_vcam(self, message: str) -> None:
        self.vcam_check.blockSignals(True)
        self.vcam_check.setChecked(False)
        self.vcam_check.blockSignals(False)
        self.set_status(message)

    # --- persistence ----------------------------------------------------------------

    # --- view size, scans and sidebar --------------------------------------------------

    def desk_size(self) -> tuple[int, int]:
        """Desk view render size: the view's own shape (so it fills it, no letterboxing),
        at Retina resolution but capped at what the camera can actually resolve."""
        w, h = max(self.view.width(), 16), max(self.view.height(), 16)
        s = min(self.view.devicePixelRatioF(), math.sqrt(MAX_VIEW_PIXELS / (w * h)))
        return max(16, round(w * s)), max(16, round(h * s))

    def view_resized(self, old_width: int, new_width: int) -> None:
        """Keep the desk's scale when the window changes size: a bigger window shows more."""
        if self.calib is not None and old_width > 0 and new_width > 0:
            self.calib.viewport.width_mm *= new_width / old_width
            self.view_changed()

    def scan(self) -> None:
        """Save a sharp picture of exactly the visible desk area to the scans folder."""
        if self.calib is None or self.step is not None or self.last_frame is None:
            self.set_status("Calibrate first: scans are taken from the desk view.")
            return
        w, h = self.desk_size()
        k = math.sqrt(SCAN_PIXELS / (w * h))
        size = (round(w * k), round(h * k))
        image = cv2.remap(self.last_frame, *build_maps(self.calib, size), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT)
        try:
            path = save_scan(image)
        except OSError as e:
            self.set_status(str(e))
            return
        self.view.flash = 1.0
        self.sidebar.refresh(select=path)
        if not self.scans_btn.isChecked():
            self.scans_btn.setChecked(True)
        self.set_status(f"Saved scan {path.name} ({size[0]}x{size[1]}).")

    def toggle_sidebar(self, show: bool) -> None:
        self.sidebar.setVisible(show)
        self.settings["sidebar"] = show
        self._save_settings()

    # --- web page -------------------------------------------------------------------

    def open_web_page(self, url: str | None, label: str = "Browser") -> None:
        """Open `url` (None: a blank page): the first time in a new browser window on the left
        half of the screen, after that as a new tab in that window."""
        area = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        half = area.width() // 2
        bounds = (area.x(), area.y(), area.x() + half, area.y() + area.height())

        # AppleScript can wait on a macOS permission prompt, so run it off the UI thread.
        for b in self.web_buttons:
            b.setEnabled(False)
        self.set_status(f"Opening {label}…")
        result: list[tuple[str, bool]] = []
        threading.Thread(target=lambda: result.append(self.launcher.open(url, bounds)), daemon=True).start()

        def poll():
            if not result:
                QTimer.singleShot(100, poll)
                return
            message, new_window = result[0]
            for b in self.web_buttons:
                b.setEnabled(True)
            if new_window:  # the browser took the left half; Desk View takes the right
                self.showNormal()
                title_h = self.frameGeometry().height() - self.geometry().height()
                self.setGeometry(area.x() + half, area.y() + title_h, area.width() - half,
                                 area.height() - title_h)
            self.set_status(message)

        poll()

    def _save_settings(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(self.settings, indent=2) + "\n")

    def schedule_save(self) -> None:
        self._save_timer.start()

    def _save_calibration(self) -> None:
        if self.calib is not None and self.camera is not None:
            self.calib.save(calibration_path(self.camera.unique_id))

    # --- frame loop -----------------------------------------------------------------

    def tick(self) -> None:
        if self.camera is None:
            return
        ok, frame = self.camera.read(timeout=0)
        if ok and frame is not None:
            self.render(frame)

    def render(self, frame: np.ndarray) -> None:
        if self.step is not None:
            self.view.set_frame(self.step.process(frame))
            self.update_panel()
            return

        self.last_frame = frame
        size = self.desk_size()
        if self.calib is not None and (self.maps_dirty or self.maps_size != size):
            self.maps = build_maps(self.calib, size) if self.desk_radio.isChecked() else None
            self.maps_size = size if self.maps is not None else None
            self.vcam_maps = None
            self.maps_dirty = False
        if self.vcam and self.calib is not None:
            if self.vcam_maps is None:
                self.vcam_maps = build_maps(self.calib, VCAM_SIZE)  # same view, framed 16:9
            self.vcam.send(cv2.remap(frame, *self.vcam_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))

        if self.desk_radio.isChecked() and self.maps is not None:
            self.view.set_frame(cv2.remap(frame, *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        else:
            vis = frame
            if self.calib is not None:
                vis = frame.copy()
                pix, in_front = self.calib.project_desk(self.calib.viewport.outline(*size))
                pts = pix[in_front].astype(np.int32)
                if len(pts):
                    cv2.polylines(vis, [pts], True, OUTLINE_COLOR, max(2, vis.shape[1] // 500), cv2.LINE_AA)
            self.view.set_frame(vis)

    def closeEvent(self, event):
        self._tick.stop()
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_calibration()
        if self.vcam:
            self.vcam.close()
        if self.camera is not None:
            self.camera.release()
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Desk View")
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    app.exec()
