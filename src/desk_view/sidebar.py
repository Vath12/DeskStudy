"""Collapsible sidebar listing scans as thumbnails; drag one into any app to drop the file."""

from pathlib import Path

from PyQt6.QtCore import QFile, QFileSystemWatcher, QMimeData, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QDrag, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QFrame, QLabel, QListWidget, QListWidgetItem, QMenu, QVBoxLayout,
)

from .scans import SCANS_DIR, describe, list_scans

THUMB = QSize(150, 110)


class ScanList(QListWidget):
    """Thumbnail list whose drags carry the scan files themselves (file URLs), so they can
    be dropped onto a browser upload area, Finder, Mail and so on."""

    def __init__(self):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.TopToBottom)
        self.setWrapping(False)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setIconSize(THUMB)
        self.setSpacing(6)
        self.setWordWrap(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemDoubleClicked.connect(lambda item: self._open(item))

    @staticmethod
    def path_of(item: QListWidgetItem) -> Path:
        return Path(item.data(Qt.ItemDataRole.UserRole))

    def mimeTypes(self):
        return ["text/uri-list"]

    def mimeData(self, items):
        data = QMimeData()
        data.setUrls([QUrl.fromLocalFile(str(self.path_of(i))) for i in items])
        return data

    def startDrag(self, actions):
        items = self.selectedItems()
        if not items:
            return
        drag = QDrag(self)
        drag.setMimeData(self.mimeData(items))
        drag.setPixmap(items[0].icon().pixmap(QSize(110, 80)))
        drag.exec(Qt.DropAction.CopyAction)

    def _open(self, item: QListWidgetItem) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path_of(item))))

    def _menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        open_action = QAction("Open", menu)
        open_action.triggered.connect(lambda: self._open(item))
        reveal = QAction("Show in Finder", menu)
        reveal.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path_of(item).parent))))
        trash = QAction("Move to Trash", menu)
        trash.triggered.connect(lambda: [QFile.moveToTrash(str(self.path_of(i))) for i in self.selectedItems() or [item]])
        for a in (open_action, reveal, trash):
            menu.addAction(a)
        menu.exec(self.viewport().mapToGlobal(pos))


class ScanSidebar(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(THUMB.width() + 34)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(4)
        self.header = QLabel()
        self.hint = QLabel("Drag a scan into your browser to upload it.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray; font-size: 11px;")
        self.list = ScanList()
        layout.addWidget(self.header)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.hint)
        self._thumbs: dict[tuple[str, float], QIcon] = {}

        # Refresh when files appear or vanish (our scans, or the user deleting some).
        SCANS_DIR.mkdir(parents=True, exist_ok=True)
        self.watcher = QFileSystemWatcher([str(SCANS_DIR)], self)
        self._debounce = QTimer(self, singleShot=True, interval=150)
        self._debounce.timeout.connect(self.refresh)
        self.watcher.directoryChanged.connect(lambda _: self._debounce.start())
        self.refresh()

    def _icon(self, path: Path) -> QIcon:
        key = (str(path), path.stat().st_mtime)
        if key not in self._thumbs:
            pm = QPixmap(str(path))
            self._thumbs[key] = QIcon(pm.scaled(THUMB, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        return self._thumbs[key]

    def refresh(self, select: Path | None = None) -> None:
        """Rebuild the list; keeps the current selection unless `select` names a new one."""
        keep = {select} if select is not None else {
            self.list.path_of(i) for i in self.list.selectedItems() if i.data(Qt.ItemDataRole.UserRole)}
        scans = [p for p in list_scans() if p.exists()]
        self.list.clear()
        for path in scans:
            item = QListWidgetItem(self._icon(path), describe(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(path.name)
            item.setSizeHint(QSize(THUMB.width() + 8, THUMB.height() + 30))
            self.list.addItem(item)
            if path in keep:
                item.setSelected(True)
                if path == select:
                    self.list.scrollToItem(item)
        self.header.setText(f"<b>Scans</b> ({len(scans)})")
        self.hint.setVisible(bool(scans))
        if not scans:
            empty = QListWidgetItem("No scans yet.\nClick Scan to take one.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(empty)
