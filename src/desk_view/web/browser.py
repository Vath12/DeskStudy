"""A small but complete browser pane (QtWebEngine / Chromium).

Behaves like a normal browser for sites such as ChatGPT: logins persist between runs,
sign-in popups open in their own window, downloads go to ~/Downloads, and sites can ask
for permissions (camera, microphone, ...), which the user approves.
"""

import re
from pathlib import Path
from urllib.parse import quote_plus

from PyQt6 import sip
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWebEngineCore import (
    QWebEngineDownloadRequest, QWebEnginePage, QWebEnginePermission, QWebEngineProfile,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLineEdit, QMessageBox, QProgressBar, QToolButton, QVBoxLayout,
    QWidget,
)

from ..geometry import CONFIG_DIR

STORAGE_DIR = CONFIG_DIR / "web"
DEFAULT_URL = "https://chatgpt.com"

# Bare names people are likely to type.
SHORTCUTS = {
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
}
SEARCH_URL = "https://www.google.com/search?q={}"

# Permissions granted without asking; everything else prompts the user.
AUTO_GRANT = {QWebEnginePermission.PermissionType.ClipboardReadWrite}
PERMISSION_NAMES = {
    QWebEnginePermission.PermissionType.MediaAudioCapture: "use your microphone",
    QWebEnginePermission.PermissionType.MediaVideoCapture: "use your camera",
    QWebEnginePermission.PermissionType.MediaAudioVideoCapture: "use your camera and microphone",
    QWebEnginePermission.PermissionType.DesktopVideoCapture: "see your screen",
    QWebEnginePermission.PermissionType.DesktopAudioVideoCapture: "see your screen and hear its audio",
    QWebEnginePermission.PermissionType.Geolocation: "know your location",
    QWebEnginePermission.PermissionType.Notifications: "show notifications",
    QWebEnginePermission.PermissionType.MouseLock: "lock your mouse pointer",
}


def to_url(text: str) -> QUrl | None:
    """Turn what the user typed ("chatgpt", "example.com", a search...) into a URL."""
    text = text.strip()
    if not text:
        return None
    if text.lower() in SHORTCUTS:
        return QUrl(SHORTCUTS[text.lower()])
    if "://" in text or text.startswith(("about:", "file:")):
        return QUrl(text)
    if " " not in text:
        if text.startswith("localhost") or re.match(r"^\d+\.\d+\.\d+\.\d+(:\d+)?(/|$)", text):
            return QUrl("http://" + text)
        if "." in text:
            return QUrl("https://" + text)
        return QUrl(f"https://{text}.com")
    return QUrl(SEARCH_URL.format(quote_plus(text)))


_profile: QWebEngineProfile | None = None


def shared_profile() -> QWebEngineProfile:
    """One persistent profile for the app, so cookies and logins survive restarts."""
    global _profile
    if _profile is None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        # Owned by the application so it outlives every page that uses it.
        _profile = QWebEngineProfile("desk_view", QApplication.instance())
        _profile.setPersistentStoragePath(str(STORAGE_DIR / "storage"))
        _profile.setCachePath(str(STORAGE_DIR / "cache"))
        _profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies)
        # Look like plain Chrome: some sign-in pages (e.g. Google) refuse "QtWebEngine".
        _profile.setHttpUserAgent(re.sub(r"QtWebEngine/\S+\s*", "", _profile.httpUserAgent()))
        _profile.setDownloadPath(str(Path.home() / "Downloads"))
        _profile.downloadRequested.connect(_accept_download)
    return _profile


def _accept_download(download: QWebEngineDownloadRequest) -> None:
    download.accept()


class BrowserPage(QWebEnginePage):
    def __init__(self, profile: QWebEngineProfile, parent=None):
        super().__init__(profile, parent)
        self.permissionRequested.connect(self._on_permission)

    def javaScriptConsoleMessage(self, level, message, line, source):
        pass  # sites log plenty; don't echo it into the terminal

    def _on_permission(self, permission: QWebEnginePermission) -> None:
        kind = permission.permissionType()
        if kind in AUTO_GRANT:
            permission.grant()
            return
        what = PERMISSION_NAMES.get(kind, "use an extra browser feature")
        host = permission.origin().host() or permission.origin().toString()
        answer = QMessageBox.question(self.view() or None, "Permission request",
                                      f"{host} wants to {what}. Allow?")
        permission.grant() if answer == QMessageBox.StandardButton.Yes else permission.deny()


class BrowserView(QWebEngineView):
    """A web view whose new-window requests (sign-in popups, target=_blank) open in a
    separate window sharing the same profile."""

    popups: list["PopupWindow"] = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPage(BrowserPage(shared_profile(), self))

    def createWindow(self, kind):
        popup = PopupWindow()
        BrowserView.popups.append(popup)
        popup.show()
        return popup.view


class PopupWindow(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(520, 680)
        self.view = BrowserView(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.view.titleChanged.connect(self.setWindowTitle)
        self.view.page().windowCloseRequested.connect(self.close)  # e.g. after OAuth completes

    def closeEvent(self, event):
        if self in BrowserView.popups:
            BrowserView.popups.remove(self)
        super().closeEvent(event)


class WebPane(QWidget):
    """Navigation bar plus a browser view."""

    def __init__(self, url: str | None = None, on_url_changed=None):
        super().__init__()
        self._on_url_changed = on_url_changed

        self.view = BrowserView(self)
        self.back_btn = self._tool("◀", "Back", self.view.back)
        self.forward_btn = self._tool("▶", "Forward", self.view.forward)
        self.reload_btn = self._tool("⟳", "Reload", self.view.reload)
        self.address = QLineEdit()
        self.address.setPlaceholderText("Type a site (e.g. chatgpt) or search")
        self.address.setClearButtonEnabled(True)
        self.address.returnPressed.connect(self.navigate_to_address)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(3)
        self.progress.setTextVisible(False)
        self.progress.hide()

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        for w in (self.back_btn, self.forward_btn, self.reload_btn):
            bar.addWidget(w)
        bar.addWidget(self.address, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(bar)
        layout.addWidget(self.progress)
        layout.addWidget(self.view, 1)

        self.view.urlChanged.connect(self._url_changed)
        self.view.loadStarted.connect(lambda: self.progress.show())
        self.view.loadProgress.connect(self.progress.setValue)
        self.view.loadFinished.connect(lambda ok: self.progress.hide())
        self.view.loadFinished.connect(lambda ok: self._update_nav())
        self._update_nav()
        self.load(url or DEFAULT_URL)

    def _tool(self, text: str, tip: str, slot) -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.clicked.connect(slot)
        return b

    def load(self, text: str) -> None:
        url = to_url(text)
        if url is not None and url.isValid():
            self.view.setUrl(url)

    def navigate_to_address(self) -> None:
        self.load(self.address.text())
        self.view.setFocus()

    def _url_changed(self, url: QUrl) -> None:
        if not self.address.hasFocus():
            self.address.setText(url.toDisplayString())
        self._update_nav()
        if self._on_url_changed and url.scheme() in ("http", "https"):
            self._on_url_changed(url.toString())

    def _update_nav(self) -> None:
        history = self.view.history()
        self.back_btn.setEnabled(history.canGoBack())
        self.forward_btn.setEnabled(history.canGoForward())

    def shutdown(self) -> None:
        """Delete pages before the profile goes away (avoids Qt warnings at exit)."""
        for popup in list(BrowserView.popups):
            popup.close()
            sip.delete(popup.view.page())
        sip.delete(self.view.page())
