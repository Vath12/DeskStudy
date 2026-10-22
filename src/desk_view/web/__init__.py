"""An embedded web browser shown beside the desk view."""

import os

# Keep Chromium's internal logging out of the terminal (must be set before it loads).
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--log-level=3")

from .browser import WebPane, to_url  # noqa: E402

__all__ = ["WebPane", "to_url"]
