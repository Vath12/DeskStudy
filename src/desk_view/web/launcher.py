"""Open a web page in the user's real browser, placed on the left half of the screen.

A real browser (rather than an embedded web view) keeps sign-in working everywhere:
Google, for one, refuses OAuth sign-in from embedded browsers.
"""

import subprocess
import sys
import webbrowser

# AI chat sites that accept image uploads, as (button label, URL).
AI_SITES = [
    ("ChatGPT", "https://chatgpt.com"),
    ("Claude", "https://claude.ai"),
    ("Gemini", "https://gemini.google.com"),
    ("DeepSeek", "https://chat.deepseek.com"),
]
MORE_AI_SITES = [
    ("Microsoft Copilot", "https://copilot.microsoft.com"),
    ("Perplexity", "https://www.perplexity.ai"),
    ("Grok", "https://grok.com"),
    ("Mistral Le Chat", "https://chat.mistral.ai"),
    ("Qwen", "https://chat.qwen.ai"),
]

# Browsers whose own AppleScript can open a window and set its bounds.
CHROMIUM_IDS = {"com.google.Chrome", "com.google.Chrome.beta", "com.google.Chrome.canary",
                "com.microsoft.edgemac", "com.brave.Browser", "org.chromium.Chromium",
                "com.vivaldi.Vivaldi"}
SAFARI_IDS = {"com.apple.Safari", "com.apple.SafariTechnologyPreview"}

# New-window scripts take: url ("" for the browser's own new-window page), left, top,
# right, bottom (screen points, top-left origin), and print the new window's id.
_CHROMIUM_WINDOW = """on run argv
  set {theURL, l, t, r, b} to argv
  tell application id "%s"
    activate
    set w to make new window
    if theURL is not "" then set URL of active tab of w to theURL
    set bounds of w to {l as integer, t as integer, r as integer, b as integer}
    return id of w
  end tell
end run"""
_SAFARI_WINDOW = """on run argv
  set {theURL, l, t, r, b} to argv
  tell application id "%s"
    activate
    if theURL is "" then
      make new document
    else
      make new document with properties {URL:theURL}
    end if
    set w to front window
    set bounds of w to {l as integer, t as integer, r as integer, b as integer}
    return id of w
  end tell
end run"""
# Any other browser: open normally, then move its window via System Events
# (needs Accessibility permission for the app running Desk View). No window id.
_GENERIC_WINDOW = """on run argv
  set {theURL, l, t, r, b} to argv
  if theURL is "" then
    do shell script "open -b %s"
  else
    do shell script "open -b %s " & quoted form of theURL
  end if
  delay 1.5
  tell application "System Events"
    tell (first process whose bundle identifier is "%s")
      set frontmost to true
      set position of front window to {l as integer, t as integer}
      set size of front window to {(r - l) as integer, (b - t) as integer}
    end tell
  end tell
  return ""
end run"""

# New-tab scripts take: url, window id. They fail if that window has been closed.
_CHROMIUM_TAB = """on run argv
  set {theURL, winId} to argv
  tell application id "%s"
    set w to window id (winId as integer)
    if theURL is "" then
      make new tab at end of tabs of w
    else
      make new tab at end of tabs of w with properties {URL:theURL}
    end if
    set active tab index of w to (count of tabs of w)
    set index of w to 1
    activate
  end tell
end run"""
_SAFARI_TAB = """on run argv
  set {theURL, winId} to argv
  tell application id "%s"
    set w to window id (winId as integer)
    tell w
      if theURL is "" then
        set newTab to make new tab at end of tabs
      else
        set newTab to make new tab at end of tabs with properties {URL:theURL}
      end if
      set current tab to newTab
    end tell
    set index of w to 1
    activate
  end tell
end run"""


def default_browser() -> tuple[str, str] | None:
    """(bundle id, display name) of the default web browser on macOS."""
    if sys.platform != "darwin":
        return None
    from AppKit import NSWorkspace
    from Foundation import NSURL, NSBundle

    app_url = NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(NSURL.URLWithString_("https://example.com"))
    bundle = NSBundle.bundleWithURL_(app_url) if app_url else None
    if bundle is None:
        return None
    name = bundle.objectForInfoDictionaryKey_("CFBundleDisplayName") or bundle.objectForInfoDictionaryKey_("CFBundleName")
    return str(bundle.bundleIdentifier()), str(name or app_url.path())


class BrowserLauncher:
    """Opens pages in the default browser: the first time in a new window on the left half
    of the screen, after that as new tabs in that same window (a new window again if the
    user has closed it)."""

    def __init__(self):
        self._window: tuple[str, str] | None = None  # (browser bundle id, window id)
        self._generic_opened = False

    def open(self, url: str | None, bounds: tuple[int, int, int, int]) -> tuple[str, bool]:
        """Open `url` (None: the browser's blank page). Blocks, so call it off the UI thread.
        Returns (status message, whether a new window was placed at `bounds`)."""
        url = url or ""
        browser = default_browser()
        if browser is None:
            webbrowser.open(url or "about:blank")
            return "Opened in your browser (window placement is only supported on macOS).", False
        bundle_id, name = browser

        if bundle_id in CHROMIUM_IDS | SAFARI_IDS:
            if self._window and self._window[0] == bundle_id:
                tab = (_CHROMIUM_TAB if bundle_id in CHROMIUM_IDS else _SAFARI_TAB) % bundle_id
                ok, _ = self._run(tab, [url, self._window[1]])
                if ok is True:
                    return (f"Opened {url} in a new tab." if url else "Opened a new tab."), False
                # The window was closed (or the browser quit): start over with a new one.
                self._window = None
            script = (_CHROMIUM_WINDOW if bundle_id in CHROMIUM_IDS else _SAFARI_WINDOW) % bundle_id
        else:
            if self._generic_opened:
                # Most browsers open links as a tab in their front window by themselves.
                subprocess.run(["open", "-b", bundle_id, *([url] if url else [])])
                return f"Opened {url or name} in {name}.", False
            script = _GENERIC_WINDOW % (bundle_id, bundle_id, bundle_id)

        ok, out = self._run(script, [url, *(str(int(v)) for v in bounds)])
        if ok is True:
            if out:
                self._window = (bundle_id, out)
            else:
                self._generic_opened = True
            return f"Opened {url or 'a new window'} in {name} on the left.", True
        return self._explain_failure(out, bundle_id, name, url), False

    @staticmethod
    def _run(script: str, args: list[str]) -> tuple[bool | None, str]:
        """(True, stdout) on success, (False, stderr) on failure, (None, "") on timeout."""
        try:
            r = subprocess.run(["osascript", "-e", script, *args], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None, ""
        return (True, r.stdout.strip()) if r.returncode == 0 else (False, r.stderr.strip())

    @staticmethod
    def _explain_failure(err: str, bundle_id: str, name: str, url: str) -> str:
        if not err:
            return f"{name} didn't respond. Check for a permission prompt."
        if "-1743" in err or "Not authorized" in err:
            return (f"macOS blocked controlling {name}. Allow it in System Settings → Privacy & Security → "
                    "Automation, then try again.")
        subprocess.run(["open", "-b", bundle_id, *([url] if url else [])])
        if "-1719" in err or "assistive" in err.lower():
            return (f"Opened in {name}, but placing its window needs Accessibility access "
                    "(System Settings → Privacy & Security → Accessibility).")
        return f"Opened in {name}, but couldn't place the window: {err}"
