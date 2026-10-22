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

# All scripts take: url ("" for the browser's own new-window page), left, top, right,
# bottom (screen points, top-left origin).
_CHROMIUM_SCRIPT = """on run argv
  set {theURL, l, t, r, b} to argv
  tell application id "%s"
    activate
    set w to make new window
    if theURL is not "" then set URL of active tab of w to theURL
    set bounds of w to {l as integer, t as integer, r as integer, b as integer}
  end tell
end run"""
_SAFARI_SCRIPT = """on run argv
  set {theURL, l, t, r, b} to argv
  tell application id "%s"
    activate
    if theURL is "" then
      make new document
    else
      make new document with properties {URL:theURL}
    end if
    set bounds of front window to {l as integer, t as integer, r as integer, b as integer}
  end tell
end run"""
# Any other browser: open normally, then move its window via System Events
# (needs Accessibility permission for the app running Desk View).
_GENERIC_SCRIPT = """on run argv
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


def open_on_left(url: str | None, bounds: tuple[int, int, int, int]) -> str:
    """Open `url` (or just a new browser window, if None) in the default browser, in a
    window covering `bounds` (left, top, right, bottom). Blocks until done, so call it off
    the UI thread. Returns a status message."""
    url = url or ""
    what = url or "a new window"
    browser = default_browser()
    if browser is None:
        webbrowser.open(url or "about:blank")
        return "Opened in your browser (window placement is only supported on macOS)."
    bundle_id, name = browser
    if bundle_id in CHROMIUM_IDS:
        script = _CHROMIUM_SCRIPT % bundle_id
    elif bundle_id in SAFARI_IDS:
        script = _SAFARI_SCRIPT % bundle_id
    else:
        script = _GENERIC_SCRIPT % (bundle_id, bundle_id, bundle_id)
    args = [url, *(str(int(v)) for v in bounds)]
    try:
        result = subprocess.run(["osascript", "-e", script, *args], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return f"{name} didn't respond. Check for a permission prompt."
    if result.returncode == 0:
        return f"Opened {what} in {name} on the left."
    err = result.stderr.strip()
    if "-1743" in err or "Not authorized" in err:
        return (f"macOS blocked controlling {name}. Allow it in System Settings → Privacy & Security → "
                "Automation, then try again.")
    fallback = ["open", "-b", bundle_id, *([url] if url else [])]
    if "-1719" in err or "assistive" in err.lower():
        subprocess.run(fallback)
        return (f"Opened in {name}, but placing its window needs Accessibility access "
                "(System Settings → Privacy & Security → Accessibility).")
    subprocess.run(fallback)
    return f"Opened in {name}, but couldn't place the window: {err or 'unknown error'}"
