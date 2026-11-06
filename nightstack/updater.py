"""In-app updates from GitHub Releases.

Each version is published as a GitHub Release (tag vX.Y.Z) with NightStack-macOS-Intel.zip
(the zipped .app).  The app compares the latest release with its own version, downloads the
zip, and a tiny helper script swaps the app bundle after NightStack quits, then relaunches it.
Files downloaded this way aren't quarantined, so macOS doesn't ask to approve the app again.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from typing import Callable, Optional

from . import __version__

REPO = "jphunter16/NightStack"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
ASSET_NAME = "NightStack-macOS-Intel.zip"


def _ssl_context():
    custom = os.environ.get("SSL_CERT_FILE")   # corporate proxies etc.
    if custom and os.path.exists(custom):
        return ssl.create_default_context(cafile=custom)
    try:
        import certifi  # bundled CA certificates (python.org builds have none by default)
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _open(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": f"NightStack/{__version__}",
                                               "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


def parse_version(v: str) -> tuple:
    nums = re.findall(r"\d+", v or "")
    return tuple(int(n) for n in nums[:3]) + (0,) * (3 - len(nums[:3]))


def is_newer(latest: str, current: str = __version__) -> bool:
    return parse_version(latest) > parse_version(current)


def check_latest(timeout: float = 8.0) -> Optional[dict]:
    """Return info about the latest release, or None if it can't be reached."""
    import urllib.error
    try:
        with _open(API_LATEST, timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:      # no release published yet
            return None
        raise
    tag = data.get("tag_name", "")
    asset = next((a for a in data.get("assets", []) if a.get("name") == ASSET_NAME), None)
    return {
        "version": tag.lstrip("v"),
        "notes": (data.get("body") or "").strip(),
        "url": asset.get("browser_download_url") if asset else None,
        "size": asset.get("size", 0) if asset else 0,
        "page": data.get("html_url", RELEASES_PAGE),
    }


def app_bundle() -> Optional[str]:
    """Path of the running NightStack.app, or None when running from source."""
    if not getattr(sys, "frozen", False):
        return None
    p = os.path.abspath(sys.executable)
    while p and p != "/" and not p.endswith(".app"):
        p = os.path.dirname(p)
    return p if p.endswith(".app") else None


def can_self_update() -> tuple:
    """(ok, reason) - whether this copy can replace itself."""
    b = app_bundle()
    if b is None:
        return False, "You're running NightStack from source. Use 'git pull' to update."
    if sys.platform != "darwin":
        return False, "Automatic updates are only available on macOS."
    if "/AppTranslocation/" in b:
        return False, ("macOS is running NightStack from a temporary location. Move NightStack.app into "
                       "your Applications folder, open it from there, then update.")
    if not os.access(os.path.dirname(b), os.W_OK):
        return False, f"NightStack can't write to {os.path.dirname(b)} (your user account needs permission)."
    return True, ""


def download(url: str, dest: str, progress: Callable[[float, str], None] = lambda p, m: None,
             total_hint: int = 0) -> str:
    with _open(url, 60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or total_hint or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                progress(min(0.95, got / total), f"Downloading update… {got / 1e6:.0f} of {total / 1e6:.0f} MB")
    return dest


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def unpack(zip_path: str, workdir: str) -> str:
    """Unzip the downloaded app; returns the path of the new NightStack.app."""
    out = os.path.join(workdir, "new")
    os.makedirs(out, exist_ok=True)
    if shutil.which("ditto"):            # macOS: keeps permissions, symlinks and code signature
        _run(["ditto", "-x", "-k", zip_path, out])
    else:
        _run(["unzip", "-q", zip_path, "-d", out])
    apps = [os.path.join(out, d) for d in os.listdir(out) if d.endswith(".app")]
    if not apps:
        raise RuntimeError("The download doesn't contain NightStack.app")
    return apps[0]


SWAP_SCRIPT = r'''#!/bin/bash
# Waits for NightStack to quit, swaps in the new app (rolling back on failure) and relaunches it.
PID="$1"; OLD="$2"; NEW="$3"; WORK="$4"
for i in $(seq 1 600); do kill -0 "$PID" 2>/dev/null || break; sleep 0.2; done
BACKUP="$OLD.previous"
rm -rf "$BACKUP"
if mv "$OLD" "$BACKUP"; then
  if (command -v ditto >/dev/null && ditto "$NEW" "$OLD") || cp -R "$NEW" "$OLD"; then
    rm -rf "$BACKUP"
  else
    rm -rf "$OLD"; mv "$BACKUP" "$OLD"
  fi
fi
command -v xattr >/dev/null && xattr -dr com.apple.quarantine "$OLD" 2>/dev/null
command -v open >/dev/null && open "$OLD"
rm -rf "$WORK"
'''


def install_and_relaunch(zip_path: str, bundle: Optional[str] = None, pid: Optional[int] = None,
                         workdir: Optional[str] = None) -> str:
    """Prepare the new app and start the swap helper. The caller must then quit the app."""
    bundle = bundle or app_bundle()
    if not bundle:
        raise RuntimeError("Not running as an installed app")
    workdir = workdir or os.path.dirname(zip_path)
    new_app = unpack(zip_path, workdir)
    script = os.path.join(workdir, "swap.sh")
    with open(script, "w") as f:
        f.write(SWAP_SCRIPT)
    os.chmod(script, 0o755)
    subprocess.Popen(["/bin/bash", script, str(pid or os.getpid()), bundle, new_app, workdir],
                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return new_app


def new_workdir() -> str:
    return tempfile.mkdtemp(prefix="nightstack-update-")
