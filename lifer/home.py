"""State location for the lifer CLI.

The code is installed from the Nix store (read-only), so all mutable
state lives in one user directory: LIFER_HOME (env var or --home),
defaulting to $XDG_DATA_HOME/lifer (~/.local/share/lifer). It holds
runs/ (pipeline outputs), cache/ (per-image detections + BioCLIP
probabilities, iNat API data) and data/ (lifelist.csv, the SQLite DB).
Model weights live in the user *cache* dir ($XDG_CACHE_HOME/lifer,
default ~/.cache/lifer) because they can be re-downloaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def lifer_home() -> Path:
    """State dir: $LIFER_HOME, else $XDG_DATA_HOME/lifer, else ~/.local/share/lifer."""
    h = os.environ.get("LIFER_HOME")
    if h:
        return Path(h).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "lifer"


def weights_dir() -> Path:
    """Model weights dir: $XDG_CACHE_HOME/lifer, else ~/.cache/lifer."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "lifer"


def find_regions_toml() -> Path:
    """regions.toml: an optional user override in LIFER_HOME wins, else the
    copy shipped with the package (repo root in dev)."""
    p = lifer_home() / "regions.toml"
    if p.is_file():
        return p
    return _PKG_DIR / "regions.toml"


def trip_id_for(folder: Path, root: Path | str | None = None) -> str:
    """Trip ID for a photo folder: the folder name when it sits directly
    under its root (so existing runs/decisions keep working), else the
    root-relative path with the separators replaced by "--" (e.g. a folder
    at root/a/b is trip "a--b")."""
    folder = folder.resolve()
    root = Path(root).expanduser().resolve() if root else folder.parent
    try:
        rel = folder.relative_to(root)
    except ValueError:
        raise SystemExit(f"{folder} is not under {root}")
    return "--".join(rel.parts) if rel.parts else folder.name


def source_folder_for_trip(trip_id: str) -> Path:
    """The source photo folder a trip was run from. `lifer run` records it in
    runs/<trip>/meta.json (needed for nested folders, which are not at
    ~/BIRDING/<trip>); runs created before meta.json existed fall back to
    the old ~/BIRDING/<trip> assumption."""
    meta = lifer_home() / "runs" / trip_id / "meta.json"
    if meta.is_file():
        try:
            p = Path(json.loads(meta.read_text())["folder"])
            if p.is_dir():
                return p
        except (OSError, ValueError, KeyError):
            pass  # corrupt/unreadable meta: fall back
    return Path.home() / "BIRDING" / trip_id
