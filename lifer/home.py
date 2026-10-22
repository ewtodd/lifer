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


def _trip_root(folder: Path) -> Path:
    """Default root for trip-ID derivation: the photo archive root when the
    folder sits under it, else the folder's own parent. Using the archive root
    keeps a nested folder's trip ID the same whether it was reached by
    run-all (root=~/BIRDING) or by `run`/`develop`/`handoff` with no --root."""
    folder = folder.resolve()
    birding = Path.home() / "BIRDING"
    birding = birding.resolve()
    try:
        folder.relative_to(birding)
        return birding
    except ValueError:
        return folder.parent


def trip_id_for(folder: Path, root: Path | str | None = None) -> str:
    """Trip ID for a photo folder: the folder name when it sits directly
    under its root (so existing runs/decisions keep working), else the
    root-relative path with the separators replaced by "--" (e.g. a folder
    at root/a/b is trip "a--b")."""
    folder = folder.resolve()
    root = Path(root).expanduser().resolve() if root else _trip_root(folder)
    try:
        rel = folder.relative_to(root)
    except ValueError:
        raise SystemExit(f"{folder} is not under {root}")
    return "--".join(rel.parts) if rel.parts else folder.name


# Subfolders that hold an outing's raws and take the parent's trip ID:
# their photos are the outing, not a nested trip (task 24).
RAW_BUCKET_NAMES = ("raw", "nef")


def is_raw_bucket(folder: Path) -> bool:
    """True for a folder named like a raw bucket (RAW, raw, NEF;
    case-insensitive) that actually holds .NEF files."""
    folder = Path(folder)
    return (folder.name.lower() in RAW_BUCKET_NAMES
            and any(p.is_file() and p.suffix.lower() == ".nef"
                    for p in folder.iterdir()))


def effective_trip_id(folder: Path, root: Path | str | None = None) -> str:
    """trip_id_for, except a raw bucket takes its parent's trip ID:
    `root/2021-07-28-DAVIE/RAW` runs as trip `2021-07-28-DAVIE`, and region
    matching uses that ID. A top-level folder named RAW (no parent outing)
    keeps its own ID."""
    folder = Path(folder).expanduser().resolve()
    if root is not None:
        root_r = Path(root).expanduser().resolve()
        if is_raw_bucket(folder) and folder.parent != root_r:
            return trip_id_for(folder.parent, root)
    return trip_id_for(folder, root)


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
