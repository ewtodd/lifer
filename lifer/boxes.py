"""Crop box resolution: manual overrides beat the detector.

One shared function, ``get_box(folder, frame)``, is the single source of truth
for the box used to crop a frame. The review app, the pipeline, and ``lifer
publish`` all call it, so a manual crop fix in the review app flows everywhere
a crop is produced. Overrides live in the SQLite DB keyed by (folder, frame
filename) so re-running the pipeline keeps them.

Coordinates are in full-preview pixel space (the 6000x4000 embedded JPEG), the
same space the detector writes, so an override needs no conversion to crop.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .home import lifer_home


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 0.0
    source: str = "detector"  # 'detector' | 'override'


def default_db_path() -> Path:
    """The default decisions/override DB (the serve app's default --db)."""
    return lifer_home() / "data" / "lifer.sqlite"


def open_db(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or default_db_path()))
    conn.row_factory = sqlite3.Row
    # the pipeline reads overrides from here; make sure the table exists on a
    # fresh LIFER_HOME (the serve app creates the full schema separately).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS crop_boxes ("
        " folder TEXT NOT NULL, frame TEXT NOT NULL, x1 REAL NOT NULL, "
        " y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL, "
        " reid_old TEXT NOT NULL DEFAULT '', reid_new TEXT NOT NULL DEFAULT '', "
        " snapshot TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL, "
        " PRIMARY KEY (folder, frame))")
    # Owner-selected representative frames for an encounter (task 12), keyed by
    # (folder, the encounter's frame set) so a re-run keeps the choice. The
    # pipeline only reads crop_boxes, so this table is created lazily here and
    # by the serve app's full SCHEMA.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS frame_choice ("
        " folder TEXT NOT NULL, frames TEXT NOT NULL, "
        " chosen TEXT NOT NULL, updated_at TEXT NOT NULL, "
        " PRIMARY KEY (folder, frames))")
    # Frames the owner dropped from their encounter in the review app; the
    # pipeline leaves them out of the encounter on a re-run.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS removed_frames ("
        " folder TEXT NOT NULL, frame TEXT NOT NULL, updated_at TEXT NOT NULL, "
        " PRIMARY KEY (folder, frame))")
    conn.commit()
    return conn


def to_list(box: Box) -> list[float]:
    return [box.x1, box.y1, box.x2, box.y2]


def _conf(det) -> float:
    if hasattr(det, "conf"):
        return float(getattr(det, "conf") or 0.0)
    if isinstance(det, (list, tuple)) and len(det) > 4:
        return float(det[4])
    return 0.0


def _box_from(det, source: str = "detector") -> Box:
    """Normalise a detection (Detection, Box, or [x1,y1,x2,y2,conf]) to a Box."""
    if hasattr(det, "x1"):
        return Box(float(det.x1), float(det.y1), float(det.x2), float(det.y2),
                   _conf(det), source)
    return Box(float(det[0]), float(det[1]), float(det[2]), float(det[3]),
               _conf(det), source)


def get_override(conn: sqlite3.Connection, folder: str, frame: str) -> Box | None:
    row = conn.execute(
        "SELECT x1, y1, x2, y2 FROM crop_boxes WHERE folder=? AND frame=?",
        (folder, frame),
    ).fetchone()
    if row is None:
        return None
    return Box(row["x1"], row["y1"], row["x2"], row["y2"], 0.0, "override")


def get_detector_box(folder: str, frame: str) -> Box | None:
    """Best detection for a frame from its cached dets (full-pixel coords)."""
    p = lifer_home() / "cache" / folder / f"{Path(frame).stem}.json"
    if not p.is_file():
        return None
    try:
        dets = (json.loads(p.read_text()) or {}).get("dets") or []
    except (json.JSONDecodeError, OSError):
        return None
    if not dets:
        return None
    return _box_from(max(dets, key=_conf))


def get_box(folder: str, frame: str,
            conn: sqlite3.Connection | None = None, dets=None) -> Box | None:
    """The box used for a frame: override if present, else the detector box.

    `dets` may be supplied when the caller already has the frame's detections
    (e.g. the pipeline) to avoid re-reading the cache; it is a list of objects
    with .x1/.y1/.x2/.y2/.conf or [x1,y1,x2,y2,conf] tuples. When `conn` is
    None a short-lived connection to the default DB is opened.
    """
    if conn is None:
        conn = open_db()
        own = True
    else:
        own = False
    try:
        o = get_override(conn, folder, frame)
        if o is not None:
            return o
        if dets is not None:
            if not dets:
                return None
            return _box_from(max(dets, key=_conf))
        return get_detector_box(folder, frame)
    finally:
        if own:
            conn.close()


def set_override(conn: sqlite3.Connection, folder: str, frame: str, box: Box,
                 reid_old: str = "", reid_new: str = "",
                 snapshot: str = "") -> None:
    """Upsert the override. `snapshot` (the original candidate fields) is only
    written on first insert so Undo restores the pipeline result even after
    several re-fixes; reid_old/reid_new always track the latest re-identify."""
    now = datetime.now().isoformat(timespec="seconds")
    row = conn.execute(
        "SELECT snapshot FROM crop_boxes WHERE folder=? AND frame=?",
        (folder, frame)).fetchone()
    if row is not None and row["snapshot"]:
        snapshot = row["snapshot"]
    conn.execute(
        "INSERT INTO crop_boxes (folder, frame, x1, y1, x2, y2, reid_old, "
        "reid_new, snapshot, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(folder, frame) DO UPDATE SET x1=excluded.x1, "
        "y1=excluded.y1, x2=excluded.x2, y2=excluded.y2, "
        "reid_old=excluded.reid_old, reid_new=excluded.reid_new, "
        "updated_at=excluded.updated_at",
        (folder, frame, box.x1, box.y1, box.x2, box.y2,
         reid_old, reid_new, snapshot, now),
    )
    conn.commit()


def clear_override(conn: sqlite3.Connection, folder: str, frame: str) -> None:
    conn.execute("DELETE FROM crop_boxes WHERE folder=? AND frame=?",
                 (folder, frame))
    conn.commit()


def get_snapshot(conn: sqlite3.Connection, folder: str,
                 frame: str) -> dict[str, str] | None:
    row = conn.execute(
        "SELECT snapshot FROM crop_boxes WHERE folder=? AND frame=?",
        (folder, frame)).fetchone()
    if row is None or not row["snapshot"]:
        return None
    try:
        return json.loads(row["snapshot"])
    except json.JSONDecodeError:
        return None


def get_reid(conn: sqlite3.Connection, folder: str,
             frame: str) -> dict[str, str] | None:
    row = conn.execute(
        "SELECT reid_old, reid_new FROM crop_boxes WHERE folder=? AND frame=?",
        (folder, frame)).fetchone()
    if row is None or not (row["reid_old"] or row["reid_new"]):
        return None
    return {"old": row["reid_old"], "new": row["reid_new"]}


# ---------------------------------------------------------------------------
# Encounter representative frame (task 12)
# ---------------------------------------------------------------------------
#
# An encounter is defined by its set of frames. The owner can pick any one of
# them as the photo that represents the encounter; `representative_frame` is
# the single shared look-up every consumer (review card, sighting/lifelist
# thumb, `lifer publish`) goes through, next to `get_box`. The choice is keyed
# by (folder, the encounter's frame set) so re-running the pipeline keeps it,
# just like the crop-box overrides are keyed by (folder, frame).


def frames_key(frames: list[str]) -> str:
    """Canonical DB key for an encounter's frame set (order-independent)."""
    return json.dumps(sorted(frames))


def get_frame_choice(conn: sqlite3.Connection, folder: str,
                     frames: list[str]) -> str | None:
    """The owner-chosen representative frame for (folder, frames), or None."""
    if not frames:
        return None
    row = conn.execute(
        "SELECT chosen FROM frame_choice WHERE folder=? AND frames=?",
        (folder, frames_key(frames))).fetchone()
    return row["chosen"] if row else None


def set_frame_choice(conn: sqlite3.Connection, folder: str,
                     frames: list[str], chosen: str) -> None:
    """Upsert the representative-frame choice for an encounter."""
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO frame_choice (folder, frames, chosen, updated_at) "
        "VALUES (?,?,?,?) ON CONFLICT(folder, frames) DO UPDATE SET "
        "chosen=excluded.chosen, updated_at=excluded.updated_at",
        (folder, frames_key(frames), chosen, now))
    conn.commit()


def clear_frame_choice(conn: sqlite3.Connection, folder: str,
                       frames: list[str]) -> None:
    conn.execute("DELETE FROM frame_choice WHERE folder=? AND frames=?",
                 (folder, frames_key(frames)))
    conn.commit()


def add_removed_frame(conn: sqlite3.Connection, folder: str, frame: str) -> None:
    """Record that the owner removed ``frame`` from its encounter."""
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT OR REPLACE INTO removed_frames (folder, frame, updated_at) "
        "VALUES (?,?,?)", (folder, frame, now))
    conn.commit()


def removed_frames(conn: sqlite3.Connection, folder: str) -> set[str]:
    """Frames the owner removed from their encounters in ``folder``."""
    return {r["frame"] for r in conn.execute(
        "SELECT frame FROM removed_frames WHERE folder=?", (folder,))}


def representative_frame(conn: sqlite3.Connection, folder: str,
                         frames: list[str], best_frame: str) -> str:
    """The frame whose crop represents an encounter everywhere a 'best frame'
    is chosen: the owner's pick (via ``set_frame_choice``) when it is still one
    of the encounter's frames, otherwise the pipeline's best frame."""
    if frames:
        chosen = get_frame_choice(conn, folder, frames)
        if chosen and chosen in frames:
            return chosen
    return best_frame
