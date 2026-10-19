"""Decisions attach to encounters by frame overlap, not by encounter ID.

A decision's (enc_id, best_frame) pair is only valid for the run that
produced it: a re-detection can renumber encounters and change the best
frame (sharpness is not bit-stable on CUDA, see bursts._best_frame_key).
So each decision stores the frame list of the encounter it was made on,
and is resolved to a current encounter by frame overlap:

1. the encounter containing the decision's best frame, else
2. the encounter with the most shared frames.

enc_id and best_frame are hints only. Ambiguous matches (a tie, or the
best frame in two encounters) and decisions with no overlap are reported,
never guessed. `migrate-decisions` uses the same logic to backfill frame
lists for legacy decisions and re-attach orphaned ones.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

from .bursts import FrameInfo, group_into_encounters


@dataclass
class EncRef:
    """One encounter of a run: its ID, best frame, and full frame list."""
    enc_id: str
    best_frame: str
    frames: list[str] = field(default_factory=list)


def parse_frames(s: str | None) -> list[str]:
    """Parse the decisions.frames column (a JSON list; '' for legacy rows)."""
    if not s:
        return []
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return []


def frames_json(frames: list[str]) -> str:
    return json.dumps(frames)


def ensure_frames_column(conn: sqlite3.Connection) -> None:
    """Legacy DBs predate the decisions.frames column; add it in place."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(decisions)")]
    if "frames" not in cols:
        conn.execute("ALTER TABLE decisions ADD COLUMN frames TEXT NOT NULL DEFAULT ''")
        conn.commit()


def resolve_decision(best_frame: str, frames: list[str],
                     encs: list[EncRef]) -> tuple[EncRef | None, str]:
    """Resolve a decision to one of `encs` by frame overlap.

    Returns (enc, status) where status is 'best-frame' (the unique
    encounter containing the decision's best frame), 'overlap' (a unique
    most-shared-frames winner), 'ambiguous' (a tie, or the best frame in
    two encounters), or 'none' (no overlap at all).
    """
    if best_frame:
        hits = [e for e in encs if best_frame in e.frames]
        if len(hits) == 1:
            return hits[0], "best-frame"
        if len(hits) > 1:
            return None, "ambiguous"
    if frames:
        fset = set(frames)
        scored = [(len(fset & set(e.frames)), e) for e in encs]
        top = max(n for n, _e in scored)
        if top > 0:
            ties = [e for n, e in scored if n == top]
            if len(ties) == 1:
                return ties[0], "overlap"
            return None, "ambiguous"
    return None, "none"


# ---------------------------------------------------------------------------
# building the current encounter set for a folder
# ---------------------------------------------------------------------------

def encounters_from_cache(home: Path, folder: str,
                          frames: set[str] | None = None) -> list[EncRef]:
    """Recompute a trip's encounters from the per-frame cache the pipeline
    wrote (timestamps + sharpness + confs, same gap grouping as the
    pipeline). This is how runs whose candidates.csv predates the frames
    column can still be resolved: the cache keeps one entry per frame, so
    the encounter frame lists are recoverable.
    """
    cache = Path(home) / "cache" / folder
    infos: list[FrameInfo] = []
    if cache.is_dir():
        for p in sorted(cache.glob("*.json")):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            name = d.get("name")
            if not name or (frames is not None and name not in frames):
                continue
            dets = d.get("dets") or []
            try:
                ts = datetime.fromisoformat(d["timestamp"])
            except (KeyError, ValueError):
                continue
            infos.append(FrameInfo(
                name=name, timestamp=ts, n_dets=len(dets),
                best_conf=max((x[4] for x in dets), default=0.0),
                sharpness=d.get("sharpness", -1.0)))
    out: list[EncRef] = []
    for i, enc in enumerate(group_into_encounters(infos), 1):
        out.append(EncRef(f"enc_{i:04d}", enc.best_frame.name,
                          [f.name for f in enc.frames]))
    return out


def encounters_from_outputs(home: Path, folder: str) -> list[EncRef]:
    """The run's encounters from runs/<folder>/ (candidates.csv +
    nodet.json). Frame lists are empty for outputs that predate the
    frames column; `current_encounters` fills those from the cache."""
    import csv

    encs: list[EncRef] = []
    run = Path(home) / "runs" / folder
    c = run / "candidates.csv"
    if c.is_file():
        with open(c, newline="") as f:
            for row in csv.DictReader(f):
                frames = [x for x in (row.get("frames") or "").split(",") if x]
                encs.append(EncRef(row["enc_id"], row["best_frame"], frames))
    n = run / "nodet.json"
    if n.is_file():
        try:
            for e in json.loads(n.read_text()):
                encs.append(EncRef(e["enc_id"], e["best_frame"],
                                   [str(x) for x in (e.get("frames") or [])]))
        except (OSError, ValueError):
            pass
    return encs


def current_encounters(home: Path, folder: str) -> list[EncRef]:
    """The current encounter set for a folder: run outputs first, with
    missing frame lists (old-format outputs) filled in from the cache.
    Folders whose run outputs are gone fall back to the cache alone."""
    home = Path(home)
    encs = encounters_from_outputs(home, folder)
    if not encs:
        return encounters_from_cache(home, folder)
    if any(not e.frames for e in encs):
        by_id = {e.enc_id: e for e in encounters_from_cache(home, folder)}
        for e in encs:
            c = by_id.get(e.enc_id)
            if c is not None:
                e.frames = list(c.frames)
    return encs


# ---------------------------------------------------------------------------
# migrate-decisions
# ---------------------------------------------------------------------------

def plan_migrate_decisions(home: Path, conn: sqlite3.Connection) -> dict[str, object]:
    """What `migrate-decisions` would do; nothing is touched.

    Per folder with decisions, every decision is resolved to the current
    encounter set:
    - already attached (same key, up-to-date frames) -> counted 'ok'
    - legacy row without a frame list -> backfilled ('backfilled')
    - key drifted (enc_id / best_frame changed) -> re-attached ('reattached')
    - no overlap, or a tie -> listed in 'ambiguous', left in place
    """
    home = Path(home)
    conn.row_factory = sqlite3.Row
    ensure_frames_column(conn)

    entries: list[dict[str, object]] = []
    ambiguous: list[dict[str, object]] = []
    n_ok = 0
    n_decisions = 0
    for (folder,) in conn.execute("SELECT DISTINCT folder FROM decisions ORDER BY folder"):
        encs = current_encounters(home, folder)
        for d in conn.execute(
                "SELECT folder, enc_id, best_frame, action, species, frames "
                "FROM decisions WHERE folder=? ORDER BY enc_id, best_frame",
                (folder,)):
            n_decisions += 1
            frames = parse_frames(d["frames"])
            enc, status = resolve_decision(d["best_frame"], frames, encs)
            if enc is None:
                reason = ("no run outputs or cache for this trip"
                          if not encs else
                          f"no overlap with the current run ({status})")
                ambiguous.append({
                    "folder": folder, "enc_id": d["enc_id"],
                    "frame": d["best_frame"], "action": d["action"],
                    "species": d["species"], "reason": reason,
                })
                continue
            same_key = (enc.enc_id == d["enc_id"]
                        and enc.best_frame == d["best_frame"])
            # the matched encounter's frame list; if it has none (run predates
            # the frames column and no cache), at least the best frame is known
            new_frames = list(enc.frames) or [enc.best_frame]
            if same_key and frames == new_frames:
                n_ok += 1
                continue
            entries.append({
                "folder": folder, "enc_id": d["enc_id"], "frame": d["best_frame"],
                "action": d["action"], "species": d["species"],
                "new_enc_id": enc.enc_id, "new_frame": enc.best_frame,
                "new_frames": new_frames,
                "changed": not same_key,
                "had_frames": bool(frames),
                "conflict": False,
            })

    return {
        "home": str(home),
        "n_decisions": n_decisions,
        "ok": n_ok,
        "backfilled": sum(1 for e in entries if not e["had_frames"]),
        "reattached": sum(1 for e in entries if e["changed"]),
        "entries": entries,
        "ambiguous": ambiguous,
    }


def apply_migrate_decisions(home: Path, conn: sqlite3.Connection) -> dict[str, object]:
    """Plan, then write the backfills and re-attachments. A re-attach whose
    target key is already occupied by a different decision is skipped and
    flagged (conflict) instead of dropping data."""
    plan = plan_migrate_decisions(home, conn)
    entries = cast("list[dict[str, object]]", plan["entries"])
    for e in entries:
        if e["changed"]:
            occupied = conn.execute(
                "SELECT 1 FROM decisions WHERE folder=? AND enc_id=? "
                "AND best_frame=?",
                (e["folder"], e["new_enc_id"], e["new_frame"])).fetchone()
            if occupied:
                e["conflict"] = True
                continue
        conn.execute(
            "UPDATE decisions SET enc_id=?, best_frame=?, frames=? "
            "WHERE folder=? AND enc_id=? AND best_frame=?",
            (e["new_enc_id"], e["new_frame"],
             frames_json(cast("list[str]", e["new_frames"])),
             e["folder"], e["enc_id"], e["frame"]))
    conn.commit()
    plan["applied"] = sum(1 for e in entries if not e["conflict"])
    plan["conflicts"] = [e for e in entries if e["conflict"]]
    return plan
