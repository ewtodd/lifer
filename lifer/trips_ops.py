"""Trip-level operations shared by the CLI and the review app (task 24).

`migrate-trips` rekeys trip state after the task-24 rules change which
folder a trip runs from:

- a RAW-bucket trip `X--RAW` becomes `X`: its decisions, crop overrides
  and chosen frames are rekeyed (same frames, same encounter IDs);
- decisions made on derived-copy JPEGs are mapped to the raw frame they
  came from (by filename stem) and attached to the raw trip's encounter
  containing that frame.

Anything that cannot be mapped is listed, never discarded.

`delete` removes a trip's run dir, cache, DB rows, and chosen frames.
Everything here is scoped to LIFER_HOME and the decisions DB; nothing
under the photo archive is ever touched.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .bursts import FrameInfo, group_into_encounters
from .decisions import ensure_bird_box_column, ensure_frames_column
from .home import trip_id_for

if TYPE_CHECKING:
    from .runall import Row  # annotations only; runall imports this module

# Derived-copy decorations on a JPEG stem, innermost first: duplicate
# suffixes (-1), copy suffixes (_01/_02), lightroom zoom (_lzn), watermark
# (WM / _WM).
_RAW_DECOR_RE = re.compile(r"(-\d+|_0\d|_lzn|_WM|WM)$")


def raw_frame_for(name: str) -> str | None:
    """The raw frame a derived JPEG came from, by filename stem:
    ``_DSC0082_lzn.jpg -> _DSC0082.NEF``. Also handles ``_01``/``_02`` copy
    suffixes and ``-1`` duplicates. None when the stem is nothing but
    decorations."""
    s = Path(name).stem
    prev = None
    while prev != s:
        prev = s
        s = _RAW_DECOR_RE.sub("", s)
    return f"{s}.NEF" if s else None


def _file_count(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0


def _count_rows(conn: sqlite3.Connection | None, table: str, tid: str) -> int:
    if conn is None:
        return 0
    return conn.execute(f"SELECT COUNT(*) n FROM {table} WHERE folder=?",
                        (tid,)).fetchone()["n"]


def _chosen_frame_count(home: Path, tid: str) -> int:
    return sum(_file_count(home / sub / tid) for sub in ("develop", "edits"))


def trip_references(home: Path, conn: sqlite3.Connection | None,
                    tid: str) -> dict:
    """Everything that references a trip: run/cache files, DB rows, and
    chosen frames (develop/ + edits/)."""
    return {
        "trip": tid,
        "run_files": _file_count(home / "runs" / tid),
        "cache_files": _file_count(home / "cache" / tid),
        "decisions": _count_rows(conn, "decisions", tid),
        "crop_overrides": _count_rows(conn, "crop_boxes", tid),
        "chosen_frames": _chosen_frame_count(home, tid),
    }


def _assert_under(path: Path, home: Path) -> None:
    if not path.resolve().is_relative_to(home):
        raise SystemExit(f"refusing to delete {path}: outside LIFER_HOME")


def plan_delete(home: Path, conn: sqlite3.Connection | None,
                trip_ids: list[str]) -> dict:
    """What `delete` would remove, per trip. Nothing is touched."""
    return {"trips": [trip_references(Path(home), conn, t) for t in trip_ids]}


def apply_delete(home: Path, conn: sqlite3.Connection | None,
                 trip_ids: list[str]) -> dict:
    """Remove each trip's run dir, cache, DB rows, and chosen frames.
    Every removed path is asserted to be under LIFER_HOME."""
    home = Path(home).resolve()
    for tid in trip_ids:
        if "/" in tid or tid in (".", ".."):
            raise SystemExit(f"invalid trip ID: {tid!r}")
    report = []
    for tid in trip_ids:
        ref = trip_references(home, conn, tid)
        for t in (home / "runs" / tid, home / "cache" / tid,
                  home / "develop" / tid, home / "edits" / tid):
            if t.is_dir():
                _assert_under(t, home)
                shutil.rmtree(t)
        if conn is not None:
            conn.execute("DELETE FROM decisions WHERE folder=?", (tid,))
            conn.execute("DELETE FROM crop_boxes WHERE folder=?", (tid,))
            conn.commit()
        report.append(ref)
    return {"trips": report}


# --- migrate-trips ---------------------------------------------------------


def _candidate_keys(run_dir: Path) -> set[tuple[str, str]]:
    """(enc_id, best_frame) pairs a run currently produces."""
    keys: set[tuple[str, str]] = set()
    c = run_dir / "candidates.csv"
    if c.is_file():
        import csv
        with open(c, newline="") as f:
            for row in csv.DictReader(f):
                keys.add((row["enc_id"], row["best_frame"]))
    n = run_dir / "nodet.json"
    if n.is_file():
        try:
            for e in json.loads(n.read_text()):
                keys.add((e["enc_id"], e["best_frame"]))
        except (OSError, ValueError):
            pass
    return keys


def encounter_frame_index(home: Path, tid: str,
                          frames: set[str] | None = None) -> dict[str, tuple[str, str]]:
    """frame name -> (enc_id, the encounter's best frame) for a trip.

    Recomputed from the per-frame cache the pipeline wrote, with the same
    grouping the pipeline used (5 s gaps, sharpest bird crop wins), so the
    encounter numbering matches the run. ``frames`` restricts to the
    trip's current source folder (the cache may hold stale entries from
    an earlier source)."""
    cache = Path(home) / "cache" / tid
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
    idx: dict[str, tuple[str, str]] = {}
    for i, enc in enumerate(group_into_encounters(infos), 1):
        enc_id = f"enc_{i:04d}"
        best = enc.best_frame.name
        for f in enc.frames:
            idx[f.name] = (enc_id, best)
    return idx


def _folder_photos(folder: Path) -> set[str]:
    from .runall import PHOTO_EXTS  # deferred: runall imports this module
    return {p.name for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in PHOTO_EXTS}


def _dec_ref(d: sqlite3.Row, why: str) -> dict:
    return {"trip": d["folder"], "enc_id": d["enc_id"], "frame": d["best_frame"],
            "action": d["action"], "species": d["species"], "reason": why}


def plan_migrate_trips(home: Path, conn: sqlite3.Connection | None,
                       root: Path, rows: list[Row]) -> dict:
    """What `migrate-trips` would do, given the current run-all plan.

    Nothing is touched. ``rows`` is the plan for ``root`` (see runall)."""
    home = Path(home)
    root = Path(root).expanduser().resolve()
    from .runall import _tree_of  # deferred: runall imports this module
    plan: dict = {"rekey": [], "map": [], "unmapped": [], "conflicts": [],
                  "mismatched": []}
    if conn is None:
        return plan
    conn.row_factory = sqlite3.Row
    ensure_frames_column(conn)

    # Case A: raw-bucket trips X--RAW becoming X.
    for r in rows:
        if r.status != "run":
            continue
        old = trip_id_for(r.folder, root)
        if old == r.trip_id or not (home / "runs" / old).is_dir():
            continue
        old_decs = conn.execute("SELECT * FROM decisions WHERE folder=?",
                                (old,)).fetchall()
        new_keys = _candidate_keys(home / "runs" / r.trip_id)
        new_dec_keys = {(d["enc_id"], d["best_frame"]) for d in conn.execute(
            "SELECT enc_id, best_frame FROM decisions WHERE folder=?",
            (r.trip_id,))}
        plan["rekey"].append({
            "from": old, "to": r.trip_id,
            "decisions": len(old_decs),
            "crop_overrides": _count_rows(conn, "crop_boxes", old),
            "chosen_frames": _chosen_frame_count(home, old),
            "conflicts": [f"{d['enc_id']} {d['best_frame']}" for d in old_decs
                          if (d["enc_id"], d["best_frame"]) in new_dec_keys],
            "mismatched": [f"{d['enc_id']} {d['best_frame']}" for d in old_decs
                           if (d["enc_id"], d["best_frame"]) not in new_keys],
        })

    # Case B: decisions on derived-copy JPEGs (stale trips, or trips whose
    # source folder changed) -> the raw frame they came from, attached to
    # the raw trip's encounter containing that frame.
    plan_ids = {r.trip_id for r in rows if r.status == "run"}
    runs = home / "runs"
    affected: list[tuple[str, Path | None]] = []  # (trip id, new folder)
    if runs.is_dir():
        for d in sorted(p for p in runs.iterdir() if p.is_dir()):
            if d.name in plan_ids:
                # A trip ID may have several plan rows (e.g. the top folder,
                # now a derived copy, and the RAW bucket that runs under it);
                # the current source is the one that runs.
                row = next((r for r in rows
                            if r.trip_id == d.name and r.status == "run"),
                           None)
                if row is None:
                    continue
                meta = d / "meta.json"
                old_folder = None
                if meta.is_file():
                    try:
                        old_folder = Path(json.loads(meta.read_text())["folder"])
                    except (OSError, KeyError, ValueError):
                        pass
                if old_folder is None or old_folder == row.folder:
                    continue  # same source: nothing to migrate
                affected.append((d.name, row.folder))
            else:
                affected.append((d.name, None))

    # Raw trips per top-level tree, with their encounter frame indices.
    # A tree can have several (one per day folder), so this is a list.
    raw_trips: list[tuple[str, str, dict[str, tuple[str, str]]]] = []
    for r in rows:
        if r.status != "run":
            continue
        photos = _folder_photos(r.folder)
        if not any(n.lower().endswith(".nef") for n in photos):
            continue
        tree = _tree_of(r.folder, root).name
        raw_trips.append((tree, r.trip_id,
                          encounter_frame_index(home, r.trip_id, photos)))

    planned_keys: set[tuple[str, str, str]] = set()
    rekey_sources = {e["from"] for e in plan["rekey"]}
    for tid, new_folder in affected:
        if tid in rekey_sources:
            continue  # case A (rekey) owns that trip's decisions
        decs = conn.execute(
            "SELECT * FROM decisions WHERE folder=? ORDER BY enc_id",
            (tid,)).fetchall()
        if not decs:
            continue
        keep = _folder_photos(new_folder) if new_folder and new_folder.is_dir() else None
        tree = tid.split("--")[0]
        for d in decs:
            if keep is not None and d["best_frame"] in keep:
                continue  # the frame still exists in the new source
            raw = raw_frame_for(d["best_frame"])
            if raw is None:
                plan["unmapped"].append(_dec_ref(d, "no NEF stem"))
                continue
            target, hit = None, None
            for tree2, tid2, idx in raw_trips:
                if tree2 != tree:
                    continue
                h = idx.get(raw)
                if h is not None:
                    target, hit = tid2, h
                    break
            if target is None or hit is None:
                plan["unmapped"].append(
                    _dec_ref(d, f"no encounter in {tree} contains {raw}"))
                continue
            enc_id, enc_best = hit
            key = (target, enc_id, enc_best)
            in_db = conn.execute(
                "SELECT 1 FROM decisions WHERE folder=? AND enc_id=? "
                "AND best_frame=?", key).fetchone()
            if in_db or key in planned_keys:
                plan["conflicts"].append(
                    _dec_ref(d, f"a decision already sits at "
                                f"{target}/{enc_id}/{enc_best}"))
                continue
            planned_keys.add(key)
            plan["map"].append({
                "from": tid, "enc_id": d["enc_id"], "frame": d["best_frame"],
                "to_trip": target, "to_enc_id": enc_id, "to_frame": enc_best,
                "action": d["action"], "species": d["species"],
                "scientific": d["scientific"], "note": d["note"],
                "frames": d["frames"],
                "updated_at": d["updated_at"],
            })
    return plan


def migration_summary(res: dict) -> str:
    """One line per done/needed migration, for run-all's output."""
    parts = []
    for entry in res["rekey"]:
        parts.append(f"rekeyed {entry['from']} -> {entry['to']} "
                     f"({entry['decisions']} decisions, "
                     f"{entry['crop_overrides']} crop fixes, "
                     f"{entry['chosen_frames']} chosen frames)")
    for entry in res["map"]:
        parts.append(f"mapped {entry['from']}/{entry['enc_id']} "
                     f"{entry['frame']} -> {entry['to_trip']}/"
                     f"{entry['to_enc_id']}")
    if res["unmapped"]:
        parts.append(f"{len(res['unmapped'])} unmapped (left in place): "
                     + ", ".join(f"{u['trip']}/{u['enc_id']} {u['frame']}"
                                 for u in res["unmapped"]))
    if res["conflicts"]:
        parts.append(f"{len(res['conflicts'])} conflicts (left in place)")
    for entry in res["rekey"]:
        for m in entry["mismatched"]:
            parts.append(f"{entry['to']}: {m} no longer matches its "
                         f"candidates")
    return "; ".join(parts) if parts else "nothing to migrate"


def apply_migrate_trips(home: Path, conn: sqlite3.Connection | None,
                        root: Path, rows: list[Row]) -> dict:
    """Apply the migrations from :func:`plan_migrate_trips`.

    Returns the plan, with ``applied`` marked on what was done; conflicts
    and unmapped decisions are left in place."""
    home = Path(home)
    plan = plan_migrate_trips(home, conn, root, rows)
    if conn is None:
        return plan
    conn.row_factory = sqlite3.Row
    ensure_bird_box_column(conn)  # the rekey copies decisions.bird_box

    # Case A: rekey raw-bucket trips into their parent.
    for entry in plan["rekey"]:
        old, new = entry["from"], entry["to"]
        conflict_keys = set(entry["conflicts"])
        for d in conn.execute("SELECT * FROM decisions WHERE folder=?",
                              (old,)).fetchall():
            label = f"{d['enc_id']} {d['best_frame']}"
            if label in conflict_keys:
                continue
            conn.execute(
                "INSERT INTO decisions (folder, enc_id, best_frame, action,"
                " species, scientific, note, frames, updated_at, bird_box)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (new, d["enc_id"], d["best_frame"], d["action"], d["species"],
                 d["scientific"], d["note"], d["frames"], d["updated_at"], d["bird_box"]))
            conn.execute(
                "DELETE FROM decisions WHERE folder=? AND enc_id=? "
                "AND best_frame=?", (old, d["enc_id"], d["best_frame"]))
        new_crops = {r["frame"] for r in conn.execute(
            "SELECT frame FROM crop_boxes WHERE folder=?", (new,))}
        for r in conn.execute("SELECT frame FROM crop_boxes WHERE folder=?",
                              (old,)).fetchall():
            if r["frame"] not in new_crops:
                conn.execute(
                    "UPDATE crop_boxes SET folder=? WHERE folder=? AND frame=?",
                    (new, old, r["frame"]))
        for sub in ("develop", "edits"):
            s, t = home / sub / old, home / sub / new
            if s.is_dir():
                if t.is_dir():
                    entry["chosen_frames_skipped"] = True
                else:
                    t.parent.mkdir(parents=True, exist_ok=True)
                    s.rename(t)
        conn.commit()
        entry["applied"] = True

    # Case B: move mapped decisions onto the raw trip's encounters.
    for m in plan["map"]:
        conn.execute(
            "INSERT INTO decisions (folder, enc_id, best_frame, action,"
            " species, scientific, note, frames, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (m["to_trip"], m["to_enc_id"], m["to_frame"], m["action"],
             m["species"], m["scientific"], m["note"], m.get("frames", ""),
             m["updated_at"]))
        conn.execute(
            "DELETE FROM decisions WHERE folder=? AND enc_id=? "
            "AND best_frame=?", (m["from"], m["enc_id"], m["frame"]))
        m["applied"] = True
    conn.commit()
    return plan
