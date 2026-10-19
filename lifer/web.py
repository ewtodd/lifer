"""Local review web app: trips, trip review (species groups + decisions), lifelist.

Flask + Jinja + vanilla JS, no frontend build step, no CDN (works offline).
State lives in SQLite (data/lifer.sqlite by default): the source of truth for
review decisions and the imported lifelist. The app reads every runs/<folder>/
the pipeline has produced, so new trips appear automatically. Decisions are
keyed by (folder, enc_id, best_frame) so a re-run of a folder does not lose
them. Nothing under ~/BIRDING is ever touched.
"""

from __future__ import annotations

import calendar
import csv
import fnmatch
import hashlib
import json
import os
import sqlite3
import tomllib
from datetime import date, datetime
from pathlib import Path

from flask import (Flask, abort, g, jsonify, render_template, request,
                   send_file)
from PIL import Image
from rapidfuzz import fuzz, process
from werkzeug.middleware.proxy_fix import ProxyFix

from .boxes import (Box, clear_override, get_box, get_override, get_reid,
                    get_snapshot, set_override, to_list)
from .extract import load_photo
from .home import lifer_home, find_regions_toml, source_folder_for_trip
from .outputs import CSS, save_crop, save_frame_preview
from .region import region_display_names

# state dir without importing .pipeline (that pulls in torch; serving needs no GPU)
RUNS = lifer_home() / "runs"
INAT_CACHE = lifer_home() / "cache" / "inat"
LIFELIST_CSV = lifer_home() / "data" / "lifelist.csv"
# Source photos live on the read-only BIRDING share (wherever the run's
# meta.json points); the crop editor re-reads them to pull the full-size
# image (read-only). Nothing is ever written there.

SECTIONS: list[tuple[str, str]] = [
    ("rare", "Rare here"),
    ("lifer", "Possible lifers"),
    ("lookalike", "Look-alikes"),
    ("other", "Everything else"),
]

DECISION_ACTIONS = {"confirm", "change", "not_bird", "unsure"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  folder TEXT NOT NULL,
  enc_id TEXT NOT NULL,
  best_frame TEXT NOT NULL,
  action TEXT NOT NULL,            -- confirm | change | not_bird | unsure
  species TEXT NOT NULL DEFAULT '',
  scientific TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL,
  PRIMARY KEY (folder, enc_id, best_frame)
);
CREATE TABLE IF NOT EXISTS lifelist (
  taxon_id INTEGER PRIMARY KEY,
  common TEXT NOT NULL,
  scientific TEXT NOT NULL,
  first_date TEXT NOT NULL DEFAULT '',   -- ISO yyyy-mm-dd (or '' if unparseable)
  first_location TEXT NOT NULL DEFAULT '',
  n_sightings INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS crop_boxes (
  folder TEXT NOT NULL,
  frame TEXT NOT NULL,
  x1 REAL NOT NULL,
  y1 REAL NOT NULL,
  x2 REAL NOT NULL,
  y2 REAL NOT NULL,
  reid_old TEXT NOT NULL DEFAULT '',
  reid_new TEXT NOT NULL DEFAULT '',
  snapshot TEXT NOT NULL DEFAULT '',   -- JSON: original candidate fields (for Undo)
  updated_at TEXT NOT NULL,
  PRIMARY KEY (folder, frame)
);
"""


# ---------------------------------------------------------------------------
# Data loading (runs/, regions.toml, caches)
# ---------------------------------------------------------------------------

def _parse_top3(s: str) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for part in s.split("; "):
        part = part.strip()
        if not part or " (" not in part:
            continue
        name, p = part.rsplit(" (", 1)
        out.append((name, "", float(p.rstrip(")"))))
    return out


def load_candidates(folder: str) -> list[dict]:
    """Parsed rows of runs/<folder>/candidates.csv, time-ordered."""
    p = RUNS / folder / "candidates.csv"
    rows: list[dict] = []
    with p.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "enc_id": r["enc_id"],
                "time_start": r["time_start"],
                "time_end": r["time_end"],
                "n_frames": int(r["n_frames"]),
                "best_frame": r["best_frame"],
                "det_conf": float(r["det_conf"]),
                "raw_top3": _parse_top3(r["raw_top3"]),
                "adj_top3": _parse_top3(r["adj_top3"]),
                "possible_lifer": r["possible_lifer"] == "True",
                "lifer_lookalike": r["lifer_lookalike"] or None,
                "rare_here": r["rare_here"] == "True",
                "rare_evidence": (r.get("rare_evidence") or "").strip(),
            })
    rows.sort(key=lambda c: c["time_start"])
    return rows


def load_nodet(folder: str) -> list[dict]:
    """No-detection entries for a run's "No bird detected" section.

    New runs write runs/<folder>/nodet.json (one entry per encounter with no
    usable detection: enc_id, time_start, time_end, n_frames, best_frame).
    Older runs have only per-frame thumbnails in crops/nodet/ (no encounter
    info); those fall back to {"file": name} entries.
    """
    p = RUNS / folder / "nodet.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [d for d in data
                        if isinstance(d, dict) and d.get("enc_id")]
        except (json.JSONDecodeError, OSError):
            pass  # corrupt nodet.json: fall back to the thumbnails
    d = RUNS / folder / "crops" / "nodet"
    if d.is_dir():
        return [{"file": n.name} for n in sorted(d.iterdir())
                if n.is_file() and n.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return []


def _load_nodet_entries(folder: str) -> list[dict]:
    p = RUNS / folder / "nodet.json"
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_nodet_entries(folder: str, entries: list[dict]) -> None:
    p = RUNS / folder / "nodet.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")


def _remove_nodet_enc(folder: str, enc_id: str) -> None:
    """Drop an encounter from nodet.json (it became a real candidate)."""
    p = RUNS / folder / "nodet.json"
    if not p.is_file():
        return
    _save_nodet_entries(folder,
                        [e for e in _load_nodet_entries(folder)
                         if e.get("enc_id") != enc_id])


def _add_nodet_enc(folder: str, entry: dict) -> None:
    """Put an encounter back into nodet.json (a crop fix was undone)."""
    entries = _load_nodet_entries(folder)
    if any(e.get("enc_id") == entry.get("enc_id") for e in entries):
        return
    entries.append({k: entry[k] for k in
                    ("enc_id", "time_start", "time_end", "n_frames", "best_frame")
                    if k in entry})
    entries.sort(key=lambda e: str(e.get("time_start", "")))
    _save_nodet_entries(folder, entries)


def _candidate_fieldnames(folder: str) -> list[str]:
    p = RUNS / folder / "candidates.csv"
    if p.is_file():
        with p.open() as f:
            names = csv.DictReader(f).fieldnames
        if names:
            return list(names)
    return ["enc_id", "time_start", "time_end", "n_frames", "best_frame",
            "det_conf", "raw_top3", "adj_top3", "possible_lifer",
            "lifer_lookalike", "rare_here", "rare_evidence"]


def _insert_candidate_row(folder: str, enc_id: str, base: dict,
                          fields: dict) -> None:
    """Append a candidate row for a no-detection encounter that got a manual
    crop fix: it had no row in candidates.csv before (task 23)."""
    path = RUNS / folder / "candidates.csv"
    fieldnames = _candidate_fieldnames(folder)
    row = {
        "enc_id": enc_id,
        "time_start": base.get("time_start", ""),
        "time_end": base.get("time_end", ""),
        "n_frames": base.get("n_frames", 0),
        "best_frame": base.get("best_frame", ""),
        "det_conf": f"{float(base.get('det_conf', 0.0)):.3f}",
        "raw_top3": _fmt_top3(fields["raw_top3"]),
        "adj_top3": _fmt_top3(fields["adj_top3"]),
        "possible_lifer": str(fields["possible_lifer"]),
        "lifer_lookalike": fields.get("lifer_lookalike") or "",
        "rare_here": str(fields["rare_here"]),
        "rare_evidence": fields.get("rare_evidence") or "",
    }
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if path.stat().st_size == 0:
            w.writeheader()
        w.writerow(row)


def _remove_candidate_row(folder: str, enc_id: str) -> None:
    """Delete a candidate row (its crop fix was undone; the encounter goes
    back to "No bird detected")."""
    path = RUNS / folder / "candidates.csv"
    if not path.is_file():
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader if r.get("enc_id") != enc_id]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def list_folders() -> list[str]:
    if not RUNS.is_dir():
        return []
    return sorted(d.name for d in RUNS.iterdir() if (d / "candidates.csv").is_file())


def region_for(folder: str) -> dict | None:
    """First regions.toml entry whose glob matches the folder name."""
    try:
        cfg = tomllib.loads(find_regions_toml().read_text())
    except FileNotFoundError:
        return None
    for r in cfg.get("region", []):
        if fnmatch.fnmatch(folder, r["glob"]):
            return r
    return None


def region_name(folder: str) -> str:
    return region_names(folder)[0]


def region_names(folder: str) -> tuple[str, str]:
    """(name, short_name) for a folder's region; iNat places API fallback for
    keys missing from regions.toml; folder name as last resort."""
    name, short = region_display_names(region_for(folder), INAT_CACHE)
    if not name:
        name = folder
    if not short:
        short = name
    return name, short


def _regional_ids(place_id: int | None) -> set[int]:
    """Taxon ids with any research-grade observation in the place (all-time)."""
    if not place_id:
        return set()
    f = INAT_CACHE / f"species_counts_{place_id}_alltime.json"
    if not f.exists():
        return set()
    return set(int(k) for k in json.loads(f.read_text()))


_aves: list[tuple[int, str, str]] | None = None
_aves_mtime: float = -1.0


def aves_taxa() -> list[tuple[int, str, str]]:
    """(taxon_id, common, scientific) for the full iNat Aves list (cached).

    Re-read when the cache file changes (e.g. after a taxonomy re-fetch).
    """
    global _aves, _aves_mtime
    f = INAT_CACHE / "aves_species.json"
    mtime = f.stat().st_mtime if f.exists() else -1.0
    if _aves is None or mtime != _aves_mtime:
        rows = json.loads(f.read_text()) if f.exists() else []
        _aves = [(v["id"], v.get("common") or "", v["scientific"]) for v in rows]
        _aves_mtime = mtime
    return _aves


def _parse_date(s: str):
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _valid_tid(tid: str) -> bool:
    """A trip ID is a single path segment: no slashes, no dot-segments."""
    return bool(tid) and "/" not in tid and tid not in (".", "..")


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def _import_lifelist(conn: sqlite3.Connection, csv_path: Path) -> None:
    """Import data/lifelist.csv into the lifelist table (re-import on change)."""
    if not csv_path.exists():
        return
    sha = hashlib.sha1(csv_path.read_bytes()).hexdigest()
    row = conn.execute("SELECT value FROM meta WHERE key='lifelist_sha'").fetchone()
    if row and row[0] == sha:
        return
    conn.execute("DELETE FROM lifelist")
    agg: dict[int, dict] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if not (r.get("taxon_id") or "").isdigit():
                continue  # unmatched lifelist entry: no taxon to key on
            tid = int(r["taxon_id"])
            d = _parse_date(r.get("date") or "")
            cur = agg.get(tid)
            if cur is None:
                agg[tid] = {
                    "common": r["common_name"], "scientific": r["scientific_name"],
                    "date": d, "raw_date": r.get("date", ""),
                    "location": r.get("location", ""), "n": 1,
                }
            else:
                cur["n"] += 1
                if d is not None and (cur["date"] is None or d < cur["date"]):
                    cur["date"] = d
                    cur["raw_date"] = r.get("date", "")
                    cur["location"] = r.get("location", "")
    for tid, v in agg.items():
        conn.execute(
            "INSERT INTO lifelist (taxon_id, common, scientific, first_date, "
            "first_location, n_sightings) VALUES (?,?,?,?,?,?)",
            (tid, v["common"], v["scientific"],
             v["date"].isoformat() if v["date"] else "",
             v["location"], v["n"]),
        )
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('lifelist_sha', ?)", (sha,))
    conn.commit()


def _decisions_for(conn: sqlite3.Connection, folder: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in conn.execute(
            "SELECT enc_id, best_frame, action, species, scientific, note, updated_at "
            "FROM decisions WHERE folder=?", (folder,)):
        out[(r["enc_id"], r["best_frame"])] = {
            "action": r["action"], "species": r["species"],
            "scientific": r["scientific"], "note": r["note"],
            "updated_at": r["updated_at"],
        }
    return out


def _upsert_decision(conn: sqlite3.Connection, folder: str, enc_id: str, best_frame: str,
                     action: str, species: str, scientific: str, note: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO decisions (folder, enc_id, best_frame, action, species, scientific, "
        "note, updated_at) VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(folder, enc_id, best_frame) DO UPDATE SET "
        "action=excluded.action, species=excluded.species, "
        "scientific=excluded.scientific, note=excluded.note, updated_at=excluded.updated_at",
        (folder, enc_id, best_frame, action, species, scientific, note, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# View models
# ---------------------------------------------------------------------------

def _badges(c: dict) -> list[str]:
    b: list[str] = []
    if c["rare_here"]:
        b.append("rare")
    if c["possible_lifer"]:
        b.append("lifer")
    if c["lifer_lookalike"]:
        b.append(f"lookalike:{c['lifer_lookalike']}")
    return b


def _group_section(encs: list[dict]) -> str:
    if any(e["rare_here"] for e in encs):
        return "rare"
    if any(e["possible_lifer"] for e in encs):
        return "lifer"
    if any(e["lifer_lookalike"] for e in encs):
        return "lookalike"
    return "other"


def _enc_view(folder: str, c: dict, decision: dict | None,
              conn: sqlite3.Connection | None = None) -> dict:
    top1 = c["adj_top3"][0]
    frame = c["best_frame"]
    override = get_override(conn, folder, frame) if conn is not None else None
    reid = get_reid(conn, folder, frame) if conn is not None else None
    return {
        "id": c["enc_id"],
        "species": top1[0],
        "sci": top1[1],
        "time": c["time_start"][11:16],
        "date": c["time_start"][:10],
        "n_frames": c["n_frames"],
        "best_frame": c["best_frame"],
        "det_conf": c["det_conf"],
        "crop": f"/media/{folder}/crops/{c['enc_id']}.jpg",
        "frame": f"/media/{folder}/frames/{c['enc_id']}.jpg",
        "src": f"/media/{folder}/frames/src/{c['enc_id']}.jpg",
        "override": to_list(override) if override else None,
        "has_override": override is not None,
        "reid": reid,
        "badges": _badges(c),
        "rare_evidence": c.get("rare_evidence", ""),
        "adj_top3": [list(x) for x in c["adj_top3"]],
        "raw_top3": [list(x) for x in c["raw_top3"]],
        "raw_map": {n: p for n, _s, p in c["raw_top3"]},
        "decision": decision,
    }


def build_trip(folder: str, conn: sqlite3.Connection) -> dict:
    """Full trip view model: species groups + no-bird frames + decisions."""
    cands = load_candidates(folder)
    decs = _decisions_for(conn, folder)
    for c in cands:
        c["decision"] = decs.get((c["enc_id"], c["best_frame"]))

    # group by adjusted top-1 (the species the pipeline is showing)
    by_species: dict[str, list[dict]] = {}
    for c in cands:
        key = c["adj_top3"][0][0].lower()
        by_species.setdefault(key, []).append(c)

    groups: list[dict] = []
    for key, encs in by_species.items():
        encs.sort(key=lambda e: e["time_start"])
        views = [_enc_view(folder, c, c["decision"], conn) for c in encs]
        top1 = encs[0]["adj_top3"][0]
        best = max(views, key=lambda v: v["det_conf"])
        badges: list[str] = []
        for v in views:
            for b in v["badges"]:
                if b not in badges:
                    badges.append(b)
        t0 = encs[0]["time_start"]
        t1 = max(e["time_end"] for e in encs)
        span = f"{t0[11:16]}\u2013{t1[11:16]}"
        if t0[:10] != t1[:10]:
            span = f"{t0[:10]} {t0[11:16]}\u2013{t1[:10]} {t1[11:16]}"
        groups.append({
            "key": key,
            "species": top1[0],
            "scientific": top1[1],
            "section": _group_section(encs),
            "badges": badges,
            "n": len(views),
            "frames": sum(v["n_frames"] for v in views),
            "span": span,
            "best_i": views.index(best),
            "encs": views,
        })
    groups.sort(key=lambda g2: g2["encs"][0]["time"])
    # global index: the template renders groups inside per-section loops, so
    # loop.index0 there would restart at 0 for every section; the JS DATA
    # array is flat, so the DOM data-g attribute must use this global index
    for i, g3 in enumerate(groups):
        g3["gi"] = i

    # No-detection entries: encounters (nodet.json, task 23) plus, for older
    # runs, lone per-frame thumbnails (no enc id, no fix-crop).
    nodet = []
    for n in load_nodet(folder):
        if n.get("enc_id"):
            nodet.append({
                "id": n["enc_id"],
                "file": n.get("best_frame", ""),
                "time": str(n.get("time_start", ""))[11:16],
                "n_frames": int(n.get("n_frames", 0)),
                "best_frame": n.get("best_frame", ""),
                "thumb": f"/media/{folder}/crops/nodet/{n['enc_id']}.jpg",
                "enc": True,
            })
        else:
            nodet.append({
                "id": n["file"].rsplit(".", 1)[0],
                "file": n["file"],
                "time": "",
                "n_frames": 0,
                "best_frame": n["file"],
                "thumb": f"/media/{folder}/crops/nodet/{n['file']}",
                "enc": False,
            })

    total = len(cands)
    decided = sum(1 for c in cands if c["decision"])
    if cands:
        date_range = f"{cands[0]['time_start'][:10]}"
        if cands[-1]["time_start"][:10] != date_range:
            date_range = f"{date_range} \u2013 {cands[-1]['time_start'][:10]}"
    else:
        date_range = ""

    # rare badge/section naming: where + when "rare" is, from the region's
    # display names and the photo's month window (m-1 .. m+1, wrapping)
    rname, rshort = region_names(folder)
    m = int(cands[0]["time_start"][5:7]) if cands else 0
    window_label = ""
    if m:
        first = m - 1 if m > 1 else 12
        last = m + 1 if m < 12 else 1
        window_label = f"{calendar.month_abbr[first]}\u2013{calendar.month_abbr[last]}"
    sections = list(SECTIONS)
    if m and rname:
        sections[0] = ("rare", f"Rare for {rname} in {window_label}")
    rare_badge = (f"Rare in {rshort} \u00b7 {calendar.month_abbr[m]}"
                  if m and rshort else "Rare here")

    return {
        "folder": folder,
        "date_range": date_range,
        "region": rname,
        "region_short": rshort,
        "region_month": calendar.month_abbr[m] if m else "",
        "region_window": window_label,
        "rare_badge": rare_badge,
        "place_id": (region_for(folder) or {}).get("place_id"),
        "sections": sections,
        "groups": groups,
        "nodet": nodet,
        "total": total,
        "decided": decided,
    }


def build_trips(conn: sqlite3.Connection) -> list[dict]:
    out: list[dict] = []
    for folder in list_folders():
        cands = load_candidates(folder)
        decs = _decisions_for(conn, folder)
        total = len(cands)
        decided = sum(1 for c in cands if decs.get((c["enc_id"], c["best_frame"])))
        pending = sum(1 for c in cands
                      if (c["rare_here"] or c["possible_lifer"])
                      and not decs.get((c["enc_id"], c["best_frame"])))
        date = min(c["time_start"][:10] for c in cands) if cands else ""
        out.append({
            "folder": folder,
            "date": date,
            "total": total,
            "decided": decided,
            "pct": round(100 * decided / total) if total else 0,
            "pending": pending,
            "region": region_name(folder),
        })
    return out


def _frame_cache(folder: str, frame: str) -> dict | None:
    """The pipeline's cached analysis JSON for one frame (sharpness, dets)."""
    p = lifer_home() / "cache" / folder / f"{Path(frame).stem}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _best_det(cache: dict | None) -> list | None:
    """(x1,y1,x2,y2,conf) of the highest-confidence detection, else None."""
    dets = (cache or {}).get("dets") or []
    if not dets:
        return None
    return max(dets, key=lambda d: d[4] if len(d) > 4 else 0.0)


def _quality_score(enc: dict) -> tuple[float, float]:
    """(sharpness, detection_confidence) for ranking the best photos."""
    cache = _frame_cache(enc["folder"], enc["best_frame"])
    sharp = float((cache or {}).get("sharpness", -1.0))
    conf = float(enc.get("det_conf", 0.0))
    return sharp, conf


def _find_candidate(folder: str, enc_id: str):
    for c in load_candidates(folder):
        if c["enc_id"] == enc_id:
            return c
    for n in load_nodet(folder):
        if n.get("enc_id") == enc_id:
            # No candidate row (no-detection encounter, task 23): a minimal
            # view so the crop editor can turn it into a real encounter.
            return {
                "enc_id": enc_id,
                "time_start": n.get("time_start", ""),
                "time_end": n.get("time_end", ""),
                "n_frames": int(n.get("n_frames", 0)),
                "best_frame": n.get("best_frame", ""),
                "det_conf": 0.0,
                "raw_top3": [], "adj_top3": [], "top1": "",
                "possible_lifer": False, "lifer_lookalike": None,
                "rare_here": False, "rare_evidence": "",
                "nodet": True,
            }
    return None


def _frame_size(folder: str, frame: str) -> tuple[int, int]:
    """Full-pixel (w, h) of a frame from the cache/detection, else by
    reading the source photo (read-only)."""
    cache = _frame_cache(folder, frame) or {}
    if cache.get("width") and cache.get("height"):
        return int(cache["width"]), int(cache["height"])
    photo = source_folder_for_trip(folder) / frame
    if not photo.is_file():
        raise FileNotFoundError(f"missing photo: {photo}")
    image = load_photo(photo)
    return image.size


def _crop_square(image, box: Box, margin: float = 0.20):
    """Square crop around a box with ~margin, clamped to the frame (mirrors
    detect.square_crop without pulling torch/ultralytics into the service)."""
    W, H = image.size
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    w, h = x2 - x1, y2 - y1
    s = max(w, h) * (1.0 + 2.0 * margin)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = min(s, W, H)
    x0 = min(max(cx - s / 2.0, 0.0), W - s)
    y0 = min(max(cy - s / 2.0, 0.0), H - s)
    crop = (int(round(x0)), int(round(y0)), int(round(x0 + s)), int(round(y0 + s)))
    return image.crop(crop)


def _regenerate_crop(folder: str, enc_id: str, frame: str,
                     box: Box) -> Image.Image:
    """Regenerate the review app's crop + frame preview + box-less source
    preview from the full-size source image. Returns the crop PIL image."""
    photo = source_folder_for_trip(folder) / frame
    if not photo.is_file():
        raise FileNotFoundError(f"missing photo: {photo}")
    image = load_photo(photo)
    crop = _crop_square(image, box)
    save_crop(crop, RUNS / folder / "crops" / f"{enc_id}.jpg")
    save_frame_preview(image, box, RUNS / folder / "frames" / f"{enc_id}.jpg")
    save_frame_preview(image, None, RUNS / folder / "frames" / "src" / f"{enc_id}.jpg")
    return crop


def _fmt_top3(items: list) -> str:
    return "; ".join(f"{name} ({float(p):.3f})" for name, _s, p in items)


def _rewrite_candidate(folder: str, enc_id: str, fields: dict) -> None:
    """Update one encounter's derived fields in candidate.csv (preserves the
    file's column order and the other rows' content)."""
    path = RUNS / folder / "candidates.csv"
    if not path.is_file():
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or [
            "enc_id", "time_start", "time_end", "n_frames", "best_frame",
            "det_conf", "raw_top3", "adj_top3", "possible_lifer",
            "lifer_lookalike", "rare_here", "rare_evidence"]
        rows = list(reader)
    changed = False
    for r in rows:
        if r["enc_id"] == enc_id:
            r["raw_top3"] = _fmt_top3(fields["raw_top3"])
            r["adj_top3"] = _fmt_top3(fields["adj_top3"])
            r["possible_lifer"] = str(fields["possible_lifer"])
            r["lifer_lookalike"] = fields["lifer_lookalike"] or ""
            r["rare_here"] = str(fields["rare_here"])
            r["rare_evidence"] = fields["rare_evidence"] or ""
            changed = True
    if changed:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def _snapshot_fields(c: dict) -> str:
    """JSON encoding of the pre-fix candidate fields (for Undo)."""
    return json.dumps({
        "raw_top3": [[n, p] for n, _s, p in c["raw_top3"]],
        "adj_top3": [[n, p] for n, _s, p in c["adj_top3"]],
        "possible_lifer": c["possible_lifer"],
        "lifer_lookalike": c["lifer_lookalike"] or "",
        "rare_here": c["rare_here"],
        "rare_evidence": c.get("rare_evidence", ""),
    }, ensure_ascii=False)


def _restore_snapshot(snapshot: dict, folder: str, enc_id: str) -> None:
    """Rebuild the fields dict from a snapshot and rewrite the candidate row."""
    fields = {
        "raw_top3": [(n, "", p) for n, p in snapshot.get("raw_top3", [])],
        "adj_top3": [(n, "", p) for n, p in snapshot.get("adj_top3", [])],
        "possible_lifer": snapshot.get("possible_lifer", False),
        "lifer_lookalike": snapshot.get("lifer_lookalike", ""),
        "rare_here": snapshot.get("rare_here", False),
        "rare_evidence": snapshot.get("rare_evidence", ""),
    }
    _rewrite_candidate(folder, enc_id, fields)


def _ensure_src_preview(folder: str, enc_id: str) -> Path | None:
    """Return the box-less full-frame preview for an encounter, generating it
    from the best frame's embedded JPEG the first time (cached under runs/)."""
    dst = RUNS / folder / "frames" / "src" / f"{enc_id}.jpg"
    if dst.is_file():
        return dst
    c = _find_candidate(folder, enc_id)
    if c is None:
        return None
    photo = source_folder_for_trip(folder) / c["best_frame"]
    if not photo.is_file():
        return None
    try:
        image = load_photo(photo)
    except (RuntimeError, OSError):
        return None
    save_frame_preview(image, None, dst)
    return dst


def _sighting_view(s: dict) -> str | None:
    """Thumb URL for a sighting, or None for ODS rows (no photo)."""
    if s["enc"] is None:
        return None
    return f"/media/{s['folder']}/crops/{s['enc']['enc_id']}.jpg"


def build_sightings(conn: sqlite3.Connection) -> dict[str, dict]:
    """Per species (keyed by common-name lower): its sightings, newest first.

    A sighting is one species on one trip, counted once; an imported
    LIFELIST.ods row is also one sighting. Only confirm/change decisions create
    trip sightings. Both the review app and `publish` consume this so they
    agree.

    Each species value is: {common, scientific, taxon_id, sightings:[...]} where
    each sighting is {date, date_dt, place, source:'ods'|'trip', folder, enc,
    encs}. For a trip sighting `enc` is the best confirmed encounter of that
    trip (the one the thumbnails use) and `encs` is every confirmed encounter
    of that trip ordered by quality, so publish can fall back if the best frame
    is unusable.
    """
    _import_lifelist(conn, LIFELIST_CSV)
    taxa = aves_taxa()
    name2taxon: dict[str, tuple[int, str, str]] = {}
    for tid, common, sci in taxa:
        if common:
            name2taxon.setdefault(common.lower(), (tid, common, sci))
        name2taxon.setdefault(sci.lower(), (tid, common, sci))

    recs: dict[str, dict] = {}

    def rec(key: str, display: str = "") -> dict:
        r = recs.get(key)
        if r is None:
            r = recs[key] = {
                "common": display or key, "scientific": "",
                "taxon_id": None, "sightings": [],
            }
        return r

    # ---- imported LIFELIST.ods rows: one sighting each ----
    if LIFELIST_CSV.exists():
        with LIFELIST_CSV.open() as f:
            for r in csv.DictReader(f):
                tid = (r.get("taxon_id") or "").strip()
                if not tid.isdigit():
                    continue
                common = (r.get("common_name") or "").strip()
                if not common:
                    continue
                raw = (r.get("date") or "").strip()
                dt = _parse_date(raw)
                entry = rec(common.lower(), common)
                entry["common"] = common
                entry["scientific"] = (r.get("scientific_name") or "").strip()
                entry["taxon_id"] = int(tid)
                entry["sightings"].append({
                    "date": dt.isoformat() if dt else raw,
                    "date_dt": dt,
                    "place": (r.get("location") or "").strip(),
                    "source": "ods",
                    "folder": None, "enc": None, "encs": [],
                })

    # ---- confirmed trip sightings: one per (species, folder) ----
    by_key_folder: dict[tuple[str, str], list[dict]] = {}
    cands_cache: dict[str, list[dict]] = {}
    for d in conn.execute(
            "SELECT folder, enc_id, best_frame, species FROM decisions "
            "WHERE action IN ('confirm','change')"):
        species = (d["species"] or "").strip()
        if not species:
            continue
        folder = d["folder"]
        cands = cands_cache.get(folder)
        if cands is None:
            if not (RUNS / folder / "candidates.csv").is_file():
                continue
            cands = cands_cache[folder] = load_candidates(folder)
        enc = next((c for c in cands
                    if c["enc_id"] == d["enc_id"] and c["best_frame"] == d["best_frame"]),
                   None)
        if enc is None:
            continue
        enc = dict(enc)             # don't mutate the shared candidates cache
        enc["folder"] = folder
        by_key_folder.setdefault((species.lower(), folder), []).append(enc)

    for (key, folder), encs in by_key_folder.items():
        encs.sort(key=_quality_score, reverse=True)
        best = encs[0]
        dt = _parse_date((best.get("time_start") or "")[:10])
        entry = rec(key)
        if not entry["common"]:
            entry["common"] = key
        if not entry["scientific"]:
            t = name2taxon.get(key)
            if t:
                entry["taxon_id"] = t[0]
                entry["common"] = t[1]
                entry["scientific"] = t[2]
        entry["sightings"].append({
            "date": (best.get("time_start") or "")[:10],
            "date_dt": dt,
            "place": region_name(folder),
            "source": "trip",
            "folder": folder, "enc": best, "encs": encs,
        })

    for entry in recs.values():
        s = entry["sightings"]
        s.sort(key=lambda x: (x["date_dt"] or date.min, x["source"]), reverse=True)

    return recs


def _summarize(rec: dict):
    """(first, latest, latest_photo) raw sightings, newest-first position.

    The rec must have at least one sighting. `latest_photo` is the most recent
    sighting that has a photo (an enc), even when the most recent sighting
    itself is a photo-less ODS row.
    """
    s = rec["sightings"]
    first = s[-1]
    latest = s[0]
    latest_photo = next((x for x in s if x["enc"] is not None), None)
    return first, latest, latest_photo


def build_lifelist(conn: sqlite3.Connection, sort: str = "taxon") -> list[dict]:
    """Merge imported lifelist rows with confirmed sightings from decisions.

    Every row has: taxon_id, common, scientific, first_date, first_location,
    latest_date, latest_location, latest_photo_date, count, new, thumb, and the
    per-sighting list `sightings` ({date, place, thumb}). The row thumbnail is
    the most-recent-sighting photo (or the most recent sighting that has a
    photo, labelled with `latest_photo_date`).
    """
    recs = build_sightings(conn)
    taxa = aves_taxa()
    rank = {tid: i for i, (tid, _c, _s) in enumerate(taxa)}

    rows: list[dict] = []
    for _key, rec in recs.items():
        if not rec["sightings"]:
            continue
        first, latest, latest_photo = _summarize(rec)
        rows.append({
            "taxon_id": rec["taxon_id"],
            "common": rec["common"],
            "scientific": rec["scientific"],
            "first_date": first["date"],
            "first_location": first["place"],
            "latest_date": latest["date"],
            "latest_location": latest["place"],
            "latest_photo_date": latest_photo["date"] if latest_photo else None,
            "count": len(rec["sightings"]),
            "new": not any(x["source"] == "ods" for x in rec["sightings"]),
            "thumb": _sighting_view(latest_photo) if latest_photo else None,
            "sightings": [
                {"date": x["date"], "place": x["place"], "thumb": _sighting_view(x)}
                for x in rec["sightings"]
            ],
            "_rank": rank.get(rec["taxon_id"]) if rec["taxon_id"] is not None else None,
            "_sort_date": first["date"],
        })

    if sort == "date":
        rows.sort(key=lambda r: (r["_sort_date"] or "9999-12-31",
                                 r["common"].lower()))
    else:
        rows.sort(key=lambda r: (r["_rank"] if r["_rank"] is not None else 10 ** 9,
                                 r["common"].lower()))
    return rows


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app(db_path: Path) -> Flask:
    _init_db(db_path)
    app = Flask(__name__, template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False

    @app.before_request
    def _open_db():
        g.db = sqlite3.connect(str(db_path))
        g.db.row_factory = sqlite3.Row
        _import_lifelist(g.db, LIFELIST_CSV)

    @app.teardown_request
    def _close_db(_exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    # ---- pages ------------------------------------------------------------

    @app.route("/")
    def page_trips():
        return render_template("index.html", trips=build_trips(g.db), css=CSS + EXTRA_CSS)

    @app.route("/trip/<folder>")
    def page_trip(folder: str):
        if not (RUNS / folder / "candidates.csv").is_file():
            abort(404)
        trip = build_trip(folder, g.db)
        data_json = json.dumps(trip, default=str).replace("</", "<\\/")
        return render_template("trip.html", t=trip, data_json=data_json,
                               css=CSS + EXTRA_CSS)

    @app.route("/lifelist")
    def page_lifelist():
        sort = request.args.get("sort", "taxon")
        if sort not in ("taxon", "date"):
            sort = "taxon"
        rows = build_lifelist(g.db, sort)
        for r in rows:
            r.pop("_rank", None)
            r.pop("_sort_date", None)
        return render_template("lifelist.html", rows=rows, sort=sort, css=CSS + EXTRA_CSS)

    # ---- media (runs/<folder>/...) -----------------------------------------

    @app.route("/media/<folder>/<path:rel>")
    def media(folder: str, rel: str):
        # The crop editor wants a box-less full-frame preview; generate it from
        # the full-size embedded JPEG the first time (cached) so later loads are
        # instant. The blob lives under runs/ so it is never written to BIRDING.
        if rel.startswith("frames/src/") and rel.endswith(".jpg"):
            p = _ensure_src_preview(folder, rel[len("frames/src/"):-4])
            if p is None:
                abort(404)
            return send_file(p)
        base = (RUNS / folder).resolve()
        p = (base / rel).resolve()
        if not str(p).startswith(str(base) + os.sep) or not p.is_file():
            abort(404)
        return send_file(p)

    # ---- API ----------------------------------------------------------------

    @app.route("/api/trips")
    def api_trips():
        return jsonify(build_trips(g.db))

    @app.route("/api/trip/<folder>")
    def api_trip(folder: str):
        if not (RUNS / folder / "candidates.csv").is_file():
            return jsonify(error="unknown folder"), 404
        return jsonify(build_trip(folder, g.db))

    @app.route("/api/species")
    def api_species():
        """Autocomplete over the cached iNat Aves list; regional species first."""
        q = (request.args.get("q") or "").strip()
        limit = min(request.args.get("limit", default=12, type=int), 25)
        if not q:
            return jsonify([])
        region = region_for(request.args.get("folder") or "")
        place_id = region.get("place_id") if region else request.args.get("place", type=int)
        regional = _regional_ids(place_id)
        taxa = aves_taxa()
        keys = [c for _t, c, _s in taxa if c]
        name2 = {c: (t, s) for t, c, s in taxa if c}
        scored = process.extract(q, keys, scorer=fuzz.WRatio, limit=40, score_cutoff=60)
        out = []
        for name, score, _i in scored:
            tid, sci = name2[name]
            out.append({"common": name, "scientific": sci, "taxon_id": tid,
                        "region": tid in regional, "score": int(score)})
        out.sort(key=lambda x: (0 if x["region"] else 1, -x["score"], x["common"].lower()))
        return jsonify(out[:limit])

    @app.route("/api/decisions", methods=["POST"])
    def api_decisions():
        p = request.get_json(force=True) or {}
        folder = p.get("folder") or ""
        enc_id = p.get("enc_id") or ""
        best_frame = p.get("best_frame") or ""
        action = p.get("action") or ""
        if not (RUNS / folder / "candidates.csv").is_file() or not enc_id or not best_frame:
            return jsonify(ok=False, error="bad folder / enc_id / best_frame"), 400
        if action == "clear":
            g.db.execute("DELETE FROM decisions WHERE folder=? AND enc_id=? AND best_frame=?",
                         (folder, enc_id, best_frame))
            g.db.commit()
            return jsonify(ok=True)
        if action not in DECISION_ACTIONS:
            return jsonify(ok=False, error=f"bad action {action!r}"), 400
        species = (p.get("species") or "").strip()
        if action in ("confirm", "change") and not species:
            return jsonify(ok=False, error="species required"), 400
        _upsert_decision(g.db, folder, enc_id, best_frame, action, species,
                         (p.get("scientific") or "").strip(), (p.get("note") or "").strip())
        return jsonify(ok=True)

    @app.route("/api/group", methods=["POST"])
    def api_group():
        """Group action: confirm every encounter of a species group, or mark
        the group as not this species (all -> unsure)."""
        p = request.get_json(force=True) or {}
        folder = p.get("folder") or ""
        species = (p.get("species") or "").strip()
        action = p.get("action") or ""
        if action not in ("confirm", "unsure"):
            return jsonify(ok=False, error="group action must be confirm|unsure"), 400
        if not (RUNS / folder / "candidates.csv").is_file() or not species:
            return jsonify(ok=False, error="bad folder / species"), 400
        cands = load_candidates(folder)
        target = [c for c in cands if c["adj_top3"][0][0].lower() == species.lower()]
        if not target:
            return jsonify(ok=False, error="no such species group"), 404
        sci = (p.get("scientific") or "").strip()
        for c in target:
            _upsert_decision(g.db, folder, c["enc_id"], c["best_frame"], action,
                             species if action == "confirm" else "", sci, "")
        return jsonify(ok=True, updated=len(target))

    @app.route("/api/cropboxes")
    def api_cropboxes():
        """Boxes + frame size + current override for the crop editor of one
        encounter's best frame (full-pixel coordinates)."""
        folder = request.args.get("folder") or ""
        enc_id = request.args.get("enc_id") or ""
        c = _find_candidate(folder, enc_id)
        if c is None:
            return jsonify(ok=False, error="unknown encounter"), 404
        frame = c["best_frame"]
        cache = _frame_cache(folder, frame) or {}
        dets = cache.get("dets") or []
        try:
            W, H = _frame_size(folder, frame)
        except FileNotFoundError:
            return jsonify(ok=False, error="missing NEF"), 404
        override = get_override(g.db, folder, frame)
        return jsonify(ok=True, frame=frame, w=W, h=H,
                       boxes=[[float(x) for x in d] for d in dets],
                       override=to_list(override) if override else None,
                       src=f"/media/{folder}/frames/src/{enc_id}.jpg")

    @app.route("/api/crop", methods=["POST"])
    def api_crop():
        """Set/clear a crop override for one encounter's best frame, regenerate
        its card crop + previews, and re-identify the new crop (CPU-only)."""
        p = request.get_json(force=True) or {}
        folder = p.get("folder") or ""
        enc_id = p.get("enc_id") or ""
        box_in = p.get("box")          # [x1,y1,x2,y2] full-pixel, or None
        undo = bool(p.get("undo"))
        c = _find_candidate(folder, enc_id)
        if c is None or not (RUNS / folder / "candidates.csv").is_file():
            return jsonify(ok=False, error="unknown encounter"), 400
        frame = c["best_frame"]
        old_top1 = c.get("top1") or (c["adj_top3"][0][0] if c["adj_top3"] else "")
        if undo:
            snapshot = get_snapshot(g.db, folder, frame)
            clear_override(g.db, folder, frame)
            box = get_box(folder, frame, g.db)   # override cleared -> detector box
            if box is None:
                # No detector box: this was a no-detection encounter that got
                # a manual crop fix (task 23). Revert to "No bird detected":
                # drop the candidate row and restore the nodet entry. (A
                # detected encounter always has a detector box to fall back
                # to, so this can't misfire.)
                _remove_candidate_row(folder, enc_id)
                _add_nodet_enc(folder, c)
                return jsonify(ok=True, overridden=False, undo=True, nodet=True)
            try:
                _regenerate_crop(folder, enc_id, frame, box)
            except (FileNotFoundError, RuntimeError) as e:
                return jsonify(ok=False, error=str(e)), 404
            if snapshot:
                _restore_snapshot(snapshot, folder, enc_id)
            return jsonify(ok=True, overridden=False, undo=True,
                           crop=f"/media/{folder}/crops/{enc_id}.jpg",
                           frame=f"/media/{folder}/frames/{enc_id}.jpg")
        if not isinstance(box_in, (list, tuple)) or len(box_in) != 4:
            return jsonify(ok=False, error="box must be [x1,y1,x2,y2]"), 400
        try:
            box = Box(*(float(v) for v in box_in), source="override")
        except (TypeError, ValueError):
            return jsonify(ok=False, error="bad box"), 400
        try:
            W, H = _frame_size(folder, frame)
        except FileNotFoundError:
            return jsonify(ok=False, error="missing NEF"), 404
        if not (0 <= box.x1 < box.x2 <= W and 0 <= box.y1 < box.y2 <= H):
            return jsonify(ok=False, error="box outside frame"), 400
        # A snapshot of the pre-fix fields is kept for Undo (set only once).
        # No-detection encounters have no pre-fix fields: their Undo reverts
        # to the "No bird detected" section instead (see above).
        snapshot = "" if c.get("nodet") else (
            _snapshot_fields(c) if get_snapshot(g.db, folder, frame) is None else "")
        try:
            crop = _regenerate_crop(folder, enc_id, frame, box)
        except (FileNotFoundError, RuntimeError) as e:
            return jsonify(ok=False, error=str(e)), 404
        try:
            from . import reid
            fields = reid.identify_crop(folder, frame, crop)
        except RuntimeError as e:
            set_override(g.db, folder, frame, box, snapshot=snapshot)
            return jsonify(ok=False, error=f"re-identify failed: {e}"), 500
        new_top1 = fields["top1"]
        reid_info = {"old": old_top1, "new": new_top1} if new_top1 != old_top1 else None
        set_override(g.db, folder, frame, box,
                     reid_old=(reid_info["old"] if reid_info else ""),
                     reid_new=(reid_info["new"] if reid_info else ""),
                     snapshot=snapshot)
        if c.get("nodet"):
            # This was a no-detection encounter: it had no row in
            # candidates.csv, so create one and drop it from nodet.json
            # (task 23).
            _insert_candidate_row(folder, enc_id, c, fields)
            _remove_nodet_enc(folder, enc_id)
        else:
            _rewrite_candidate(folder, enc_id, {
                "raw_top3": fields["raw_top3"],
                "adj_top3": fields["adj_top3"],
                "possible_lifer": fields["possible_lifer"],
                "lifer_lookalike": fields["lifer_lookalike"],
                "rare_here": fields["rare_here"],
                "rare_evidence": fields["rare_evidence"],
            })
        return jsonify(ok=True, overridden=True,
                       reid=reid_info,
                       elapsed=fields["elapsed"],
                       crop=f"/media/{folder}/crops/{enc_id}.jpg",
                       frame=f"/media/{folder}/frames/{enc_id}.jpg",
                       adj_top3=[[n, s, p] for n, s, p in fields["adj_top3"]])

    @app.route("/api/lifelist")
    def api_lifelist():
        rows = build_lifelist(g.db, request.args.get("sort", "taxon"))
        for r in rows:
            r.pop("_rank", None)
            r.pop("_sort_date", None)
        return jsonify(rows)

    @app.route("/api/lifelist/export")
    def api_lifelist_export():
        fmt = request.args.get("format", "csv")
        rows = build_lifelist(g.db, request.args.get("sort", "taxon"))
        cols = ["common_name", "scientific_name", "first_date", "first_location",
                "latest_date", "latest_location", "sightings", "new"]
        data = [[r["common"], r["scientific"], r["first_date"], r["first_location"],
                 r["latest_date"], r["latest_location"],
                 r["count"], "yes" if r["new"] else ""] for r in rows]
        if fmt == "csv":
            import io
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(cols)
            w.writerows(data)
            return (buf.getvalue(), 200,
                    {"Content-Type": "text/csv; charset=utf-8",
                     "Content-Disposition": 'attachment; filename="lifer-lifelist.csv"'})
        if fmt == "ods":
            return _ods_export(cols, data)
        return jsonify(ok=False, error="format must be csv|ods"), 400

    # ---- delete trip (task 24) ---------------------------------------------

    @app.route("/api/delete/plan")
    def api_delete_plan():
        """What 'Delete trip' would remove for one trip. Nothing is touched."""
        from .trips_ops import plan_delete
        tid = request.args.get("folder", "")
        if not _valid_tid(tid):
            return jsonify(ok=False, error="bad trip ID"), 400
        return jsonify(ok=True, **plan_delete(lifer_home(), g.db, [tid]))

    @app.route("/api/delete", methods=["POST"])
    def api_delete():
        """Delete a trip's run dir, cache, and DB rows. The client must echo
        the trip ID in `confirm` (typed confirmation in the dialog)."""
        from .trips_ops import apply_delete
        p = request.get_json(force=True) or {}
        tid = p.get("folder") or ""
        if not _valid_tid(tid):
            return jsonify(ok=False, error="bad trip ID"), 400
        if p.get("confirm") != tid:
            return jsonify(ok=False,
                           error="confirmation does not match the trip ID"), 400
        return jsonify(ok=True, **apply_delete(lifer_home(), g.db, [tid]))

    return app


def _ods_export(cols: list[str], data: list[list]) -> tuple:
    from odf.opendocument import OpenDocumentSpreadsheet
    from odf.table import Table, TableColumn, TableRow, TableCell
    from odf.text import P

    doc = OpenDocumentSpreadsheet()
    table = Table(name="Lifelist")
    for _c in cols:
        table.addElement(TableColumn())

    def cell(v):
        c = TableCell()
        c.addElement(P(text=str(v)))
        return c

    head = TableRow()
    for c in cols:
        head.addElement(cell(c))
    table.addElement(head)
    for row in data:
        tr = TableRow()
        for v in row:
            tr.addElement(cell(v))
        table.addElement(tr)
    doc.spreadsheet.addElement(table)
    import io
    buf = io.BytesIO()
    doc.save(buf)
    return (buf.getvalue(), 200,
            {"Content-Type": "application/vnd.oasis.opendocument.spreadsheet",
             "Content-Disposition": 'attachment; filename="lifer-lifelist.ods"'})


def serve_app(host: str, port: int, db_path: Path) -> None:
    if not db_path.is_absolute():
        db_path = lifer_home() / db_path
    app = create_app(db_path)
    # Behind a reverse proxy (e.g. Caddy) the app sees its own socket, not the
    # client's scheme/host. Trust X-Forwarded-Proto/X-Forwarded-Host from one
    # hop so request.is_secure / request.host match what the browser used, and
    # any absolute URLs or scheme-dependent behaviour are correct. Werkzeug's
    # ProxyFix rewrites the WSGI environ; wrap the app's wsgi_app, not the app.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    print(f"lifer review app: http://{host}:{port}  (db: {db_path})")
    app.run(host=host, port=port, debug=False)


# ---------------------------------------------------------------------------
# Extra CSS (added to outputs.CSS; plain string, no f-string fields)
# ---------------------------------------------------------------------------

EXTRA_CSS = """
/* buttons */
.btn {
  font:inherit; font-size:.8rem; font-weight:600; cursor:pointer;
  color:var(--fg); background:var(--card); border:1px solid var(--border);
  border-radius:999px; padding:5px 13px;
}
.btn:hover { border-color:var(--bar); }
.btn.primary { background:var(--lifer-bg); color:var(--lifer-fg); border-color:var(--lifer-bd); }
.btn.sm { font-size:.72rem; padding:3px 10px; }
/* species groups */
.group {
  background:var(--card); border:1px solid var(--border); border-radius:12px;
  margin-bottom:16px; overflow:hidden;
}
.g-head { display:flex; gap:16px; padding:14px; }
.g-best {
  width:230px; min-width:180px; aspect-ratio:1/1; object-fit:cover; border-radius:10px;
  background:var(--thumb-bg); cursor:zoom-in; display:block;
}
.g-info { flex:1; min-width:0; }
.g-title { margin:0; font-size:1.08rem; font-weight:700; line-height:1.35; }
.g-title .sci { font-weight:400; font-style:italic; color:var(--muted); font-size:.9rem; }
.g-meta { color:var(--muted); font-size:.82rem; margin-top:6px; }
.g-strip { display:flex; gap:6px; margin-top:10px; flex-wrap:wrap; }
.g-strip img {
  width:56px; height:56px; object-fit:cover; border-radius:8px; cursor:zoom-in;
  background:var(--thumb-bg); border:1px solid var(--border);
}
.g-strip img.sel { outline:2px solid var(--bar); }
.g-actions { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; align-items:center; }
.g-encs { border-top:1px solid var(--border); padding:6px 14px 12px; }
.enc { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid var(--border); }
.enc:last-child { border-bottom:none; }
.enc-crop {
  width:92px; height:92px; object-fit:cover; border-radius:8px; cursor:zoom-in;
  background:var(--thumb-bg); flex:0 0 auto;
}
.enc-info { flex:1; min-width:0; }
.enc-line { font-size:.8rem; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--muted); }
.enc .top3 { margin-top:6px; max-width:430px; }
.enc-actions { display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; align-items:center; position:relative; }
/* decision chips */
.chip {
  display:inline-block; font-size:.7rem; font-weight:700; padding:2px 9px;
  border-radius:999px; margin-left:8px; vertical-align:middle; white-space:nowrap;
}
.chip.none { background:var(--bar-track); color:var(--muted); font-weight:500; }
.chip.confirm { background:var(--lifer-bg); color:var(--lifer-fg); }
.chip.change { background:var(--la-bg); color:var(--la-fg); }
.chip.notbird { background:var(--rare-bg); color:var(--rare-fg); }
.chip.unsure { background:var(--bar-track); color:var(--muted); }
.chip-note { font-size:.72rem; color:var(--muted); font-style:italic; margin-left:6px; }
/* species autocomplete */
.specpick { position:relative; display:inline-flex; align-items:center; gap:6px; }
.specpick input, .note-input {
  font:inherit; font-size:.78rem; color:var(--fg); background:var(--bg);
  border:1px solid var(--border); border-radius:999px; padding:5px 12px;
}
.specpick input { width:190px; }
.note-input { width:150px; }
.ac-list {
  position:absolute; top:calc(100% + 4px); left:0; z-index:60; min-width:260px; max-width:340px;
  background:var(--card); border:1px solid var(--border); border-radius:10px;
  box-shadow:0 10px 30px rgba(0,0,0,.18); overflow:hidden; display:none;
}
.ac-list.open { display:block; }
.ac-item { padding:7px 12px; font-size:.8rem; cursor:pointer; }
.ac-item:hover, .ac-item.hl { background:var(--bar-track); }
.ac-item i { color:var(--muted); font-size:.74rem; }
.ac-item .ac-region { color:var(--lifer-fg); font-size:.66rem; font-weight:700; margin-left:6px; }
/* lightbox actions */
.lb-actions {
  margin-top:14px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:center;
  max-width:92vw;
}
.lb-actions .btn { color:#fff; background:rgba(255,255,255,.12); border-color:rgba(255,255,255,.3); }
.lb-actions .btn:hover { background:rgba(255,255,255,.24); }
.lb-actions .btn.primary { background:rgba(74,222,128,.22); border-color:rgba(74,222,128,.55); color:#4ade80; }
.lb-actions input {
  color:#fff; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.3);
  border-radius:999px; padding:5px 12px; font-size:.78rem;
}
.lb-hint { width:100%; text-align:center; color:#8b93a0; font-size:.72rem; margin-top:2px; }
.lb-actions .ac-list { background:#171a20; border-color:#2c3340; }
.lb-actions .ac-item { color:#e6e8eb; }
.lb-actions .ac-item:hover, .lb-actions .ac-item.hl { background:#232833; }
/* lightbox: photo + toggle + caption + actions as one centered column
   (task 04); the toggle stays in normal flow as in task 03 */
.lb-inner { max-height:94vh; justify-content:center; gap:12px; }
.lb-img { max-height:56vh; }
.lb-toggle { margin-top:0; }
.lb-caption { margin-top:0; }
.lb-actions { margin-top:0; }
/* tables (trips + lifelist) */
.tbl { width:100%; border-collapse:collapse; font-size:.9rem; }
.tbl th {
  text-align:left; font-size:.74rem; text-transform:uppercase; letter-spacing:.05em;
  color:var(--muted); padding:8px 14px; border-bottom:1px solid var(--border);
}
.tbl td { padding:11px 14px; border-bottom:1px solid var(--border); vertical-align:middle; }
.tbl a { color:var(--fg); text-decoration:none; font-weight:600; }
.tbl a:hover { color:var(--bar); }
.tbl .muted { color:var(--muted); font-size:.8rem; }
.prog { display:inline-block; width:130px; height:8px; background:var(--bar-track); border-radius:999px; overflow:hidden; vertical-align:middle; }
.prog-bar { display:block; height:100%; background:var(--lifer-fg); }
.prog-n { margin-left:8px; font-size:.8rem; color:var(--muted); font-variant-numeric:tabular-nums; }
.life-thumb { width:52px; height:52px; object-fit:cover; border-radius:8px; background:var(--thumb-bg); }
.toggle { display:inline-flex; align-items:center; gap:6px; font-size:.82rem; color:var(--fg); cursor:pointer; }
.counts-prog { font-size:.82rem; font-weight:600; }
.counts a.on { border-color:var(--bar); }
.group.is-hidden, section.is-hidden { display:none; }
.badge.new { background:var(--lifer-bg); color:var(--lifer-fg); border-color:var(--lifer-bd); }
/* lifelist expanded sightings (task 09) */
.life-row { cursor:pointer; }
.life-row:hover td { background:var(--bar-track); }
.life-exp td { background:var(--bg); padding:12px 14px 16px; }
.life-sights { display:flex; flex-wrap:wrap; gap:10px; }
.life-sight {
  display:flex; gap:10px; align-items:center; background:var(--card);
  border:1px solid var(--border); border-radius:10px; padding:8px 10px; min-width:230px;
}
.life-sight img { width:64px; height:64px; object-fit:cover; border-radius:8px; background:var(--thumb-bg); }
.life-sight .d { font-size:.82rem; font-weight:700; }
.life-sight .p { font-size:.76rem; color:var(--muted); }
"""
