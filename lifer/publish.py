"""Public lifelist export for the website.

Writes ``data/birds.json`` and the photo set under ``static/birds/`` into a
Hugo website repo. The public output is deliberately minimal: common name,
scientific name, IOC taxonomic rank, first + most recent sighting, the sighting
count, and one photo per sighting (the trip's best confirmed frame) for the 6
most recent sightings of species that have confirmed encounters. It never
publishes sighting notes/DETAILS, and places stay at the lifelist/region
granularity (city or county), never finer.

Only two things in the site are touched: ``data/birds.json`` and
``static/birds/``. Re-running replaces the generated files and deletes photos
of species that are no longer confirmed (idempotent).
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .boxes import get_box, representative_frame
from .extract import load_photo
from .home import lifer_home, source_folder_for_trip
from .web import (_init_db, aves_taxa, build_sightings, _summarize)

# Source photos live on the read-only BIRDING share (wherever the run's
# meta.json points); publish reads the full-frame image for cropping.
# Publish only reads.

LONG_EDGE = 1600     # px, photo long edge
THUMB_EDGE = 400     # px, thumbnail long edge
JPEG_Q = 85
THUMB_Q = 80


def _slug(name: str) -> str:
    """Filesystem/URL-safe slug for a species common name."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "species"


def _edited(folder: str, frame: str) -> Path | None:
    """A user-edited JPEG for this frame, if one was dropped in edits/."""
    stem = Path(frame).stem
    for root in (lifer_home() / "edits" / folder, lifer_home() / "edits"):
        for ext in ("jpg", "jpeg"):
            p = root / f"{stem}.{ext}"
            if p.is_file():
                return p
    return None


def _crop_breathing(image: Image.Image, det) -> Image.Image:
    """Crop around the detection so the bird is ~1/3 of the crop width.

    The crop is 3:2 (landscape) and centred on the bird, then clamped to the
    frame. A 3:2 sensor is already landscape, so a full-frame clamp keeps the
    ratio; a taller-than-wide bird only widens the crop.
    """
    W, H = image.size
    x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return image

    cw = 3.0 * w                      # bird is ~1/3 of the width
    cw = max(cw, 1.5 * h)             # keep the whole bird in frame vertically
    ch = cw * (2.0 / 3.0)             # 3:2

    cw = min(cw, float(W))
    ch = min(ch, float(H))

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x0 = max(0.0, min(cx - cw / 2.0, W - cw))
    y0 = max(0.0, min(cy - ch / 2.0, H - ch))

    box = (int(round(x0)), int(round(y0)),
           int(round(x0 + cw)), int(round(y0 + ch)))
    return image.crop(box)


def _downscale(image: Image.Image, long_edge: int) -> Image.Image:
    """Scale so the long side is <= long_edge, preserving aspect ratio."""
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) <= long_edge:
        return image
    if w >= h:
        nw = long_edge
        nh = max(1, round(h * long_edge / w))
    else:
        nh = long_edge
        nw = max(1, round(w * long_edge / h))
    return image.resize((nw, nh), Image.LANCZOS)


def _save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    """Save as a clean JPEG: no EXIF/XMP/GPS/serial. A fresh RGB image carries
    no metadata, and we don't pass `exif`/`icc_profile`, so nothing leaks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=quality, optimize=True)


def _render_photo(enc: dict[str, Any], out_full: Path, out_thumb: Path, conn) -> bool:
    """Write the full + thumbnail for one confirmed encounter. Returns False
    if the source (edited file or NEF) is unusable so the caller can try the
    next-best encounter."""
    folder = enc["folder"]
    # Task 12: an owner-chosen representative frame wins over the pipeline's
    # best frame, so publish shows the same photo the owner picked in review.
    frame = representative_frame(conn, folder,
                                 enc.get("frames") or [], enc["best_frame"])
    edited = _edited(folder, frame)
    if edited is not None:
        try:
            image = Image.open(edited).convert("RGB")
        except (OSError, ValueError):
            return False
    else:
        photo = source_folder_for_trip(folder) / frame
        if not photo.is_file():
            return False
        try:
            image = load_photo(photo)
        except (OSError, ValueError, RuntimeError):
            return False
        # Owner crop override wins; otherwise the detection box.
        box = get_box(folder, frame, conn)
        if box is None:
            return False
        image = _crop_breathing(image, box)

    _save_jpeg(_downscale(image, LONG_EDGE), out_full, JPEG_Q)
    _save_jpeg(_downscale(image, THUMB_EDGE), out_thumb, THUMB_Q)
    return True


def publish(site: Path, db_path: Path) -> dict:
    """Write data/birds.json + static/birds/ into the website repo.

    `site` is the website repo root; `db_path` is the lifer SQLite DB. Only
    data/birds.json and static/birds/ are created or modified. Idempotent: a
    re-run replaces the file and removes photos of species no longer confirmed.
    """
    site = site.resolve()
    data_dir = site / "data"
    static_birds = site / "static" / "birds"

    db_path = db_path.expanduser()
    if not db_path.is_absolute():       # same convention as `serve`
        db_path = lifer_home() / db_path
    db_path = db_path.resolve()
    _init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        recs = build_sightings(conn)
        rank = {tid: i for i, (tid, _c, _s) in enumerate(aves_taxa())}

        species: list[dict] = []
        slug_counts: dict[str, int] = {}
        kept_dirs: set[str] = set()

        for rec in recs.values():
            if not rec["sightings"]:
                continue
            slug = _slug(rec["common"])
            slug_counts[slug] = slug_counts.get(slug, 0) + 1
            if slug_counts[slug] > 1:      # collision: make it unique
                slug = f"{slug}-{slug_counts[slug]}"
            kept_dirs.add(slug)
            sdir = static_birds / slug

            # One photo per sighting (that trip's best confirmed frame, falling
            # back through the trip's other frames), at most the 6 most recent
            # sightings. Files are numbered in the order photos render, so a
            # re-run stays idempotent within a species.
            n = 0
            sightings_public: list[dict] = []
            for s in rec["sightings"][:6]:
                if s["enc"] is None:                     # imported ODS row
                    sightings_public.append(
                        {"date": s["date"], "place": s["place"], "photo": None})
                    continue
                photo = None
                for enc in s["encs"]:                    # best first, then fall back
                    cand = n + 1
                    if _render_photo(enc, sdir / f"{cand}.jpg", sdir / f"{cand}-thumb.jpg", conn):
                        n = cand
                        photo = {"src": f"/birds/{slug}/{cand}.jpg",
                                 "thumb": f"/birds/{slug}/{cand}-thumb.jpg",
                                 "date": s["date"]}
                        break
                sightings_public.append(
                    {"date": s["date"], "place": s["place"], "photo": photo})

            # dirty photos from a previous run, beyond this run's count
            for fn in range(n + 1, 7):
                (sdir / f"{fn}.jpg").unlink(missing_ok=True)
                (sdir / f"{fn}-thumb.jpg").unlink(missing_ok=True)

            first, _latest, _latest_photo = _summarize(rec)
            latest_public = sightings_public[0]
            latest_photo_public = next(
                (x for x in sightings_public if x["photo"]), None)

            species.append({
                "slug": slug,
                "common": rec["common"],
                "scientific": rec["scientific"],
                "rank": rank.get(rec["taxon_id"]) if rec["taxon_id"] is not None else None,
                "count": len(rec["sightings"]),
                "first": {"date": first["date"], "place": first["place"]},
                "latest": {"date": latest_public["date"],
                           "place": latest_public["place"],
                           "photo": latest_public["photo"]},
                "latest_photo": latest_photo_public,
                "sightings": sightings_public,
            })

        # top-level idempotence: drop static/birds/<slug> dirs for species that
        # are no longer in the lifelist
        if static_birds.is_dir():
            for d in static_birds.iterdir():
                if d.is_dir() and d.name not in kept_dirs:
                    shutil.rmtree(d, ignore_errors=True)

        payload = {
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "species_count": len(species),
            "with_photos": sum(1 for s in species
                               if any(x["photo"] for x in s["sightings"])),
            "species": species,
        }
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "birds.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return {
            "species_count": len(species),
            "with_photos": payload["with_photos"],
            "photos": sum(1 for s in species
                          for x in s["sightings"] if x["photo"]),
        }
    finally:
        conn.close()
