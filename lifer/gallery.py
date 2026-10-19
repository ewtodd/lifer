"""Public finished-photo gallery for the website (task 27).

Scan darktable XMP sidecars (``xmp:Rating``) under the configured edit
roots (``[gallery] roots`` in LIFER_HOME/config.toml, default ``~/BIRDING``
read-only and ``~/Pictures``) and publish the owner's finished 5-star photos
into a Hugo website repo.

For each 5-star sidecar we find the finished export in a
``darktable_exported/`` folder beside the raw (``<stem>.jpg`` for the base
version, ``<stem>_01.jpg`` etc. for darktable duplicates, newest mtime wins),
then match the photo to the confirmed decision whose encounter contains that
frame - by raw filename AND capture time (EXIF), so the same filename in two
trips/folders cannot collide. Photos that are rated but not exported or not
(yet) confirmed are skipped and counted.

Output: ``data/gallery.json`` (species, scientific, date, place, file, thumb,
width, height; newest first) plus a 2400px watermarked JPEG (q88, all metadata
stripped) and a 500px un-watermarked thumbnail in ``static/birds/gallery/``.
Idempotent: re-running replaces the files and removes photos no longer rated 5.
Only ``data/gallery.json`` and ``static/birds/gallery/`` in the site are
written.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import rawpy
from PIL import Image

from .config import load_config
from .home import lifer_home, source_folder_for_trip
from .watermark import Watermark, apply_watermark
from .web import _init_db, load_candidates, region_name

RATING = 5                      # xmp:Rating value that puts a photo in the gallery
LONG_EDGE = 2400                # px, photo long edge
JPEG_Q = 88
THUMB_EDGE = 500                # px, thumbnail long edge
THUMB_Q = 80
SIDECAR_RE = re.compile(r"\.NEF\.xmp$")
RATING_ATTR = re.compile(r'xmp:Rating="(\d+)"')
RATING_EL = re.compile(r"<xmp:Rating>(\d+)</xmp:Rating>")
DERIVED_FROM = re.compile(r'xmpMM:DerivedFrom="([^"]+)"')

# A darktable duplicate exports under the *base* stem only when the owner kept
# the same version suffix, so a base sidecar may also have a duplicate export.
EXPORT_CANDIDATES = lambda stem: (f"{stem}.jpg", f"{stem}_*.jpg")


def _roots(cfg: dict[str, Any]) -> list[Path]:
    """Edit roots for the scan; ~/BIRDING (read-only) and ~/Pictures by default."""
    raw = (cfg.get("gallery") or {}).get("roots")
    if raw is None:
        raw = ["~/BIRDING", "~/Pictures"]
    if isinstance(raw, str):
        raw = [raw]
    return [Path(r).expanduser().resolve() for r in raw]


def _sidecar_rating(path: Path) -> int | None:
    """The xmp:Rating value from a sidecar, or None when absent/unreadable."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = RATING_ATTR.search(text) or RATING_EL.search(text)
    return int(m.group(1)) if m else None


def _base_raw(sidecar: Path) -> str:
    """The raw filename a sidecar edits.

    ``xmpMM:DerivedFrom`` names the base raw for a darktable duplicate
    (``_DSC8819_01.NEF.xmp`` -> ``_DSC8819.NEF``); otherwise the sidecar's own
    name minus the ``.xmp`` suffix is the raw it sits beside.
    """
    try:
        text = sidecar.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    m = DERIVED_FROM.search(text)
    if m:
        return m.group(1)
    return sidecar.name[:-len(".xmp")]


def _stem(sidecar: Path) -> str:
    """The version stem: ``_DSC8819`` or ``_DSC8819_01`` (for the export file)."""
    return sidecar.name[:-len(".NEF.xmp")]


def find_export(sidecar: Path) -> Path | None:
    """The owner's finished export for a sidecar, or None.

    Look in ``darktable_exported/`` beside the raw for ``<stem>.jpg`` (base
    version) and ``<stem>_*.jpg`` (duplicates); the newest mtime wins.
    """
    export_dir = sidecar.parent / "darktable_exported"
    if not export_dir.is_dir():
        return None
    stem = _stem(sidecar)
    cands = list(export_dir.glob(EXPORT_CANDIDATES(stem)[0]))
    cands += list(export_dir.glob(EXPORT_CANDIDATES(stem)[1]))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _capture_time(path: Path) -> datetime | None:
    """EXIF capture time (DateTimeOriginal) of a raw, or None when unreadable."""
    try:
        with rawpy.imread(str(path)) as raw:
            ts = raw.other.timestamp
    except Exception:
        return None
    if ts is None:
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return None
    return ts


def _frame_index(conn: sqlite3.Connection, needed: set[str]) -> dict[str, list[tuple[datetime, dict]]]:
    """Confirmed decisions that contain any needed raw, keyed by raw filename.

    Each value is ``[(capture_time, decision), ...]`` so a sidecar can pick the
    decision whose capture time matches its own raw. ``decision`` carries the
    row plus the encounter's time_start date.
    """
    idx: dict[str, list[tuple[datetime, dict[str, Any]]]] = {}
    if not needed:
        return idx
    date_cache: dict[str, dict[str, Any]] = {}
    for d in conn.execute(
            "SELECT folder, enc_id, species, scientific, frames FROM decisions "
            "WHERE action IN ('confirm','change')"):
        frames = json.loads(d["frames"] or "[]")
        date = _encounter_date(d["folder"], d["enc_id"], date_cache)
        rec: dict[str, Any] = {
            "folder": d["folder"], "enc_id": d["enc_id"],
            "species": d["species"], "scientific": d["scientific"],
            "date": date,
        }
        for fr in frames:
            if fr not in needed:
                continue
            src = source_folder_for_trip(d["folder"]) / fr
            t = _capture_time(src)
            if t is None:
                continue
            idx.setdefault(fr, []).append((t, rec))
    return idx


def _encounter_date(folder: str, enc_id: str, cache: dict[str, dict[str, Any]]) -> str:
    """The date of an encounter (its first frame), from candidates.csv."""
    cands = cache.get(folder)
    if cands is None:
        try:
            cands = {c["enc_id"]: c for c in load_candidates(folder)}
        except (OSError, ValueError):
            cands = {}
        cache[folder] = cands
    c = cands.get(enc_id)
    return (c.get("time_start") or "")[:10] if c else ""


def _full_frame_size(src: Path) -> tuple[int, int] | None:
    """Size of the embedded full-frame JPEG (the box coordinate space)."""
    try:
        from .extract import load_frame
        with Image.open(io.BytesIO(load_frame(src).jpeg)) as im:
            return im.size
    except Exception:
        return None


def _bird_in_export(folder: str, frame: str, src: Path, out_size: tuple[int, int],
                    conn: sqlite3.Connection) -> tuple[float, float, float, float] | None:
    """The bird box mapped into the export's pixel coords, or None.

    The owner's export is already a finished crop. When we can determine that
    the export is the full frame (its aspect ratio matches the source), we map
    the detection box by scale; otherwise we cannot know where the crop sits
    and we leave placement to the watermark's calmest-corner fallback (None).
    """
    box = _get_box(conn, folder, frame)
    if box is None:
        return None
    full = _full_frame_size(src)
    if full is None or full[0] == 0:
        return None
    sx = out_size[0] / full[0]
    sy = out_size[1] / full[1]
    if abs(sx - sy) > 0.02 * sx:
        return None
    return (box.x1 * sx, box.y1 * sy, box.x2 * sx, box.y2 * sy)


def _get_box(conn: sqlite3.Connection, folder: str, frame: str):
    from .boxes import get_box
    return get_box(folder, frame, conn)


def _downscale(image: Image.Image, long_edge: int) -> Image.Image:
    w, h = image.size
    scale = long_edge / max(w, h)
    if scale >= 1:
        return image
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                        Image.LANCZOS)


def _save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save via bytes so we control every byte; no EXIF/comment metadata is
    # carried across (the photo metadata is stripped deliberately).
    image.save(path, "JPEG", quality=quality, optimize=True)


def _render(export: Path, folder: str, frame: str, out_full: Path, out_thumb: Path,
            conn: sqlite3.Connection) -> tuple[int, int]:
    """Write the watermarked full photo and its thumbnail."""
    img = Image.open(export).convert("RGB")
    full_img = _downscale(img, LONG_EDGE)
    src = source_folder_for_trip(folder) / frame
    bird = _bird_in_export(folder, frame, src, full_img.size, conn)
    full_img = apply_watermark(full_img, bird, Watermark.from_config())
    _save_jpeg(full_img, out_full, JPEG_Q)
    _save_jpeg(_downscale(img, THUMB_EDGE), out_thumb, THUMB_Q)
    return full_img.size


def publish_gallery(site: Path, db_path: Path) -> dict:
    """Write data/gallery.json + static/birds/gallery/ into the website repo.

    Returns ``{in_gallery, rated_not_exported, rated_not_confirmed}``.
    """
    site = site.resolve()
    data_dir = site / "data"
    gal_dir = site / "static" / "birds" / "gallery"

    db_path = db_path.expanduser()
    if not db_path.is_absolute():       # same convention as `serve`/`publish`
        db_path = lifer_home() / db_path
    db_path = db_path.resolve()
    _init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    cfg = load_config()
    roots = _roots(cfg)
    sidecars = [p for root in roots if root.is_dir()
                for p in sorted(root.rglob("*")) if SIDECAR_RE.search(p.name)]

    rated_not_exported = 0
    rated_not_confirmed = 0
    entries: list[dict[str, Any]] = []
    out_files: set[str] = set()

    try:
        # Build the decision index only for raw names a 5-star sidecar needs.
        needed: set[str] = set()
        for sc in sidecars:
            if _sidecar_rating(sc) == RATING:
                needed.add(_base_raw(sc))
        idx = _frame_index(conn, needed)

        for sc in sidecars:
            if _sidecar_rating(sc) != RATING:
                continue
            raw_name = _base_raw(sc)
            export = find_export(sc)
            if export is None:
                rated_not_exported += 1
                continue
            raw_path = sc.parent / raw_name
            spot = _capture_time(raw_path)
            match = None
            if spot is not None:
                for dt, d in idx.get(raw_name, []):
                    if dt == spot:
                        match = d
                        break
            if match is None:
                rated_not_confirmed += 1
                continue

            folder, frame = match["folder"], raw_name
            slug = f"{folder}_{_stem(sc)}"
            out_full = gal_dir / f"{slug}.jpg"
            out_thumb = gal_dir / f"{slug}-thumb.jpg"
            width, height = _render(export, folder, frame, out_full, out_thumb, conn)
            entries.append({
                "species": match["species"],
                "scientific": match["scientific"],
                "date": match["date"] or (spot.date().isoformat() if spot else ""),
                "place": region_name(folder),
                "file": f"/birds/gallery/{slug}.jpg",
                "thumb": f"/birds/gallery/{slug}-thumb.jpg",
                "width": width, "height": height,
            })
            out_files.add(out_full.name)
            out_files.add(out_thumb.name)

        # Idempotent: drop gallery files from a previous run that are no longer
        # produced (photos no longer rated 5-star, or no longer confirmed).
        if gal_dir.is_dir():
            for p in gal_dir.iterdir():
                if p.is_file() and p.name not in out_files:
                    p.unlink(missing_ok=True)

        entries.sort(key=lambda e: (e["date"], e["species"]), reverse=True)
        payload = {
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "photos": entries,
        }
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "gallery.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

        return {
            "in_gallery": len(entries),
            "rated_not_exported": rated_not_exported,
            "rated_not_confirmed": rated_not_confirmed,
        }
    finally:
        conn.close()
