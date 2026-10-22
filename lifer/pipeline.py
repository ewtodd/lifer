"""Pipeline orchestration: photos -> detect -> encounters -> BioCLIP -> outputs.

A "photo" is an NEF (its embedded full-size JPEG) or, in a folder with no
NEFs, a standalone JPEG. Slow/networked work is cached under cache/ keyed per
image and per API result, so re-runs skip finished work. Nothing is ever
written under the source folder.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import sys
import time
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from .bioclip import BioCLIP, label_text
from .birds import (MIN_LONG_SIDE, BirdDet, bird_boxes, bird_frames,
                    average_probs, group_extra_birds, moved_frames, species_flags)
from .boxes import Box, added_birds, get_box, open_db, removed_frames
from .bursts import (Encounter, FrameInfo, best_frame_key,
                     group_into_encounters)
from .detect import (BirdDetector, Detection, FrameDetections, sharpness,
                     square_box, square_crop, usable_crop)
from .extract import (Frame, jpeg_to_image, load_frame, load_frame_jpg,
                      load_photo)
from .home import effective_trip_id, find_regions_toml, lifer_home
from .lifelist import (LifeEntry, lifer_set, match_lifelist, parse_lifelist,
                       write_lifelist_csv)
from .outputs import Candidate, save_crop, save_frame_preview, write_candidates_csv, write_review_html
from .region import (lifelist_place_counts, normalized, region_display_names,
                     species_counts)
from .subject import CENTRE, iou, read_af_points
from .taxonomy import aves_species

LIFELIST = Path.home() / "BIRDING" / "LIFELIST.ods"


def resolve_region(trip_id: str) -> dict:
    """First regions.toml entry whose glob matches the trip ID (a nested
    outing's ID carries its parent folder, so globs can match either)."""
    cfg = tomllib.loads(find_regions_toml().read_text())
    for region in cfg.get("region", []):
        import fnmatch
        if fnmatch.fnmatch(trip_id, region["glob"]):
            return region
    raise SystemExit(
        f"no region in regions.toml matches trip {trip_id!r}; "
        "add a [[region]] entry with glob/name/place_id"
    )


@dataclass
class FrameRecord:
    path: Path
    name: str
    timestamp: datetime
    dets: FrameDetections
    sharpness: float  # of best bird crop; -1 when no detection
    _jpeg: bytes | None = None  # populated lazily, not cached


def _frame_cache_path(cache_dir: Path, nef: Path) -> Path:
    return cache_dir / f"{nef.stem}.json"


class AFPoints:
    """A folder's AF points, read with one exiftool call on first use (a
    fully cached re-run never needs them)."""

    def __init__(self, folder: Path):
        self.folder = folder
        self._points: dict[str, tuple[float, float]] | None = None

    def get(self, name: str) -> tuple[float, float]:
        if self._points is None:
            self._points = read_af_points(self.folder)
        return self._points.get(name, CENTRE)


def _cache_key(detector: BirdDetector) -> dict:
    return {
        "yolo": detector.weights_key,
        "imgsz": detector.imgsz,
        "conf": detector.conf,
        "recipe": detector.recipe,
    }


def _read_photo(photo: Path) -> Frame:
    return load_frame(photo) if photo.suffix.lower() == ".nef" else load_frame_jpg(photo)


def cached_record(nef: Path, detector: BirdDetector,
                  cache_dir: Path) -> FrameRecord | None:
    """The frame's record from the per-frame cache, or None when missing,
    stale (file or detector recipe changed) or corrupt."""
    cache = _frame_cache_path(cache_dir, nef)
    key = _cache_key(detector)
    st = nef.stat()
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if (data.get("name") == nef.name and data.get("mtime") == st.st_mtime
                    and data.get("size") == st.st_size and data.get("key") == key):
                dets = data["dets"]
                fd = FrameDetections(
                    frame_name=nef.name,
                    dets=[Detection(d[0], d[1], d[2], d[3], d[4]) for d in dets],
                    af=tuple(data.get("af") or CENTRE),
                    width=data.get("width", 0), height=data.get("height", 0))
                return FrameRecord(
                    path=nef, name=nef.name,
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    dets=fd, sharpness=data["sharpness"],
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupt cache entry: recompute
    return None


def _prefetch_frames(photos: list[Path], detector: BirdDetector,
                     cache_dir: Path, ahead: int = 8):
    """Yield (photo, frame) with the frames of uncached photos read on worker
    threads a few ahead: reading a NEF off the NFS share (~0.3 s) costs more
    than detecting it (~0.07 s on the GPU). `frame` is None for cached photos."""
    futures: dict[int, Future] = {}
    queued = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, photo in enumerate(photos):
            while queued < min(i + ahead, len(photos)):
                if cached_record(photos[queued], detector, cache_dir) is None:
                    futures[queued] = pool.submit(_read_photo, photos[queued])
                queued += 1
            fut = futures.pop(i, None)
            yield photo, (fut.result() if fut else None)


def load_frame_with_dets(
    nef: Path, detector: BirdDetector, cache_dir: Path,
    af_points: AFPoints | None = None, frame: Frame | None = None,
) -> FrameRecord:
    """Extract + detect one NEF, using the per-frame cache when valid.
    `frame` is the already-read photo, when the caller prefetched it."""
    rec = cached_record(nef, detector, cache_dir)
    if rec is not None:
        return rec
    cache = _frame_cache_path(cache_dir, nef)
    key = _cache_key(detector)
    st = nef.stat()
    if frame is None:
        frame = _read_photo(nef)
    af = af_points.get(nef.name) if af_points is not None else CENTRE
    gpu = detector.decode(frame.jpeg)
    fd = detector.detect(gpu, af)
    fd.frame_name = nef.name
    if fd.best is not None:
        # cut the sharpness crop on the GPU: no CPU decode of the full frame
        l, t, r, b = square_box((fd.width, fd.height), fd.best)
        crop = Image.fromarray(
            gpu[:, t:b, l:r].permute(1, 2, 0).contiguous().cpu().numpy())
        sh = sharpness(crop)
    else:
        sh = -1.0
    del gpu
    rec = FrameRecord(
        path=nef, name=nef.name, timestamp=frame.timestamp,
        dets=fd, sharpness=sh, _jpeg=frame.jpeg,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "name": nef.name, "mtime": st.st_mtime, "size": st.st_size, "key": key,
        "timestamp": frame.timestamp.isoformat(),
        "dets": [[d.x1, d.y1, d.x2, d.y2, d.conf] for d in fd.dets],
        "af": [af[0], af[1]],
        "width": fd.width, "height": fd.height,
        "sharpness": sh,
    }))
    return rec


def _crop_cache_path(cache_dir: Path, nef: Path) -> Path:
    return cache_dir / "bioclip" / f"{nef.stem}.json"


def classify_encounter(
    enc: Encounter, records: dict[str, FrameRecord], classes: dict,
    bioclip: BioCLIP, txt: torch.Tensor, cache_dir: Path, class_hash: str,
    temperature: float, top_k: int = 3, folder: str = "", conn=None
) -> list[float]:
    """Average BioCLIP probabilities over the sharpest detected crops.

    `txt` is the precomputed (n_labels, dim) label matrix. Returns the
    probability vector over `classes` (ordered by taxon id).
    Per-frame crop probabilities are cached (keyed by class set AND
    temperature, since both change the numbers stored). A frame with a manual
    crop override uses the override box and is never read from / written to
    the crop cache, so Undo restores the detector result.
    """
    best = enc.best_frame
    sharpest = sorted((f for f in enc.frames if f.n_dets > 0),
                      key=lambda f: f.sharpness, reverse=True)[:top_k]
    order = sorted(classes)
    vectors = []
    crops = []
    skip_cache = set()
    for f in sharpest:
        rec = records[f.name]
        box = get_box(folder, f.name, conn, dets=rec.dets.dets,
                      af=rec.dets.af, size=(rec.dets.width, rec.dets.height))
        if box is None:
            continue
        cf = _crop_cache_path(cache_dir, rec.path)
        if box.source != "override" and cf.exists():
            try:
                data = json.loads(cf.read_text())
                if (data.get("class_hash") == class_hash
                        and data.get("temp") == temperature):
                    vectors.append({tid: p for tid, p in data["probs"]})
                    continue
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        if box.source == "override":
            skip_cache.add(f.name)
        image = jpeg_to_image(rec._jpeg) if rec._jpeg else load_photo(rec.path)
        crop = square_crop(image, box)
        if not usable_crop(crop):
            # blank/black frame or zero-area box: nothing to classify
            continue
        crops.append((f.name, crop))

    if crops:
        probs = bioclip.probabilities([c for _, c in crops], txt, temperature)
        for (name, _), row in zip(crops, probs):
            rec = records[name]
            vec = {tid: float(p) for tid, p in zip(order, row)}
            vectors.append(vec)
            if name in skip_cache:
                continue  # override crop is used for the average but not cached
            cf = _crop_cache_path(cache_dir, rec.path)
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps({
                "class_hash": class_hash,
                "temp": temperature,
                "probs": [[tid, p] for tid, p in vec.items()],
            }))

    if not vectors:
        return [0.0] * len(classes)
    n = len(vectors)
    return [sum(v.get(t, 0.0) for v in vectors) / n for t in order]


def _write_bird_images(rec: FrameRecord, box, out_dir: Path, enc_id: str) -> None:
    """The review page's images of one bird, cut from `rec`'s photo: the card
    crop, the full frame with its box, and a box-less full frame for the crop
    editor (same scale)."""
    image = jpeg_to_image(rec._jpeg) if rec._jpeg else load_photo(rec.path)
    save_crop(square_crop(image, box), out_dir / "crops" / f"{enc_id}.jpg")
    save_frame_preview(image, box, out_dir / "frames" / f"{enc_id}.jpg")
    save_frame_preview(image, None, out_dir / "frames" / "src" / f"{enc_id}.jpg")


def _det_cache_path(cache_dir: Path, nef: Path) -> Path:
    return cache_dir / "bioclip_dets" / f"{nef.stem}.json"


def _box_key(box) -> str:
    """Cache key of a detection box: rounded, because a re-detection can
    shift a box by a fraction of a pixel without changing the bird."""
    return ",".join(str(int(round(v))) for v in box)


def classify_dets(frame_boxes: dict[str, list[tuple[tuple, float, bool]]],
                  records: dict[str, FrameRecord], bioclip: BioCLIP,
                  txt: torch.Tensor, order: list[int], cache_dir: Path,
                  class_hash: str, temperature: float) -> list[dict]:
    """BioCLIP on every given (box, conf, added) of each frame, one crop per
    box. Per-detection results are cached beside the per-frame crop cache,
    keyed by class set + temperature + the rounded box, so re-runs only read
    the photo when a box is new. A crop that is blank is cached as such and
    left out of the result."""
    out: list[dict] = []
    for name, items in frame_boxes.items():
        rec = records[name]
        cf = _det_cache_path(cache_dir, rec.path)
        cached: dict[str, dict] = {}
        if cf.exists():
            try:
                data = json.loads(cf.read_text())
                if (data.get("class_hash") == class_hash
                        and data.get("temp") == temperature):
                    cached = data["dets"]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        todo = []
        for box, conf, added in items:
            hit = cached.get(_box_key(box))
            if hit is None:
                todo.append((box, conf, added))
            elif hit["p"]:
                out.append({"frame": name, "box": box, "conf": conf,
                            "added": added, "sharp": hit["sharp"],
                            "raw": {t: p for t, p in hit["p"]}})
        if not todo:
            continue
        image = jpeg_to_image(rec._jpeg) if rec._jpeg else load_photo(rec.path)
        crops = []
        for box, conf, added in todo:
            crop = square_crop(image, Box(*box))
            if usable_crop(crop):
                crops.append((box, conf, added, crop))
            else:
                cached[_box_key(box)] = {"p": [], "sharp": 0.0}
        if crops:
            probs = bioclip.probabilities([c[3] for c in crops], txt, temperature)
            for (box, conf, added, crop), row in zip(crops, probs):
                sharp = round(sharpness(crop), 1)
                vec = {t: float(p) for t, p in zip(order, row)}
                # drop the tail: ~1000 near-zero entries per box add nothing
                cached[_box_key(box)] = {
                    "p": [[t, round(p, 6)] for t, p in vec.items() if p >= 1e-5],
                    "sharp": sharp}
                out.append({"frame": name, "box": box, "conf": conf,
                            "added": added, "sharp": sharp, "raw": vec})
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps({"class_hash": class_hash, "temp": temperature,
                                  "dets": cached}))
    return out


def burst_birds(enc: Encounter, records: dict[str, FrameRecord], folder: str,
                added: list[tuple[str, list[float]]], conn, bioclip: BioCLIP,
                txt: torch.Tensor, order: list[int], season: dict,
                alpha: float, cache_dir: Path, class_hash: str,
                temperature: float, primary_adj: dict[int, float]):
    """The extra birds of a burst (see birds.py): every sizeable detection
    plus the owner's added boxes is classified on its own, and the ones that
    are clearly another species than the primary become birds."""
    frame_order = [f.name for f in enc.frames]
    wanted: dict[str, list[tuple[tuple, float, bool]]] = {}
    subject_box: dict[str, Box | None] = {}
    for name in frame_order:
        rec = records[name]
        # the primary's box in this frame (override, else the AF-point subject)
        subject_box[name] = get_box(folder, name, conn, dets=rec.dets.dets,
                                    af=rec.dets.af,
                                    size=(rec.dets.width, rec.dets.height))
        mine = [(tuple(b), 1.0, True) for fr, b in added if fr == name]
        for d in rec.dets.dets:
            box = (d.x1, d.y1, d.x2, d.y2)
            if max(d.x2 - d.x1, d.y2 - d.y1) < MIN_LONG_SIDE:
                continue
            # an owner-drawn box stands in for the detector box it overlaps
            if any(iou(box, m[0]) >= 0.5 for m in mine):
                continue
            wanted.setdefault(name, []).append((box, d.conf, False))
        if mine:
            wanted.setdefault(name, []).extend(mine)
    found = classify_dets(wanted, records, bioclip, txt, order, cache_dir,
                          class_hash, temperature)
    dets = []
    for r in found:
        sb = subject_box[r["frame"]]
        adj = {t: r["raw"].get(t, 0.0) * season.get(t, 0.0) ** alpha for t in order}
        total = sum(adj.values()) or 1.0
        dets.append(BirdDet(
            frame=r["frame"], box=tuple(r["box"]), conf=r["conf"], sharp=r["sharp"],
            raw=r["raw"], adj={t: v / total for t, v in adj.items()},
            subject=(not r["added"] and sb is not None
                     and iou(r["box"], (sb.x1, sb.y1, sb.x2, sb.y2)) >= 0.5),
            added=r["added"]))
    return group_extra_birds(dets, primary_adj, frame_order)


def run_folder(folder_arg: str | Path, root: str | Path | None = None,
               gap_seconds: float = 5.0,
               alpha: float = 0.3,
               yolo_weights: str = "yolo11m.pt", imgsz: int = 2560,
               conf: float = 0.25, temperature: float = 3.0) -> Path:
    t_start = time.time()
    folder = Path(folder_arg).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")
    home = lifer_home()
    name = effective_trip_id(folder, root)
    out_dir = home / "runs" / name
    cache_dir = home / "cache" / name
    inat_cache = home / "cache" / "inat"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Where the photos live, for everything that re-opens them later (review
    # app, re-ID, publish, develop): nested folders are not at ~/BIRDING/<name>.
    (out_dir / "meta.json").write_text(json.dumps(
        {"trip_id": name, "folder": str(folder)}, indent=2) + "\n")
    timings: dict[str, float] = {}
    db_conn = open_db()  # reads manual crop overrides so re-runs keep fixes

    if not torch.cuda.is_available():
        raise SystemExit("torch cannot see a CUDA GPU; refusing to run (see AGENTS.md)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- region ---------------------------------------------------------
    t0 = time.time()
    region = resolve_region(name)
    place_id = int(region["place_id"])
    region_name, region_short = region_display_names(region, inat_cache)
    region_name = region_name or name  # no name anywhere: folder name
    print(f"region: {region_name} (place_id={place_id})")

    # --- lifelist -------------------------------------------------------
    t0 = time.time()
    taxa = aves_species(inat_cache)
    entries = parse_lifelist(LIFELIST)
    # each entry's own location settles a near-tied fuzzy match (American not
    # European Herring Gull in Wisconsin); lifelist.csv is shared by every
    # trip, so it must not depend on the region being run
    place_counts, all_counts = lifelist_place_counts(inat_cache)
    matched = match_lifelist(entries, taxa, counts_by_place=place_counts,
                             default_counts=all_counts)
    write_lifelist_csv(matched, home / "data" / "lifelist.csv")
    lifers = lifer_set(matched)
    n_unmatched = sum(1 for e in matched if e.taxon is None)
    timings["lifelist"] = time.time() - t0
    print(f"lifelist: {len(matched)} entries, {n_unmatched} unmatched "
          f"({len(lifers)} distinct taxa) [{timings['lifelist']:.1f}s]")

    # --- frames: extract + detect ---------------------------------------
    t0 = time.time()
    nefs = sorted(folder.glob("*.NEF"))
    if nefs:
        photos = nefs  # mixed folders use the NEFs, never the sidecar JPGs
    else:
        photos = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    if not photos:
        raise SystemExit(f"no .NEF or .jpg/.jpeg files in {folder}")
    detector = BirdDetector(weights=yolo_weights, imgsz=imgsz, conf=conf)
    records: dict[str, FrameRecord] = {}
    af_points = AFPoints(folder)
    for i, (photo, frame) in enumerate(
            _prefetch_frames(photos, detector, cache_dir), 1):
        rec = load_frame_with_dets(photo, detector, cache_dir, af_points, frame)
        records[rec.name] = rec
        if i % 25 == 0 or i == len(photos):
            print(f"  frames {i}/{len(photos)} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    n_detected = sum(1 for r in records.values() if r.dets.dets)
    timings["detect"] = time.time() - t0
    print(f"detection: {n_detected}/{len(photos)} frames with birds "
          f"[{timings['detect']:.0f}s]")

    # --- encounters ------------------------------------------------------
    t0 = time.time()
    infos = [FrameInfo(name=r.name, timestamp=r.timestamp, n_dets=len(r.dets.dets),
                       best_conf=r.dets.best.conf if r.dets.best else 0.0,
                       sharpness=r.sharpness) for r in records.values()]
    encounters = group_into_encounters(infos, gap_seconds)
    # Frames the owner removed in the review app stay out of their encounter.
    # Dropped after grouping, so the encounters (and their IDs) don't shift.
    removed = removed_frames(db_conn, name)
    for enc in encounters:
        enc.frames = [f for f in enc.frames if f.name not in removed]
    timings["group"] = time.time() - t0
    print(f"encounters: {len(encounters)} (gap {gap_seconds:g}s) "
          f"[{timings['group']:.1f}s]")

    # --- region species list --------------------------------------------
    t0 = time.time()
    month = records[infos[0].name].timestamp.date()
    classes, alltime_counts, seasonal_counts = species_counts(place_id, month, inat_cache)
    season = normalized(seasonal_counts)
    seasonal_total = sum(seasonal_counts.values()) or 1.0
    alltime_total = sum(alltime_counts.values()) or 1.0
    m = month.month
    # month window (m-1 .. m+1, wrapping) as a label, e.g. "Nov\u2013Jan"
    first = m - 1 if m > 1 else 12
    last = m + 1 if m < 12 else 1
    window_label = f"{calendar.month_abbr[first]}\u2013{calendar.month_abbr[last]}"
    # per-month seasonals are cached; other months in the folder fall back
    # to all-time (both demo folders are single-day trips).
    timings["region"] = time.time() - t0
    print(f"region classes: {len(classes)} "
          f"(seasonal {len(seasonal_counts)}) [{timings['region']:.1f}s]")

    # --- BioCLIP ---------------------------------------------------------
    t0 = time.time()
    bioclip = BioCLIP()
    class_hash = hashlib.sha1(
        json.dumps(sorted(t for t in classes)).encode()).hexdigest()[:16]
    labels = [label_text(classes[t].scientific, classes[t].common) for t in sorted(classes)]
    # label text features depend only on the class set: cache them across runs
    label_cache = inat_cache / f"bioclip_labels_{class_hash}.pt"
    if label_cache.exists():
        txt = torch.load(label_cache, map_location="cuda")
        print(f"  labels loaded from cache [{time.time() - t0:.0f}s]", flush=True)
    else:
        print(f"  encoding {len(labels)} labels x 80 templates ...", flush=True)
        txt = bioclip.precompute_labels(labels)
        inat_cache.mkdir(parents=True, exist_ok=True)
        torch.save(txt.cpu(), label_cache)
        print(f"  labels encoded [{time.time() - t0:.0f}s]", flush=True)
    # the owner's hand-drawn birds; each becomes an extra bird of its burst
    added = added_birds(db_conn, name)
    n_extra = 0
    for sub in ("crops", "frames", "frames/src"):  # extra birds are rebuilt below
        for stale in (out_dir / sub).glob("enc_*-*.jpg"):
            stale.unlink()
    cands: list[Candidate] = []
    nodet_encs: list[dict[str, object]] = []  # encounters with no usable detection
    nodet_frames: set[str] = set()  # their frames, for the thumbnail pass below
    order = sorted(classes)
    for i, enc in enumerate(encounters, 1):
        enc_id = f"enc_{i:04d}"
        if not enc.frames:
            continue  # every frame removed by the owner
        best = enc.best_frame
        rec = records[best.name]
        has_det = any(f.n_dets > 0 for f in enc.frames)
        if has_det:
            prob_vec = classify_encounter(enc, records, classes, bioclip, txt, cache_dir,
                                          class_hash, temperature, folder=name,
                                          conn=db_conn)
        else:
            # nothing to classify: no crop is ever fed to BioCLIP
            prob_vec = [0.0] * len(order)
        # all-zeros also when a detected frame's crop was blank/black (the
        # classifier skips unusable crops): treat the encounter as no-detection
        if not has_det or max(prob_vec) <= 0.0:
            # No usable detection: no species and no candidates.csv row. The
            # "No bird detected" section lists the encounter with a full-frame
            # thumbnail (task 23), and the review app's crop editor can turn
            # it into a real encounter by drawing a box.
            nodet_encs.append({
                "enc_id": enc_id,
                "time_start": enc.start.strftime("%Y-%m-%d %H:%M:%S"),
                "time_end": enc.end.strftime("%Y-%m-%d %H:%M:%S"),
                "n_frames": len(enc.frames),
                "best_frame": best.name,
                "frames": [f.name for f in enc.frames],
            })
            nodet_frames.update(f.name for f in enc.frames)
            continue
        p_clip = dict(zip(order, prob_vec))
        p_adj = {t: p_clip[t] * season.get(t, 0.0) ** alpha for t in order}
        adj_total = sum(p_adj.values()) or 1.0
        p_adj = {t: v / adj_total for t, v in p_adj.items()}
        # has_det is True here: no-detection encounters `continue`d above
        flags = species_flags(p_clip, p_adj, classes, lifers, seasonal_counts,
                              alltime_counts, region_name, window_label)

        # Other species in the same burst become birds of their own. A frame
        # whose subject (the bird at the AF point) is one of them leaves the
        # primary's frames: the photographer re-aimed mid-burst.
        extras = burst_birds(enc, records, name, added, db_conn, bioclip, txt,
                             order, season, alpha, cache_dir, class_hash,
                             temperature, p_adj)
        gone = moved_frames(extras)
        kept = [f for f in enc.frames if f.name not in gone] or [best]
        if best.name not in {f.name for f in kept}:
            best = max(kept, key=lambda f: best_frame_key(f.name, f.sharpness))
            rec = records[best.name]
        cands.append(Candidate(
            enc_id=enc_id,
            time_start=kept[0].timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            time_end=kept[-1].timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            n_frames=len(kept),
            best_frame=best.name,
            det_conf=best.best_conf,
            has_det=True,
            frames=[f.name for f in kept],
            **flags,
        ))
        # best crop for the review page (no-detection encounters never get
        # here; they are listed in the "No bird detected" section instead)
        crop_box = get_box(name, best.name, db_conn, dets=rec.dets.dets,
                           af=rec.dets.af, size=(rec.dets.width, rec.dets.height))
        if crop_box is not None:
            _write_bird_images(rec, crop_box, out_dir, enc_id)

        frame_order = [f.name for f in enc.frames]
        for n, bird in enumerate(extras, 1):
            bird_id = f"{enc_id}-{n}"
            b_flags = species_flags(
                average_probs(bird, "raw"), average_probs(bird, "adj"), classes,
                lifers, seasonal_counts, alltime_counts, region_name, window_label)
            b_frames = bird_frames(bird, frame_order)
            stamps = [records[f].timestamp for f in b_frames]
            cands.append(Candidate(
                enc_id=bird_id,
                time_start=min(stamps).strftime("%Y-%m-%d %H:%M:%S"),
                time_end=max(stamps).strftime("%Y-%m-%d %H:%M:%S"),
                n_frames=len(b_frames),
                best_frame=bird.best.frame,
                det_conf=max(d.conf for d in bird.dets),
                has_det=True,
                frames=b_frames,
                bird=n,
                boxes=bird_boxes(bird),
                **b_flags,
            ))
            _write_bird_images(records[bird.best.frame], Box(*bird.best.box),
                               out_dir, bird_id)
            n_extra += 1
        for f in enc.frames:
            if records[f.name].dets.dets:
                records[f.name]._jpeg = None  # the burst is done with its photos
        if i % 10 == 0 or i == len(encounters):
            print(f"  classified {i}/{len(encounters)} encounters "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    timings["bioclip"] = time.time() - t0
    print(f"bioclip: {timings['bioclip']:.0f}s")

    # --- no-detection thumbnails ----------------------------------------
    t0 = time.time()
    no_det: list[dict[str, object]] = []
    for ne in nodet_encs:
        # full-frame thumbnail of the encounter's sharpest (det-less) frame
        r = records[str(ne["best_frame"])]
        image = jpeg_to_image(r._jpeg) if r._jpeg else load_photo(r.path)
        r._jpeg = None
        thumb_rel = f"crops/nodet/{ne['enc_id']}.jpg"
        save_frame_preview(image, None, out_dir / thumb_rel, max_side=320)
        no_det.append({
            "id": ne["enc_id"], "time": str(ne["time_start"])[11:16],
            "n_frames": ne["n_frames"], "best_frame": ne["best_frame"],
            "thumb": thumb_rel,
        })
    # individual no-detection frames inside an encounter that has detected
    # frames (the encounter itself is classified on its detected frames)
    for r in records.values():
        if r.dets.dets or r.name in nodet_frames or r.name in removed:
            continue
        image = jpeg_to_image(r._jpeg) if r._jpeg else load_photo(r.path)
        thumb_rel = f"crops/nodet/{r.name.rsplit('.', 1)[0]}.jpg"
        save_frame_preview(image, None, out_dir / thumb_rel, max_side=320)
        no_det.append({
            "id": r.name.rsplit(".", 1)[0],
            "time": r.timestamp.strftime("%H:%M"),
            "n_frames": 1, "best_frame": r.name, "thumb": thumb_rel,
        })
    timings["thumbs"] = time.time() - t0

    # --- outputs ---------------------------------------------------------
    t0 = time.time()
    cands.sort(key=lambda c: (0 if c.possible_lifer
                              else 1 if c.lifer_lookalike
                              else 2, c.time_start))
    write_candidates_csv(cands, out_dir / "candidates.csv")
    # no-detection encounters, for the review app's "No bird detected" section
    # (full-frame thumbnail + fix-crop); a re-run rewrites this file
    (out_dir / "nodet.json").write_text(
        json.dumps(nodet_encs, indent=1) + "\n", encoding="utf-8")
    write_review_html(
        cands, no_det, out_dir / "review.html", name,
        region={
            "name": region_name,
            "short": region_short,
            "month": calendar.month_abbr[m],
            "window": window_label,
        },
    )
    timings["outputs"] = time.time() - t0
    timings["total"] = time.time() - t_start

    (out_dir / "timing.json").write_text(json.dumps(
        {k: round(v, 1) for k, v in timings.items()}, indent=1))

    n_lifers = sum(1 for c in cands if c.possible_lifer)
    n_lookalikes = sum(1 for c in cands if c.lifer_lookalike)
    print(f"\n{name}: {len(photos)} frames, {len(encounters)} encounters "
          f"(+{n_extra} extra birds), "
          f"{n_lifers} possible lifers, {n_lookalikes} lifer lookalikes, "
          f"{len(nodet_encs)} encounters + {len(no_det) - len(nodet_encs)} "
          f"frames w/o birds")
    print(f"outputs: {out_dir}  (total {timings['total']:.0f}s)")
    db_conn.close()
    return out_dir
