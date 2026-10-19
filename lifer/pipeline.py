"""Pipeline orchestration: NEF -> detect -> encounters -> BioCLIP -> outputs.

Slow/networked work is cached under cache/ keyed per image and per API result,
so re-runs skip finished work. Nothing is ever written under the source folder.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from .bioclip import BioCLIP, label_text
from .boxes import get_box, open_db
from .bursts import Encounter, FrameInfo, group_into_encounters
from .detect import BirdDetector, Detection, FrameDetections, square_crop, sharpness
from .extract import load_frame, jpeg_to_image
from .home import lifer_home, find_regions_toml
from .lifelist import (LifeEntry, lifer_set, match_lifelist, parse_lifelist,
                       write_lifelist_csv)
from .outputs import Candidate, save_crop, save_frame_preview, write_candidates_csv, write_review_html
from .region import normalized, region_display_names, species_counts
from .taxonomy import aves_species

LIFELIST = Path.home() / "BIRDING" / "LIFELIST.ods"

# A lifelist species in the top 3 whose adjusted probability is at least this
# fraction of the top-1's counts as "close behind": the encounter is then a
# lookalike of a known bird, not a possible lifer. 0.1 sits in the natural
# gap of the Dec 29 data between close second places (0.13-0.43, e.g. the
# plover and scaup pairs) and distant ones (<0.11); see REPORT.md Task 02.
LOOKALIKE_FRACTION = 0.1

# "Rare here": the model is confident in a species (raw, T-calibrated top-1
# above RARE_CONF) AND that species is uncommon at this place in the photo's
# month window (share of the seasonal counts below RARE_SHARE, or absent).
# RARE_CONF: at T=3.0 a top-1 of 0.60 means the model clearly leads one species
# (not a coin-flip) -- it catches the weakest AMGP plover (0.651) with margin.
# RARE_SHARE: 0.2% of the regional month-window observations. AMGP sits at
# ~0.17% (below), its common look-alike Black-bellied Plover at ~0.89% (above),
# so the prior's favorite stays unflagged while the true vagrant is lifted.
RARE_CONF = 0.60
RARE_SHARE = 0.002


def resolve_region(folder_name: str) -> dict:
    """First regions.toml entry whose glob matches the folder name."""
    cfg = tomllib.loads(find_regions_toml().read_text())
    for region in cfg.get("region", []):
        import fnmatch
        if fnmatch.fnmatch(folder_name, region["glob"]):
            return region
    raise SystemExit(
        f"no region in regions.toml matches folder {folder_name!r}; "
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


def load_frame_with_dets(
    nef: Path, detector: BirdDetector, cache_dir: Path
) -> FrameRecord:
    """Extract + detect one NEF, using the per-frame cache when valid."""
    cache = _frame_cache_path(cache_dir, nef)
    key = {
        "yolo": detector.weights_key,
        "imgsz": detector.imgsz,
        "conf": detector.conf,
    }
    st = nef.stat()
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if (data.get("name") == nef.name and data.get("mtime") == st.st_mtime
                    and data.get("size") == st.st_size and data.get("key") == key):
                dets = data["dets"]
                fd = FrameDetections(frame_name=nef.name, dets=[
                    Detection(d[0], d[1], d[2], d[3], d[4]) for d in dets])
                return FrameRecord(
                    path=nef, name=nef.name,
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    dets=fd, sharpness=data["sharpness"],
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupt cache entry: recompute

    frame = load_frame(nef)
    image = jpeg_to_image(frame.jpeg)
    fd = detector.detect(image)
    fd.frame_name = nef.name
    if fd.best is not None:
        crop = square_crop(image, fd.best)
        sh = sharpness(crop)
    else:
        sh = -1.0
    rec = FrameRecord(
        path=nef, name=nef.name, timestamp=frame.timestamp,
        dets=fd, sharpness=sh, _jpeg=frame.jpeg,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "name": nef.name, "mtime": st.st_mtime, "size": st.st_size, "key": key,
        "timestamp": frame.timestamp.isoformat(),
        "dets": [[d.x1, d.y1, d.x2, d.y2, d.conf] for d in fd.dets],
        "width": image.width, "height": image.height,
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
        box = get_box(folder, f.name, conn, dets=rec.dets.dets)
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
        image = jpeg_to_image(rec._jpeg) if rec._jpeg else jpeg_to_image(load_frame(rec.path).jpeg)
        crops.append((f.name, square_crop(image, box)))
        rec._jpeg = None  # drop the decoded JPEG; keep only the crop

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


def run_folder(folder_arg: str, gap_seconds: float = 5.0, alpha: float = 0.3,
               yolo_weights: str = "yolo11m.pt", imgsz: int = 1280,
               conf: float = 0.25, temperature: float = 3.0) -> Path:
    t_start = time.time()
    folder = Path(folder_arg).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")
    home = lifer_home()
    name = folder.name
    out_dir = home / "runs" / name
    cache_dir = home / "cache" / name
    inat_cache = home / "cache" / "inat"
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
    matched = match_lifelist(entries, taxa)
    write_lifelist_csv(matched, home / "data" / "lifelist.csv")
    lifers = lifer_set(matched)
    n_unmatched = sum(1 for e in matched if e.taxon is None)
    timings["lifelist"] = time.time() - t0
    print(f"lifelist: {len(matched)} entries, {n_unmatched} unmatched "
          f"({len(lifers)} distinct taxa) [{timings['lifelist']:.1f}s]")

    # --- frames: extract + detect ---------------------------------------
    t0 = time.time()
    nefs = sorted(folder.glob("*.NEF"))
    if not nefs:
        raise SystemExit(f"no .NEF files in {folder}")
    detector = BirdDetector(weights=yolo_weights, imgsz=imgsz, conf=conf)
    records: dict[str, FrameRecord] = {}
    for i, nef in enumerate(nefs, 1):
        rec = load_frame_with_dets(nef, detector, cache_dir)
        records[rec.name] = rec
        if i % 25 == 0 or i == len(nefs):
            print(f"  frames {i}/{len(nefs)} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    n_detected = sum(1 for r in records.values() if r.dets.dets)
    timings["detect"] = time.time() - t0
    print(f"detection: {n_detected}/{len(nefs)} frames with birds "
          f"[{timings['detect']:.0f}s]")

    # --- encounters ------------------------------------------------------
    t0 = time.time()
    infos = [FrameInfo(name=r.name, timestamp=r.timestamp, n_dets=len(r.dets.dets),
                       best_conf=r.dets.best.conf if r.dets.best else 0.0,
                       sharpness=r.sharpness) for r in records.values()]
    encounters = group_into_encounters(infos, gap_seconds)
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
    cands: list[Candidate] = []
    order = sorted(classes)
    for i, enc in enumerate(encounters, 1):
        enc_id = f"enc_{i:04d}"
        best = enc.best_frame
        rec = records[best.name]
        prob_vec = classify_encounter(enc, records, classes, bioclip, txt, cache_dir,
                                      class_hash, temperature, folder=name,
                                      conn=db_conn)
        p_clip = dict(zip(order, prob_vec))
        p_adj = {t: p_clip[t] * season.get(t, 0.0) ** alpha for t in order}
        adj_total = sum(p_adj.values()) or 1.0
        p_adj = {t: v / adj_total for t, v in p_adj.items()}

        def top3(dist: dict[int, float]) -> list[tuple[str, str, float]]:
            top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
            return [(classes[t].common, classes[t].scientific, p) for t, p in top]

        adj_sorted = sorted(p_adj.items(), key=lambda kv: kv[1], reverse=True)
        adj_top1, adj_top1_p = adj_sorted[0]
        has_det = bool(rec.dets.dets)

        # "Rare here": a confident raw top-1 on a species that is uncommon at
        # this place in the photo's month window. Computed on the RAW
        # (T-calibrated, prior-free) probabilities so the regional prior can't
        # mask a true vagrant; beats lifer_lookalike (see run_folder docstring).
        raw_top1_tid, raw_top1_p = max(p_clip.items(), key=lambda kv: kv[1])
        rare_here = (
            has_det
            and raw_top1_p >= RARE_CONF
            and (seasonal_counts.get(raw_top1_tid, 0.0) / seasonal_total) < RARE_SHARE
        )
        # evidence behind the rare flag: the species' share of the place's
        # month-window records and of its all-time records
        rare_evidence = ""
        if rare_here:
            s_share = seasonal_counts.get(raw_top1_tid, 0.0) / seasonal_total
            a_share = alltime_counts.get(raw_top1_tid, 0.0) / alltime_total
            rare_evidence = (
                f"{s_share * 100:.2f}% of {window_label} iNaturalist records "
                f"in {region_name} (all-time {a_share * 100:.1f}%)"
            )

        possible_lifer = False
        lookalike = None
        if has_det and adj_top1 not in lifers:
            # Closest lifelist species in the top 3; if it's within
            # LOOKALIKE_FRACTION of the top-1's probability, flag it as a
            # lookalike of a known bird instead of a possible lifer.
            close = max(((t, p) for t, p in adj_sorted[:3] if t in lifers),
                        key=lambda tp: tp[1], default=None)
            if close is not None and close[1] >= LOOKALIKE_FRACTION * adj_top1_p:
                lookalike = classes[close[0]].common
            else:
                possible_lifer = True
        cands.append(Candidate(
            enc_id=enc_id,
            time_start=enc.start.strftime("%Y-%m-%d %H:%M:%S"),
            time_end=enc.end.strftime("%Y-%m-%d %H:%M:%S"),
            n_frames=len(enc.frames),
            best_frame=best.name,
            det_conf=best.best_conf,
            raw_top3=top3(p_clip),
            adj_top3=top3(p_adj),
            possible_lifer=possible_lifer,
            lifer_lookalike=lookalike,
            has_det=has_det,
            rare_here=rare_here,
            rare_evidence=rare_evidence,
        ))
        # best crop for the review page (encounters without any detection
        # get a placeholder card in the HTML)
        if has_det:
            crop_box = get_box(name, best.name, db_conn, dets=rec.dets.dets)
            if crop_box is not None:
                image = jpeg_to_image(rec._jpeg) if rec._jpeg else jpeg_to_image(load_frame(rec.path).jpeg)
                save_crop(square_crop(image, crop_box), out_dir / "crops" / f"{enc_id}.jpg")
                # full-frame preview with the detection box, for the lightbox
                save_frame_preview(image, crop_box, out_dir / "frames" / f"{enc_id}.jpg")
                # box-less full-frame preview for the crop editor (same scale)
                save_frame_preview(image, None, out_dir / "frames" / "src" / f"{enc_id}.jpg")
                rec._jpeg = None
        if i % 10 == 0 or i == len(encounters):
            print(f"  classified {i}/{len(encounters)} encounters "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    timings["bioclip"] = time.time() - t0
    print(f"bioclip: {timings['bioclip']:.0f}s")

    # --- no-detection thumbnails ----------------------------------------
    t0 = time.time()
    no_det: list[tuple[str, str]] = []
    for r in records.values():
        if r.dets.dets:
            continue
        image = jpeg_to_image(r._jpeg) if r._jpeg else jpeg_to_image(load_frame(r.path).jpeg)
        thumb_rel = f"crops/nodet/{r.name.rsplit('.', 1)[0]}.jpg"
        save_frame_preview(image, None, out_dir / thumb_rel, max_side=320)
        no_det.append((r.name, thumb_rel))
    timings["thumbs"] = time.time() - t0

    # --- outputs ---------------------------------------------------------
    t0 = time.time()
    cands.sort(key=lambda c: (0 if c.possible_lifer
                              else 1 if c.lifer_lookalike
                              else 2, c.time_start))
    write_candidates_csv(cands, out_dir / "candidates.csv")
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
    print(f"\n{name}: {len(nefs)} frames, {len(encounters)} encounters, "
          f"{n_lifers} possible lifers, {n_lookalikes} lifer lookalikes, "
          f"{len(no_det)} frames w/o birds")
    print(f"outputs: {out_dir}  (total {timings['total']:.0f}s)")
    db_conn.close()
    return out_dir
