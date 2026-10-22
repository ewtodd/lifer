"""Re-identify a crop after a manual box fix (CPU-only; no GPU in the service).

The always-on review service must not hold GPU memory, so the BioCLIP image
encoder is loaded lazily on the CPU on first use and freed after ~10 idle
minutes. Label text embeddings are reused from LIFER_HOME/cache/inat (they
only depend on the class set, so the pipeline's cached matrix is reused here).
Re-identification recomputes the encounter's raw/adj top 3 with the same prior
and temperature as the pipeline.
"""

from __future__ import annotations

import calendar
import gc
import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

from .bioclip import BioCLIP
from .birds import species_flags
from .home import lifer_home, find_regions_toml

# Must match the pipeline's run_folder defaults/knobs (the values the
# candidate's top 3 were produced with).
TEMP = 3.0
ALPHA = 0.3
IDLE_SECONDS = 600
LIFELIST = Path.home() / "BIRDING" / "LIFELIST.ods"

_encoder: BioCLIP | None = None
_encoder_at = 0.0
_lock = threading.Lock()
_watchdog: threading.Thread | None = None


def _region_for(folder: str) -> dict:
    """First regions.toml [[region]] whose glob matches the folder name."""
    import fnmatch
    import tomllib
    cfg = tomllib.loads(find_regions_toml().read_text())
    for region in cfg.get("region", []):
        if fnmatch.fnmatch(folder, region["glob"]):
            return region
    raise SystemExit(f"no region in regions.toml matches folder {folder!r}")


def _frame_cache(folder: str, frame: str) -> dict[str, object]:
    p = lifer_home() / "cache" / folder / f"{Path(frame).stem}.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _month_from_frame(fr: dict) -> datetime:
    ts = fr.get("timestamp")
    try:
        return datetime.fromisoformat(ts) if ts else datetime.now()
    except (ValueError, TypeError):
        return datetime.now()


def _free() -> None:
    global _encoder, _encoder_at
    with _lock:
        if _encoder is not None:
            _encoder = None
            _encoder_at = 0.0
            gc.collect()
            print("[reid] image encoder freed (idle)", flush=True)


def _watchdog_loop() -> None:
    while True:
        time.sleep(60)
        if _encoder is not None and time.time() - _encoder_at > IDLE_SECONDS:
            _free()


def _ensure_watchdog() -> None:
    global _watchdog
    if _watchdog is None:
        _watchdog = threading.Thread(target=_watchdog_loop, daemon=True)
        _watchdog.start()


def _get_encoder() -> BioCLIP:
    global _encoder, _encoder_at
    if _encoder is not None and time.time() - _encoder_at > IDLE_SECONDS:
        _free()
    if _encoder is None:
        t0 = time.time()
        enc = BioCLIP(device="cpu")
        with _lock:
            _encoder = enc
            _encoder_at = time.time()
        _ensure_watchdog()
        print(f"[reid] image encoder loaded (cpu) in {time.time() - t0:.1f}s", flush=True)
    _encoder_at = time.time()
    return _encoder


def identify_crop(folder: str, frame: str, crop) -> dict:
    """Re-identify `crop` (a PIL Image) against the folder's species set.

    Returns the derived candidate fields (raw/adj top 3 + flags) plus the
    single-crop inference time. Uses the cached label embeddings, so it needs
    only the image encoder, loaded on the CPU.
    """
    from .lifelist import lifer_set, match_lifelist, parse_lifelist
    from .region import (lifelist_place_counts, normalized,
                         region_display_names, species_counts)
    from .taxonomy import aves_species

    home = lifer_home()
    inat_cache = home / "cache" / "inat"

    region = _region_for(folder)
    place_id = int(region["place_id"])
    fr = _frame_cache(folder, frame)
    month = _month_from_frame(fr).date()
    classes, alltime_counts, seasonal_counts = species_counts(
        place_id, month, inat_cache)
    season = normalized(seasonal_counts)
    rn, _rs = region_display_names(region, inat_cache)
    region_name = rn or folder
    m = month.month
    first = m - 1 if m > 1 else 12
    last = m + 1 if m < 12 else 1
    window_label = f"{calendar.month_abbr[first]}\u2013{calendar.month_abbr[last]}"

    taxa = aves_species(inat_cache)
    # same per-location matching as the pipeline, so both agree on the lifers
    place_counts, all_counts = lifelist_place_counts(inat_cache)
    matched = match_lifelist(parse_lifelist(LIFELIST), taxa,
                             counts_by_place=place_counts, default_counts=all_counts)
    lifers = lifer_set(matched)

    order = sorted(classes)
    class_hash = hashlib.sha1(json.dumps(order).encode()).hexdigest()[:16]
    label_cache = inat_cache / f"bioclip_labels_{class_hash}.pt"
    if not label_cache.is_file():
        raise RuntimeError(
            f"no cached label embeddings at {label_cache}; run the pipeline first")
    txt = torch.load(label_cache, map_location="cpu")

    enc = _get_encoder()
    t0 = time.time()
    probs = enc.probabilities([crop], txt, TEMP)[0]
    elapsed = time.time() - t0
    p_clip = {t: float(p) for t, p in zip(order, probs)}
    p_adj_raw = {t: p_clip[t] * season.get(t, 0.0) ** ALPHA for t in order}
    adj_total = sum(p_adj_raw.values()) or 1.0
    p_adj = {t: v / adj_total for t, v in p_adj_raw.items()}

    flags = species_flags(p_clip, p_adj, classes, lifers, seasonal_counts,
                          alltime_counts, region_name, window_label)
    adj_top = flags["adj_top3"]
    return {
        **flags,
        # the review app stores "no lookalike" as an empty string
        "lifer_lookalike": flags["lifer_lookalike"] or "",
        "elapsed": elapsed,
        "top1": adj_top[0][0] if adj_top else "",
    }
