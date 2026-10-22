"""Regional species list from the iNaturalist API, cached under cache/.

``observations/species_counts`` for Aves (taxon_id=3) in a place, research
grade. Two views: all-time (base class list) and a seasonal prior: the
photo's calendar month plus its two neighbors, each across ALL years
(iNat month= without year=), blended with the all-time counts. No API key.
"""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .home import find_regions_toml

API = "https://api.inaturalist.org/v1"
AVES = 3


@dataclass
class Species:
    taxon_id: int
    scientific: str
    common: str


def _fetch_counts(place_id: int, d1: date | None, d2: date | None,
                  month: int | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "taxon_id": AVES,
        "place_id": place_id,
        "quality_grade": "research",
        "per_page": 200,
    }
    if d1 is not None:
        params["d1"] = d1.isoformat()
    if d2 is not None:
        params["d2"] = d2.isoformat()
    if month is not None:
        # month= without year= means "this calendar month, all years"
        params["month"] = month
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        r = None
        for attempt in range(5):  # back off on the ~50 req/min public limit
            r = requests.get(f"{API}/observations/species_counts", params={**params, "page": page}, timeout=60)
            if r.status_code not in (403, 429):
                break
            time.sleep(2.0 * (attempt + 1))
        r.raise_for_status()
        data = r.json()
        out.extend(data["results"])
        if page * int(params["per_page"]) >= data["total_results"]:
            break
        page += 1
        time.sleep(0.3)
    return out


def species_counts(
    place_id: int, month: date | None, cache_dir: Path
) -> tuple[dict[int, Species], dict[int, float], dict[int, float]]:
    """Return (classes, all_time_counts, seasonal_counts).

    classes: union of taxa seen in both queries.
    all_time_counts: research-grade counts over the place's full history.
    seasonal_counts: the photo month's 3-month window (all years) summed
        with all_time_counts; see the module docstring for why.
    Counts are raw observation counts, not normalized.
    Everything is cached under cache_dir, so re-runs skip the network.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"species_counts_{place_id}"

    alltime_file = cache_dir / f"{tag}_alltime.json"
    if alltime_file.exists():
        raw = json.loads(alltime_file.read_text())
        alltime = {int(k): Species(**{kk: vv for kk, vv in v.items() if kk != "count"})
                   for k, v in raw.items()}
        alltime_counts = {int(k): v["count"] for k, v in raw.items()}
    else:
        rows = _fetch_counts(place_id, None, None)
        alltime = {}
        alltime_counts = {}
        for row in rows:
            t = row["taxon"]
            if t.get("rank") != "species":
                continue
            alltime[t["id"]] = Species(
                taxon_id=t["id"],
                scientific=t["name"],
                common=t.get("preferred_common_name") or "",
            )
            alltime_counts[t["id"]] = row["count"]
        alltime_file.write_text(json.dumps(
            {k: {**asdict(v), "count": alltime_counts[k]} for k, v in alltime.items()}, indent=0))

    if month is not None:
        # Seasonal window: the photo's month plus its two neighbors
        # (e.g. Nov/Dec/Jan for December), each across ALL years. A single
        # calendar month for a single year is far too sparse to be
        # representative; pooling three months smooths that out.
        m = month.month
        window = (m - 1 if m > 1 else 12, m, m + 1 if m < 12 else 1)
        season_file = cache_dir / f"{tag}_win{m:02d}.json"
        if season_file.exists():
            seasonal = {int(k): float(v) for k, v in json.loads(season_file.read_text()).items()}
        else:
            pooled: dict[int, int] = {}
            for wm in window:
                for row in _fetch_counts(place_id, None, None, month=wm):
                    t = row["taxon"]
                    if t.get("rank") == "species":
                        pooled[t["id"]] = pooled.get(t["id"], 0) + row["count"]
            # Blend with the all-time counts: raw count sums, with all-time
            # acting as a heavy stabilizer so sparse months can't swing the
            # prior too far.
            seasonal = {tid: pooled.get(tid, 0) + c for tid, c in alltime_counts.items()}
            for tid, c in pooled.items():
                if tid not in seasonal:
                    seasonal[tid] = c
            season_file.write_text(json.dumps(seasonal))
    else:
        seasonal = {}

    classes = dict(alltime)
    for tid in seasonal:
        classes.setdefault(tid, Species(taxon_id=tid, scientific=f"taxon_{tid}", common=""))

    return classes, alltime_counts, seasonal


def lifelist_place_counts(cache_dir: Path) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    """All-time counts for every regions.toml place, for lifelist matching.

    Returns ({lowercased region name: {taxon_id: count}}, the counts summed
    over every region). The lifelist's location column names places the way
    regions.toml does ("Leon County, FL"), so each entry can be matched
    against the place it was seen; the sum serves any other location.
    """
    regions = tomllib.loads(find_regions_toml().read_text()).get("region", [])
    by_id: dict[int, dict[int, float]] = {}
    by_place: dict[str, dict[int, float]] = {}
    total: dict[int, float] = {}
    for region in regions:
        place_id = int(region["place_id"])
        if place_id not in by_id:
            try:
                _, counts, _ = species_counts(place_id, None, cache_dir)
            except requests.RequestException as e:
                # only an uncached place can fail; match without it this time
                print(f"  iNat species counts unavailable for place {place_id}: {e}")
                continue
            by_id[place_id] = counts
            for tid, c in counts.items():
                total[tid] = total.get(tid, 0) + c
        name, _short = region_display_names(region, cache_dir)
        if name:
            by_place[name.strip().lower()] = by_id[place_id]
    return by_place, total


def normalized(counts: dict[int, float], floor: float = 1.0) -> dict[int, float]:
    """Normalize counts to a distribution; `floor` (virtual counts) keeps rare
    species from being zeroed out by the prior."""
    shifted = {k: v + floor for k, v in counts.items()}
    total = sum(shifted.values()) or 1.0
    return {k: v / total for k, v in shifted.items()}


def place_display_name(place_id: int, cache_dir: Path) -> str | None:
    """Display name from the iNat places API, cached under cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"place_{place_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text()).get("display_name")
        except json.JSONDecodeError:
            pass  # corrupt cache; refetch below
    try:
        r = requests.get(f"{API}/places/{place_id}", timeout=15)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        print(f"  iNat place lookup failed for {place_id}: {e}")
        return None
    path.write_text(json.dumps(j))
    return j.get("display_name")


def region_display_names(region: dict[str, Any] | None,
                         cache_dir: Path | None = None) -> tuple[str, str]:
    """(name, short_name) for a regions.toml entry.

    Missing keys fall back to the iNat places API display name (short name
    then to its first component); both empty when there is no region.
    """
    if not region:
        return "", ""
    name = region.get("name") or ""
    if not name and cache_dir is not None and "place_id" in region:
        name = place_display_name(int(region["place_id"]), cache_dir) or ""
    short = region.get("short_name") or (name.split(",")[0].strip() if name else "")
    return name, short
