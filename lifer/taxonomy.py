"""Full iNaturalist Aves species list, for fuzzy-matching the lifelist.

The public ``/taxa`` API hard-caps pagination at offset 10,000 (page 51 at
per_page=200 returns 403), and Aves has ~11.4k species. So we chunk by
order: fetch Aves' orders (one page), then species per order — the largest
order (Passeriformes, ~6.4k) stays under the cap. Results are cached as JSON
under cache/, so this only runs once.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

API = "https://api.inaturalist.org/v1"
AVES = 3


def _get(path: str, params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    """GET with backoff on 403/429: the unauthenticated API allows ~50 req/min.

    (403 also means "offset past the 10k cap" — that is a caller bug, not a
    transient error; callers chunk requests to stay under it.)
    """
    r = None
    for attempt in range(retries):
        r = requests.get(f"{API}/{path}", params=params, timeout=60)
        if r.status_code in (403, 429):
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    assert r is not None
    r.raise_for_status()
    raise RuntimeError("unreachable")


def _taxa_for(taxon_id: int) -> list[dict[str, Any]]:
    """All species under a taxon, chunked to respect the 10k offset cap."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        data = _get("taxa", {"taxon_id": taxon_id, "rank": "species",
                             "per_page": 200, "page": page})
        rows = data["results"]
        if not rows:
            break
        out.extend(rows)
        if len(out) >= data["total_results"]:
            break
        page += 1
        time.sleep(0.3)  # stay well under the ~50 req/min public limit
    return out


@dataclass
class Taxon:
    id: int
    scientific: str
    common: str


def aves_species(cache_dir: Path) -> list[Taxon]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / "aves_species.json"
    if f.exists():
        return [Taxon(**v) for v in json.loads(f.read_text())]

    data = _get("taxa", {"taxon_id": AVES, "rank": "order", "per_page": 100})
    orders = data["results"]
    out: list[Taxon] = []
    for order in orders:
        for t in _taxa_for(order["id"]):
            out.append(Taxon(id=t["id"], scientific=t["name"],
                             common=t.get("preferred_common_name") or ""))
        print(f"  taxonomy: {order['name']}: {len(out)} species so far", flush=True)
    f.write_text(json.dumps([asdict(t) for t in out], indent=0))
    return out
