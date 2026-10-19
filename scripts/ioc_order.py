"""Sort cache/inat/aves_species.json into true taxonomic (IOC) order.

The iNat cache lists the Aves species in iNat's *default* (popularity) order.
iNat's /taxa API has no taxonomic-order option (verified: it returns the same
default order for any rank mix), so the sequence comes from the IOC World
Bird List v15.2 (master_ioc_list_v15.2.xlsx, cached under cache/ioc/; rows
are already in taxonomic order).

The iNat species set is kept exactly (it is the BioCLIP label list); each
species is ranked by its position in the IOC list, matched on the
normalized scientific binomial, with a rapidfuzz fallback for renamed
taxa. Unmatched species sort to the end, alphabetically.

Usage (from the repo root, with the repo-root cache/ populated):
    nix develop . --command python scripts/ioc_order.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import openpyxl
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent
INAT = ROOT / "cache" / "inat" / "aves_species.json"
IOC = ROOT / "cache" / "ioc" / "master_ioc_list_v15.2.xlsx"
BAK = INAT.with_suffix(".json.bak")  # task-1 list, default order


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def ioc_species_order(xlsx: Path) -> list[str]:
    """Scientific binomials of every IOC species, in list (taxonomic) order."""
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb["Master"]
    order: list[str] = []
    genus = ""
    for row in ws.iter_rows(values_only=True, min_row=5):
        if not row:
            continue
        if row[5]:
            genus = str(row[5]).strip()
        sp = row[6]
        if sp and genus:
            order.append(f"{genus} {str(sp).strip()}")
    return order


def main() -> None:
    src = INAT if INAT.exists() else BAK
    taxa = json.loads(src.read_text())
    print(f"iNat species: {len(taxa)} (from {src.name})")

    ioc = ioc_species_order(IOC)
    ioc_norm = [norm(s) for s in ioc]
    print(f"IOC species: {len(ioc)}")

    rank = {s: i for i, s in enumerate(ioc_norm)}
    matched, fuzzy, missing = 0, 0, []
    ranked = []
    for t in taxa:
        n = norm(t["scientific"])
        if n in rank:
            r = rank[n]
            matched += 1
        else:
            hits = process.extractOne(n, ioc_norm, scorer=fuzz.WRatio, score_cutoff=90)
            if hits:
                r = hits[1]
                fuzzy += 1
            else:
                r = None
                missing.append(t["scientific"])
        ranked.append((r if r is not None else 10 ** 9, n, t))

    ranked.sort(key=lambda x: (x[0], x[1]))
    out = [t for _r, _n, t in ranked]
    INAT.write_text(json.dumps(out))
    print(f"matched exact: {matched}, fuzzy: {fuzzy}, unmatched: {len(missing)}")
    if missing:
        print("unmatched (sorted to end):", missing[:15])
    print("first 12:", [t["scientific"] for t in out[:12]])


if __name__ == "__main__":
    main()
