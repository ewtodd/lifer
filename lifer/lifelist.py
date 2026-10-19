"""Lifelist handling: parse LIFELIST.ods, fuzzy-match entries to iNat taxa.

The ODS has columns SPECIES, DATE, LOCATION, DETAILS (plus legend columns on
the right, which we ignore). Matching is by common name with rapidfuzz
WRatio against the global Aves species list; entries that don't match well
are recorded as unmatched.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from odf.opendocument import load
from odf.table import TableCell, TableRow, Table
from odf.text import P
from rapidfuzz import fuzz, process

from .taxonomy import Taxon


@dataclass
class LifeEntry:
    typed: str
    date: str
    location: str
    taxon: Taxon | None
    score: float


def _cell_text(cell: TableCell) -> str:
    # odfpy stores cell text as Text nodes (nodeType 3) inside the P elements
    parts: list[str] = []
    for p in cell.getElementsByType(P):
        for node in p.childNodes:
            if node.nodeType == 3:
                parts.append(node.data)
    return "".join(parts).strip()


def parse_ods(path: Path) -> list[list[str]]:
    """Return the first four non-legend columns of the first sheet as rows."""
    doc = load(str(path))
    sheet: Table = doc.spreadsheet.getElementsByType(Table)[0]
    rows: list[list[str]] = []
    for row in sheet.getElementsByType(TableRow):
        values: list[str] = []
        for cell in row.getElementsByType(TableCell):
            if len(values) >= 4:
                break
            values.append(_cell_text(cell))
        if any(v for v in values):
            rows.append(values + [""] * (4 - len(values)))
    return rows


def _find_header(rows: list[list[str]]) -> tuple[int, list[int]]:
    for i, row in enumerate(rows[:5]):
        head = [v.strip().upper() for v in row]
        if "SPECIES" in head:
            # map the four canonical columns by name, fall back to position
            idx = {name: head.index(name) for name in ("SPECIES", "DATE", "LOCATION", "DETAILS") if name in head}
            order = ["SPECIES", "DATE", "LOCATION", "DETAILS"]
            return i, [idx.get(n, n_pos) for n, n_pos in enumerate(order)]
    return 0, [0, 1, 2, 3]


def parse_lifelist(path: Path) -> list[list[str]]:
    """Rows as [species, date, location, details], header stripped."""
    rows = parse_ods(path)
    header_i, order = _find_header(rows)
    data = []
    for row in rows[header_i + 1:]:
        data.append([row[i].strip() if i < len(row) else "" for i in order])
    return data


def match_lifelist(entries: list[list[str]], taxa: list[Taxon], cutoff: int = 80) -> list[LifeEntry]:
    """Fuzzy-match each typed species name to an Aves taxon by common name.

    Scientific names in the lifelist also match: we key the candidate table on
    both common and scientific names so either input form resolves.
    """
    # candidate table: name -> taxon. Common names are the primary key;
    # scientific names included as secondary candidates.
    candidates: dict[str, Taxon] = {}
    for t in taxa:
        if t.common:
            candidates[t.common.lower()] = t
        candidates[t.scientific.lower()] = t
    keys = list(candidates.keys())

    out: list[LifeEntry] = []
    for typed, date, location, _details in entries:
        if not typed:
            continue
        # WRatio is case-sensitive; keys are lowercase, so lowercase the query
        best = process.extractOne(typed.lower(), keys, scorer=fuzz.WRatio, score_cutoff=cutoff)
        if best is None:
            out.append(LifeEntry(typed=typed, date=date, location=location, taxon=None, score=0.0))
        else:
            name, score, _ = best
            out.append(LifeEntry(typed=typed, date=date, location=location,
                                 taxon=candidates[name], score=score))
    return out


def write_lifelist_csv(entries: list[LifeEntry], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["typed_name", "common_name", "scientific_name", "taxon_id",
                    "date", "location", "match_score"])
        for e in entries:
            w.writerow([
                e.typed,
                e.taxon.common if e.taxon else "",
                e.taxon.scientific if e.taxon else "",
                e.taxon.id if e.taxon else "",
                e.date,
                e.location,
                f"{e.score:.0f}",
            ])


def lifer_set(entries: list[LifeEntry]) -> set[int]:
    """Taxon ids of every matched lifelist entry (unmatched contribute nothing)."""
    return {e.taxon.id for e in entries if e.taxon is not None}
