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


def normalize_date(s: str) -> str:
    """Normalise a LIFELIST.ods date: ``UK`` -> ``UNK``.

    The ODS uses UNK for an unknown date (and ``x/UNK/xxxx`` when month+year
    are known), but some rows were typed UK. Rewrite a standalone UK, or a UK
    as a component of a slash-separated date (``7/UK/2021``), to UNK.
    Case-insensitive and whole-component only: text that merely contains "uk"
    is left alone.
    """
    s = (s or "").strip()
    if not s:
        return s
    if s.lower() == "uk":
        return "UNK"
    if "/" in s:
        return "/".join("UNK" if p.strip().lower() == "uk" else p
                        for p in s.split("/"))
    return s


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
            return i, [idx.get(name, pos) for pos, name in enumerate(order)]
    return 0, [0, 1, 2, 3]


def parse_lifelist(path: Path) -> list[list[str]]:
    """Rows as [species, date, location, details], header stripped.

    Dates are normalised (UK -> UNK) so every downstream consumer of the
    import -- lifelist.csv, the review app, `lifer publish` -- sees UNK.
    """
    rows = parse_ods(path)
    header_i, order = _find_header(rows)
    data = []
    for row in rows[header_i + 1:]:
        vals = [row[i].strip() if i < len(row) else "" for i in order]
        vals[1] = normalize_date(vals[1])
        data.append(vals)
    return data


# How close to the outright best a fuzzy score must be to count as a near-tie.
# A split pair scores the same against the old name ("Herring Gull" is 90
# against both European and American Herring Gull); a different species that
# merely shares a word scores well below.
NEAR_TIE = 3


def match_lifelist(entries: list[list[str]], taxa: list[Taxon], cutoff: int = 80,
                   counts_by_place: dict[str, dict[int, float]] | None = None,
                   default_counts: dict[int, float] | None = None) -> list[LifeEntry]:
    """Fuzzy-match each typed species name to an Aves taxon by common name.

    Scientific names in the lifelist also match: we key the candidate table on
    both common and scientific names so either input form resolves.

    A near-tie (a recent split typed under the old name, e.g. European vs
    American Herring Gull) is resolved toward the taxon most observed where
    the entry was seen: ``counts_by_place`` maps a lowercased location as the
    lifelist writes it ("leon county, fl") to iNat observation counts per
    taxon id, and ``default_counts`` serves locations not in it. Each entry
    uses its own location, so the result never depends on which trip is
    being run. An exact name match is never overridden.
    """
    # candidate table: name -> taxon. Common names are the primary key;
    # scientific names included as secondary candidates.
    candidates: dict[str, Taxon] = {}
    for t in taxa:
        if t.common:
            candidates[t.common.lower()] = t
        candidates[t.scientific.lower()] = t
    keys = list(candidates.keys())
    by_place = counts_by_place or {}

    out: list[LifeEntry] = []
    for typed, date, location, _details in entries:
        if not typed:
            continue
        # WRatio is case-sensitive; keys are lowercase, so lowercase the query.
        # Keep the top handful so a near-tied taxon can still win below.
        top = process.extract(typed.lower(), keys, scorer=fuzz.WRatio,
                              score_cutoff=cutoff, limit=8)
        if not top:
            out.append(LifeEntry(typed=typed, date=date, location=location,
                                 taxon=None, score=0.0))
            continue
        best_name, best_score, _ = top[0]
        chosen = candidates[best_name]
        chosen_score = best_score
        counts = by_place.get(location.strip().lower(), default_counts)
        if counts and best_score < 100:
            near = [(candidates[name], score) for name, score, _ in top
                    if score >= best_score - NEAR_TIE]
            # max() keeps the first (best-scoring) of equally observed taxa
            n_obs, taxon, score = max(((counts.get(t.id, 0), t, s) for t, s in near),
                                      key=lambda x: x[0])
            if n_obs > 0:
                chosen, chosen_score = taxon, score
        out.append(LifeEntry(typed=typed, date=date, location=location,
                             taxon=chosen, score=chosen_score))
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
