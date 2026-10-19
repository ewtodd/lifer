"""run-all: process every photo folder under a root (the backlog).

`lifer run-all <root>` walks <root> (read-only) and finds every folder that
directly contains photos (NEFs, or, in a folder with no NEFs, JPEGs).
Folders that are just copies of the same shots are skipped:

- a JPEG-only folder inside a top-level outing tree that has NEFs anywhere
  is a derived copy of those raws (exported previews, WATERMARKED,
  darktable_exported, a parent camera); trees with no NEFs at all keep the
  JPEG-only behaviour (task 24);
- a folder whose own capture set (by filename + size) is contained in a
  larger folder (curated `pics/`-style subsets).

A subfolder that holds the outing's raws and is named like a raw bucket
(RAW, raw, NEF) runs under the parent's trip ID, not its own nested one
(task 24). Folders whose trip id matches no region in regions.toml are
skipped and listed too. The rest is processed one at a time -- so GPU
memory never exceeds a single `lifer run` -- and an error on one folder
does not stop the rest. The plan (plus the stale runs and orphaned
decisions it lists) and the final summary go to stdout and, in a real run,
to LIFER_HOME/runs/run-all-<ts>.log.
"""

from __future__ import annotations

import fnmatch
import json
import sqlite3
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .home import effective_trip_id, find_regions_toml, lifer_home, trip_id_for

PHOTO_EXTS = (".nef", ".jpg", ".jpeg")


def photo_files(d: Path) -> list[Path]:
    """Direct-child photo files (NEF or JPEG) of a folder, sorted by name."""
    try:
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in PHOTO_EXTS)
    except OSError:
        return []


@dataclass
class Row:
    folder: Path
    trip_id: str
    n_photos: int
    region: str  # matched region name, "" when none
    status: str  # "run" | "skip"
    reason: str  # "" for run, else the skip reason


def _has_nef(d: Path) -> bool:
    return any(p.suffix.lower() == ".nef" for p in photo_files(d))


def _tree_of(d: Path, root: Path) -> Path:
    """The top-level outing tree containing d (the direct child of root
    that d is nested under; d itself when d is top-level)."""
    if d == root:
        return d
    return root / d.relative_to(root).parts[0]


def nonderived_keys(files: list[Path]) -> dict[str, int]:
    """(name -> size) of a folder's own capture set: every NEF, plus the
    JPEGs that are not derived copies of an NEF in the same folder (a JPEG
    whose stem extends an NEF stem, e.g. `_DSC6581_lzn.jpg` for
    `_DSC6581.NEF`)."""
    nef_stems = [p.stem for p in files if p.suffix.lower() == ".nef"]
    out: dict[str, int] = {}
    for p in files:
        if p.suffix.lower() == ".nef":
            out[p.name] = p.stat().st_size
        elif not any(p.stem.startswith(s) for s in nef_stems):
            out[p.name] = p.stat().st_size
    return out


def build_plan(root: Path, regions: list[dict]) -> list[Row]:
    """Every photo folder under root with its trip id, region and status."""
    root = root.resolve()
    entries: list[tuple[Path, list[Path]]] = []
    for d in [root, *sorted(root.rglob("*"))]:
        if not d.is_dir():
            continue
        files = photo_files(d)
        if files:
            entries.append((d, files))

    # Task 24, rule 1: within a top-level outing tree, NEFs anywhere make
    # the tree a raw outing and every JPEG-only folder in it a derived
    # copy. Trees with no NEFs at all keep the task-20 behaviour.
    trees: dict[Path, bool] = {}
    for d, _files in entries:
        tree = _tree_of(d, root)
        trees[tree] = trees.get(tree, False) or _has_nef(d)

    nd_keys = {d: nonderived_keys(files) for d, files in entries}
    rows: list[Row] = []
    for d, files in entries:
        # Task 24, rule 2: a raw bucket runs under its parent's trip ID.
        tid = effective_trip_id(d, root)
        region = next((r for r in regions if fnmatch.fnmatch(tid, r["glob"])),
                      None)
        keys = nd_keys[d]
        if not _has_nef(d) and trees[_tree_of(d, root)]:
            reason = "derived copy (JPEGs of an NEF outing)"
        else:
            sup = next((effective_trip_id(g, root) for g, gk in nd_keys.items()
                        if g is not d and keys and keys.items() <= gk.items()
                        and len(gk) > len(keys)), None)
            if sup:
                reason = f"duplicate subset of {sup}"
            elif region is None:
                reason = "no region"
            else:
                reason = ""
        rows.append(Row(folder=d, trip_id=tid, n_photos=len(files),
                        region=region["name"] if region else "",
                        status="run" if reason == "" else "skip",
                        reason=reason))
    rows.sort(key=lambda r: r.trip_id)
    return rows


def format_plan(root: Path, rows: list[Row]) -> str:
    """The plan as text: one line per photo folder, then a summary."""
    if not rows:
        return f"run-all: no photo folders under {root}"
    w_id = max(len(r.trip_id) for r in rows)
    w_rg = max([len(r.region) for r in rows] + [len("REGION")])
    lines = [f"run-all: {len(rows)} photo folders under {root}"]
    for r in rows:
        line = (f"  {r.trip_id:<{w_id}}  {r.region:<{w_rg}}  "
                f"{r.n_photos:>5} photos  {r.status}")
        if r.reason:
            line += f": {r.reason}"
        lines.append(line)
    n_run = sum(1 for r in rows if r.status == "run")
    n_frames = sum(r.n_photos for r in rows if r.status == "run")
    n_derived = sum(1 for r in rows if "derived copy" in r.reason)
    n_dup = sum(1 for r in rows if "duplicate subset" in r.reason)
    n_noregion = sum(1 for r in rows if r.reason == "no region")
    lines.append(
        f"summary: {n_run} folders / {n_frames} photos to run; "
        f"{len(rows) - n_run} skipped ({n_derived} derived copies, "
        f"{n_dup} duplicate subsets, {n_noregion} no region)")
    return "\n".join(lines)


def _count_rows(conn: sqlite3.Connection | None, table: str, tid: str) -> int:
    if conn is None:
        return 0
    return conn.execute(f"SELECT COUNT(*) n FROM {table} WHERE folder=?",
                        (tid,)).fetchone()["n"]


def _chosen_frame_count(home: Path, tid: str) -> int:
    n = 0
    for sub in ("develop", "edits"):
        d = home / sub / tid
        if d.is_dir():
            n += sum(1 for p in d.rglob("*") if p.is_file())
    return n


def stale_runs_text(home: Path, conn: sqlite3.Connection | None,
                    rows: list[Row]) -> str:
    """Stale run dirs (trip ID the current rules no longer generate) with
    their DB / chosen-frame references, plus decisions that would be
    orphaned when a trip's source changes from JPEGs to NEFs (same trip
    ID, different frames). Report only -- nothing is deleted; that is
    `lifer delete`'s job."""
    lines: list[str] = []
    live = {r.trip_id for r in rows if r.status == "run"}
    runs = home / "runs"
    stale = []
    if runs.is_dir():
        stale = sorted(p.name for p in runs.iterdir()
                       if p.is_dir() and p.name not in live)
    for tid in stale:
        lines.append(
            f"  {tid}: {_count_rows(conn, 'decisions', tid)} decisions, "
            f"{_count_rows(conn, 'crop_boxes', tid)} crop fixes, "
            f"{_chosen_frame_count(home, tid)} chosen frames -- stale "
            f"(not generated by the current rules)")
    for r in rows:
        if r.status != "run":
            continue
        meta = home / "runs" / r.trip_id / "meta.json"
        if not meta.is_file():
            continue
        try:
            old_folder = Path(json.loads(meta.read_text())["folder"])
        except (OSError, KeyError, ValueError):
            continue
        if not old_folder.is_dir() or old_folder == r.folder:
            continue
        photos = {p.name for p in photo_files(r.folder)}
        if conn is None:
            continue
        decs = conn.execute(
            "SELECT enc_id, best_frame, action, species FROM decisions "
            "WHERE folder=? ORDER BY enc_id", (r.trip_id,)).fetchall()
        orph = [x for x in decs if x["best_frame"] not in photos]
        if orph:
            lines.append(
                f"  {r.trip_id} (source changed {old_folder.name} -> "
                f"{r.folder.name}): {len(orph)} decisions reference frames "
                f"that will no longer exist:")
            for x in orph:
                sp = f", {x['species']}" if x["species"] else ""
                lines.append(f"    {x['enc_id']} {x['best_frame']} "
                             f"({x['action']}{sp})")
    return "\n".join(lines)


def run_all(root_arg: str, dry_run: bool = False,
            apply_migrations: bool = False, **run_kwargs) -> str:
    """Plan (and unless dry_run, execute) every photo folder under root.

    `run_kwargs` are passed through to pipeline.run_folder (gap_seconds,
    alpha, yolo_weights, imgsz, conf, temperature). Returns the summary
    line. With `apply_migrations`, the task-24 trip migrations are applied
    after the trips have run (see trips_ops).
    """
    root = Path(root_arg).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    regions = tomllib.loads(find_regions_toml().read_text()).get("region", [])
    rows = build_plan(root, regions)
    plan_text = format_plan(root, rows)
    print(plan_text, flush=True)

    home = lifer_home()
    db_path = home / "data" / "lifer.sqlite"
    conn: sqlite3.Connection | None = None
    if db_path.is_file():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    stale = stale_runs_text(home, conn, rows)
    if stale:
        print("\nstale runs / decisions to migrate:", flush=True)
        print(stale, flush=True)
        plan_text += "\n\nstale runs / decisions to migrate:\n" + stale

    from .trips_ops import plan_migrate_trips
    mig = plan_migrate_trips(home, conn, root, rows)
    n_mig = len(mig["rekey"]) + len(mig["map"])
    if n_mig:
        print(f"\nmigrations available ({n_mig}): pass --apply-migrations "
              f"to apply them after the runs"
              + (" [dry-run: not applied]" if dry_run else ""), flush=True)

    to_run = [r for r in rows if r.status == "run"]
    if dry_run:
        return plan_text.splitlines()[-1]

    # Imported here so --dry-run stays torch-free (no GPU to plan with).
    import gc
    import time

    from .pipeline import run_folder
    import torch

    results: list[tuple[str, str, float, str]] = []
    for i, r in enumerate(to_run, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(to_run)}] {r.trip_id} "
              f"({r.n_photos} photos, {r.region})", flush=True)
        msg = ""
        try:
            run_folder(r.folder, root=root, **run_kwargs)
        except (SystemExit, Exception) as e:
            msg = str(e) or e.__class__.__name__
            print(f"  FAILED: {msg}", flush=True)
        finally:
            # Drop this folder's GPU memory before the next one: run-all must
            # never hold more than a single run does.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append((r.trip_id, "ok" if not msg else "FAILED",
                        time.time() - t0, msg))

    if apply_migrations:
        from .trips_ops import apply_migrate_trips, migration_summary
        print("\napplying trip migrations:", flush=True)
        res = apply_migrate_trips(home, conn, root, rows)
        print("  " + migration_summary(res), flush=True)

    n_ok = sum(1 for _, s, _, _ in results if s == "ok")
    final = (f"run-all done: {n_ok}/{len(to_run)} folders ok, "
             f"{len(to_run) - n_ok} failed "
             f"[{sum(t for _, _, t, _ in results):.0f}s]")
    log_path = lifer_home() / "runs" / (
        f"run-all-{datetime.now():%Y%m%dT%H%M%S}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        plan_text + "\n\n"
        + "\n".join(f"  {tid}  {status}  {t:.0f}s  {msg}"
                    for tid, status, t, msg in results)
        + f"\n\n{final}\n")
    print(f"\n{final}\nlog: {log_path}")
    return final
