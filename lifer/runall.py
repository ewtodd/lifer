"""run-all: process every photo folder under a root (the backlog).

`lifer run-all <root>` walks <root> (read-only) and finds every folder that
directly contains photos (NEFs, or, in a folder with no NEFs, JPEGs). Folders
that are just copies of the same shots are skipped: a folder whose own
capture set (by filename + size) is contained in a larger folder (curated
`pics/`-style subsets), and JPEG-only folders that sit inside an NEF outing
(darktable export folders, a second camera's sub-folder). Folders whose trip
id matches no region in regions.toml are skipped and listed too. The rest is
processed one at a time -- so GPU memory never exceeds a single `lifer run` --
and an error on one folder does not stop the rest. The plan and the final
summary go to stdout and, in a real run, to LIFER_HOME/runs/run-all-<ts>.log.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .home import find_regions_toml, lifer_home, trip_id_for

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


def is_derived_folder(folder: Path, files: list[Path], root: Path) -> bool:
    """A JPEG-only folder whose own chain up to root holds NEFs: developed
    or curated copies of the outing's shots, not a capture set of their own."""
    if any(f.suffix.lower() == ".nef" for f in files):
        return False
    d = folder
    while d != root:
        if _has_nef(d):
            return True
        d = d.parent
    return False


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

    nd_keys = {d: nonderived_keys(files) for d, files in entries}
    rows: list[Row] = []
    for d, files in entries:
        tid = trip_id_for(d, root)
        region = next((r for r in regions if fnmatch.fnmatch(tid, r["glob"])),
                      None)
        keys = nd_keys[d]
        if is_derived_folder(d, files, root):
            reason = "derived copy (JPEGs of an NEF outing)"
        else:
            sup = next((trip_id_for(g, root) for g, gk in nd_keys.items()
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


def run_all(root_arg: str, dry_run: bool = False, **run_kwargs) -> str:
    """Plan (and unless dry_run, execute) every photo folder under root.

    `run_kwargs` are passed through to pipeline.run_folder (gap_seconds,
    alpha, yolo_weights, imgsz, conf, temperature). Returns the summary line.
    """
    root = Path(root_arg).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    regions = tomllib.loads(find_regions_toml().read_text()).get("region", [])
    rows = build_plan(root, regions)
    plan_text = format_plan(root, rows)
    print(plan_text, flush=True)
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
