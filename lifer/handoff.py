"""``lifer handoff``: hand confirmed encounters' raws + darktable history over.

For each confirmed encounter in a run folder, lifer takes the encounter's
chosen frame (task 12's ``representative_frame``), makes sure it has a ``lifer
develop`` edit, then copies the raw NEF and lifer's darktable history beside
it (``<name>.NEF.xmp``) into the destination (default ``~/Pictures/<folder>``).
This is the one place lifer writes outside LIFER_HOME; the read-only BIRDING
share is never written, and files already in the destination are left alone.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

from PIL import Image

from .boxes import open_db, representative_frame
from .config import load_config
from .decisions import EncRef, parse_frames, resolve_decision
from .develop import develop
from .home import effective_trip_id, lifer_home, source_folder_for_trip
from .web import _decisions_for, _enc_refs, load_candidates, load_nodet

BIRDING = Path.home() / "BIRDING"


def _is_within(child: Path, parent: Path) -> bool:
    """True if ``child`` is inside ``parent`` (both resolved)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def confirmed_encounters(conn: sqlite3.Connection, folder: str) -> list[EncRef]:
    """Current encounter refs whose decision is confirm/change with a species.

    Mirrors ``build_sightings``: decisions are resolved against the *current*
    encounter set (via ``_enc_refs``/``resolve_decision``) so a confirmed
    species tracks the encounter it belongs to now, not the one it was made on.
    """
    cands = load_candidates(folder)
    if not cands:
        return []
    refs = _enc_refs(folder, cands, load_nodet(folder))
    seen: dict[str, EncRef] = {}
    for d in _decisions_for(conn, folder).values():
        if d["action"] not in ("confirm", "change"):
            continue
        if not (d.get("species") or "").strip():
            continue
        enc, _status = resolve_decision(d["best_frame"], parse_frames(d["frames"]), refs)
        if enc is not None:
            seen.setdefault(enc.enc_id, enc)
    return list(seen.values())


def _pixel_diff(a: Path, b: Path) -> float:
    """Mean absolute per-channel difference between two images (resized alike)."""
    import numpy as np

    ia = Image.open(a).convert("RGB")
    ib = Image.open(b).convert("RGB")
    if ia.size != ib.size:
        ia = ia.resize(ib.size, Image.LANCZOS)
    A = np.asarray(ia, dtype=np.float32)
    B = np.asarray(ib, dtype=np.float32)
    return float(np.abs(A - B).mean())


def _verify(dest: Path, folder: str, frames: list[str]) -> list[tuple[str, str, str]]:
    """Render each handoff sidecar with darktable-cli and diff vs lifer's edit.

    ``darktable-cli <nef> <xmp> <tmp> --core --library :memory:
    --conf write_sidecar_files=never`` must reproduce lifer's develop output
    (the edit jpg); a large diff means the sidecar does not describe the edit.
    """
    out_dir = lifer_home() / "edits" / folder
    results: list[tuple[str, str, str]] = []
    for frame in frames:
        stem = Path(frame).stem
        edit = out_dir / f"{stem}.jpg"
        tmp = dest / f".verify-{stem}.jpg"
        cmd = ["darktable-cli", str(dest / frame), str(dest / f"{frame}.xmp"),
               str(tmp), "--core", "--library", ":memory:",
               "--conf", "write_sidecar_files=never"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.is_file():
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            results.append((frame, "render-failed", detail[-1] if detail else "?"))
            continue
        diff = _pixel_diff(tmp, edit)
        tmp.unlink(missing_ok=True)
        results.append((frame, f"ok  mae={diff:.2f}", ""))
    return results


def handoff(folder_arg: str, dest: Path | None = None, db_path: Path | None = None,
            style_sidecar: Path | None = None,
            root: str | Path | None = None) -> int:
    """Hand each confirmed encounter's chosen frame (NEF + .NEF.xmp) to ``dest``."""
    home = lifer_home()
    conn = open_db(db_path)

    arg = Path(folder_arg).expanduser()
    if arg.is_dir():
        folder = effective_trip_id(arg, root)
        source = arg
    else:
        # a trip id (folder basename under the photo root) rather than a path
        folder = folder_arg
        source = source_folder_for_trip(folder)

    if dest is None:
        dest = Path.home() / "Pictures" / folder
    dest = dest.expanduser()

    # The BIRDING share is read-only for this project; never write under it
    # (the share is also where `source` lives, so a source-nest is caught too).
    if _is_within(dest, BIRDING) or _is_within(dest, source):
        raise SystemExit(
            f"error: refusing to write handoff into {dest} "
            "(under the read-only photo share / source folder)")
    if dest.is_file():
        raise SystemExit(f"error: destination {dest} is a file, not a folder")

    encs = confirmed_encounters(conn, folder)
    if not encs:
        print(f"no confirmed encounters in {folder}")
        return 0

    # chosen frame per confirmed encounter (task 12), deduped
    chosen: list[str] = []
    for enc in encs:
        frame = representative_frame(conn, folder, enc.frames, enc.best_frame)
        if frame not in chosen:
            chosen.append(frame)

    # make sure each chosen frame has a lifer develop edit (run it if missing)
    out_dir = home / "edits" / folder
    xmp_dir = out_dir / "xmp"
    missing = [f for f in chosen if not (out_dir / f"{Path(f).stem}.jpg").is_file()]
    if missing:
        style = style_sidecar
        if style is None:
            raw = (load_config().get("develop", {}) or {}).get("style_sidecar")
            style = Path(raw) if isinstance(raw, (str, Path)) else None
        if not style:
            print("error: %d frame(s) need `lifer develop` but no style sidecar "
                  "(--style-sidecar or [develop] style_sidecar in config.toml)" % len(missing))
            return 2
        print(f"developing {len(missing)} frame(s): {', '.join(missing)}")
        develop(str(source), frames=list(missing),
                style_sidecar=Path(style), root=root, conn=conn)

    # copy NEF + sidecar, never overwriting what's already there
    copied: list[str] = []
    skipped: list[str] = []
    for frame in chosen:
        dest_nef = dest / frame
        dest_xmp = dest / f"{frame}.xmp"
        if dest_nef.exists() or dest_xmp.exists():
            skipped.append(frame)
            continue
        src_nef = source / frame
        if not src_nef.is_file():
            skipped.append(frame)
            continue
        src_xmp = xmp_dir / f"{Path(frame).stem}.xmp"
        if not src_xmp.is_file():
            skipped.append(frame)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_nef, dest_nef)
        shutil.copy2(src_xmp, dest_xmp)
        copied.append(frame)

    print(f"\nhandoff: {len(copied)} copied, {len(skipped)} skipped, "
          f"{len(missing)} developed now")
    if skipped:
        print(f"  skipped (already in {dest} or unusable): {', '.join(skipped)}")
    if copied:
        print(f"  -> {dest}")
        print("  import into darktable: File > Import, point at the folder above; "
              "each RAW's edit history is in its '<name>.NEF.xmp' sidecar.")

    rows = _verify(dest, folder, copied)
    if rows:
        print("  verification (darktable-cli render vs lifer edit, mean abs diff):")
        for frame, status, detail in rows:
            print(f"    {frame}: {status}")
            if detail:
                print(f"      {detail}")
        worst = max((r[1] for r in rows), key=lambda s: 0.0 if "mae=" not in s
                    else float(s.split("mae=")[1]))
        if "mae=" in worst:
            val = float(worst.split("mae=")[1])
            print(f"  (worst diff {val:.2f}; a matching sidecar is near 0)")
    return 0
