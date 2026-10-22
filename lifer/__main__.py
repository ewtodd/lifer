"""lifer CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


# Shell-completion hints, set as `.complete` on argparse actions and read by
# shtab when the package build generates the completion scripts
# (lifer/completion.py). Plain dicts, so the CLI itself never imports shtab.
DIRS = {"bash": "_shtab_compgen_dirs", "zsh": "_files -/"}
FILES = {"bash": "_shtab_compgen_files", "zsh": "_files"}
TRIPS = {"bash": "_lifer_trips", "zsh": "_lifer_trips"}
TRIPS_OR_DIRS = {"bash": "_lifer_trips_or_dirs", "zsh": "_lifer_trips_or_dirs"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lifer",
        description="Find birds in raw photos, identify candidate species, feed a lifelist.",
    )
    p.add_argument("--home", default=None,
                   help="state directory (default $LIFER_HOME or ~/.local/share/lifer)"
                   ).complete = DIRS
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the pipeline on one photo folder")
    run.add_argument("folder", help="folder of photos (NEFs, or JPEGs with no NEFs; read-only input)"
                     ).complete = DIRS
    run.add_argument("--root", default=None,
                     help="archive root the trip ID is relative to (default: the "
                          "folder's parent, i.e. the trip ID is the folder name)"
                     ).complete = DIRS
    run.add_argument("--gap", type=float, default=5.0, help="burst gap threshold in seconds (default 5)")
    run.add_argument("--alpha", type=float, default=0.3, help="seasonal-prior exponent (default 0.3)")
    run.add_argument("--yolo", default="yolo11m.pt", help="YOLO weights (default yolo11m.pt)"
                     ).complete = FILES
    run.add_argument("--imgsz", type=int, default=1280, help="detection input size (default 1280)")
    run.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold (default 0.25)")
    run.add_argument("--temp", type=float, default=3.0,
                     help="BioCLIP temperature: logits/T before softmax (default 3.0)")

    runall = sub.add_parser(
        "run-all",
        help="process every photo folder under a root (the backlog)")
    runall.add_argument("root", help="archive root to scan for photo folders (read-only)"
                        ).complete = DIRS
    runall.add_argument("--dry-run", action="store_true",
                        help="print the plan without running anything")
    runall.add_argument("--gap", type=float, default=5.0, help="burst gap threshold in seconds (default 5)")
    runall.add_argument("--alpha", type=float, default=0.3, help="seasonal-prior exponent (default 0.3)")
    runall.add_argument("--yolo", default="yolo11m.pt", help="YOLO weights (default yolo11m.pt)"
                        ).complete = FILES
    runall.add_argument("--imgsz", type=int, default=1280, help="detection input size (default 1280)")
    runall.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold (default 0.25)")
    runall.add_argument("--temp", type=float, default=3.0,
                        help="BioCLIP temperature: logits/T before softmax (default 3.0)")
    runall.add_argument("--apply-migrations", action="store_true",
                        help="after the trips have run, apply the task-24 trip "
                             "migrations (rekey RAW-bucket trips into their "
                             "parent, map derived-copy decisions onto the raw "
                             "frames)")

    migr = sub.add_parser(
        "migrate-trips",
        help="rekey trip state after the task-24 rules "
             "(dry run without --apply)")
    migr.add_argument("--root", default=str(Path.home() / "BIRDING"),
                      help="archive root the current plan is built from "
                           "(default: ~/BIRDING)").complete = DIRS
    migr.add_argument("--apply", action="store_true",
                      help="write the changes (default: print the plan only)")
    migr.add_argument("--db", default=None,
                      help="decisions DB (default: LIFER_HOME/data/lifer.sqlite)"
                      ).complete = FILES

    delete = sub.add_parser(
        "delete",
        help="remove a trip's run dir, cache, and DB rows "
             "(dry run without --apply)")
    delete.add_argument("trips", nargs="+", help="trip ID(s) to delete").complete = TRIPS
    delete.add_argument("--apply", action="store_true",
                        help="delete (default: print what would be deleted)")
    delete.add_argument("--db", default=None,
                        help="decisions DB (default: LIFER_HOME/data/lifer.sqlite)"
                        ).complete = FILES

    mdec = sub.add_parser(
        "migrate-decisions",
        help="backfill decision frame lists and re-attach orphaned decisions "
             "after a re-detection (task 25; dry run without --apply)")
    mdec.add_argument("--db", default=None,
                      help="decisions DB (default: LIFER_HOME/data/lifer.sqlite)"
                      ).complete = FILES
    mdec.add_argument("--apply", action="store_true",
                      help="write the changes (default: print the plan only)")

    serve = sub.add_parser("serve", help="serve the review web app (default 127.0.0.1)")
    serve.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8737, help="port (default 8737)")
    serve.add_argument("--db", default="data/lifer.sqlite",
                       help="SQLite state file (default data/lifer.sqlite)").complete = FILES

    publish = sub.add_parser("publish", help="export the public lifelist into a website repo")
    publish.add_argument("--site", required=True, help="path to the Hugo website repo"
                         ).complete = DIRS
    publish.add_argument("--db", default="data/lifer.sqlite",
                         help="SQLite state file (default data/lifer.sqlite)").complete = FILES

    develop = sub.add_parser("develop", help="develop selected raw photos into edits")
    develop.add_argument("folder", help="folder of .NEF files already run (read-only input)"
                         ).complete = DIRS
    develop.add_argument("--root", default=None,
                         help="archive root the trip ID is relative to (default: "
                              "the folder's parent, i.e. the trip ID is the folder name)"
                         ).complete = DIRS
    develop.add_argument("--species", default=None,
                         help="species to pick frames for (adjusted top-1)")
    develop.add_argument("--n", type=int, default=5,
                         help="number of different encounters to develop (default 5)")
    develop.add_argument("--frames", default=None,
                         help="explicit comma-separated frame names (overrides auto-picking)")
    develop.add_argument("--style-sidecar", default=None,
                         help="sidecar .xmp of an edited photo defining the develop "
                              "style template (required)").complete = FILES
    develop.add_argument("--benchmark", action="store_true",
                         help="develop frames from a reference export folder and "
                              "compare lifer vs the reference "
                              "(writes edits/benchmark/compare.html)")
    develop.add_argument("--reference", default=None,
                         help="folder of reference darktable exports "
                              "(required for --benchmark)").complete = DIRS
    develop.add_argument("--reference-sidecars", default=None,
                         help="folder with the reference's edited .xmp sidecars, "
                              "for the final crops (default: the --reference folder)"
                         ).complete = DIRS

    handoff = sub.add_parser(
        "handoff",
        help="hand confirmed encounters' raws + darktable history to the owner")
    handoff.add_argument("folder",
                         help="a run folder (dir of .NEFs) or a trip ID "
                              "whose confirmed encounters to hand off"
                         ).complete = TRIPS_OR_DIRS
    handoff.add_argument("--dest", default=None,
                         help="destination folder (default: ~/Pictures/<folder>); "
                              "never under the read-only BIRDING share").complete = DIRS
    handoff.add_argument("--db", default=None,
                         help="decisions/override DB (default: LIFER_HOME data/lifer.sqlite)"
                         ).complete = FILES
    handoff.add_argument("--style-sidecar", default=None,
                         help="sidecar .xmp defining the develop style, used when "
                              "a chosen frame has no edit yet "
                              "(default: [develop].style_sidecar in config.toml)"
                         ).complete = FILES
    handoff.add_argument("--root", default=None,
                         help="archive root the trip ID is relative to").complete = DIRS

    style = sub.add_parser("dt-style",
                           help="summarise a darktable develop style from edited sidecars")
    style.add_argument("--sidecars", default=None,
                       help="comma-separated edited sidecar .xmp paths to analyse"
                       ).complete = FILES
    style.add_argument("--source", default=None,
                       help="folder containing edited .xmp sidecars "
                            "(one of --sidecars/--source is required)").complete = DIRS

    migrate = sub.add_parser(
        "migrate",
        help="move the old state dirs into the lifer layout "
             "(default: dry-run, prints the plan and changes nothing)")
    migrate.add_argument("--dry-run", action="store_true",
                         help="print the plan without changing anything (default)")
    migrate.add_argument("--apply", action="store_true",
                         help="move the state dirs and rewrite the old paths")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.home:
        # set before importing subcommand modules: they read LIFER_HOME
        os.environ["LIFER_HOME"] = str(Path(args.home).expanduser().resolve())
    if args.cmd == "run":
        # imported here so `serve` stays torch-free (no GPU needed to review)
        from .pipeline import run_folder
        run_folder(
            args.folder,
            root=args.root,
            gap_seconds=args.gap,
            alpha=args.alpha,
            yolo_weights=args.yolo,
            imgsz=args.imgsz,
            conf=args.conf,
            temperature=args.temp,
        )
    elif args.cmd == "run-all":
        from .runall import run_all
        run_all(
            args.root,
            dry_run=args.dry_run,
            apply_migrations=args.apply_migrations,
            gap_seconds=args.gap,
            alpha=args.alpha,
            yolo_weights=args.yolo,
            imgsz=args.imgsz,
            conf=args.conf,
            temperature=args.temp,
        )
    elif args.cmd == "serve":
        from .web import serve_app
        serve_app(host=args.host, port=args.port, db_path=Path(args.db))
    elif args.cmd == "publish":
        from .publish import publish
        info = publish(site=Path(args.site), db_path=Path(args.db))
        print(f"published {info['species_count']} species "
              f"({info['with_photos']} with photos, {info['photos']} photos) "
              f"-> {Path(args.site).resolve()}/data/birds.json")
        from .gallery import publish_gallery
        g = publish_gallery(site=Path(args.site), db_path=Path(args.db))
        print(f"gallery {g['in_gallery']} in gallery "
              f"({g['rated_not_exported']} rated-not-exported, "
              f"{g['rated_not_confirmed']} rated-not-confirmed) "
              f"-> {Path(args.site).resolve()}/data/gallery.json")
    elif args.cmd == "develop":
        # imported here so `run`/`serve` stay torch-free; develop only needs
        # rawpy/PIL and darktable-cli on PATH
        from .develop import benchmark, develop
        frames = None
        if args.frames:
            frames = [f.strip() for f in args.frames.split(",") if f.strip()]
        if not args.style_sidecar:
            raise SystemExit("error: --style-sidecar is required "
                             "(a sidecar .xmp of an edited photo)")
        style_sidecar = Path(args.style_sidecar).expanduser()
        if args.benchmark:
            if not args.reference:
                raise SystemExit("error: --reference is required with --benchmark "
                                 "(folder of the reference darktable exports)")
            ref_dir = Path(args.reference).expanduser()
            ref_sidecars = (Path(args.reference_sidecars).expanduser()
                            if args.reference_sidecars else ref_dir)
            benchmark(args.folder, reference_dir=ref_dir, frames=frames,
                      style_sidecar=style_sidecar, reference_sidecars=ref_sidecars,
                      root=args.root)
        elif args.species:
            develop(args.folder, species=args.species, n=args.n, frames=frames,
                    style_sidecar=style_sidecar, root=args.root)
        else:
            develop(args.folder, frames=frames, style_sidecar=style_sidecar,
                    root=args.root)
    elif args.cmd == "handoff":
        from .handoff import handoff
        sys.exit(handoff(
            args.folder,
            dest=Path(args.dest).expanduser() if args.dest else None,
            db_path=Path(args.db).expanduser() if args.db else None,
            style_sidecar=(Path(args.style_sidecar).expanduser()
                           if args.style_sidecar else None),
            root=args.root))
    elif args.cmd == "dt-style":
        from .dtstyle import run_dt_style
        run_dt_style(sidecars=args.sidecars, source=args.source)
    elif args.cmd == "migrate":
        from .migrate import run_migrate
        sys.exit(run_migrate(apply=args.apply))
    elif args.cmd == "migrate-trips":
        _cmd_migrate_trips(args)
    elif args.cmd == "migrate-decisions":
        _cmd_migrate_decisions(args)
    elif args.cmd == "delete":
        _cmd_delete(args)


def _open_decisions_db(db_arg: str | None):
    """The decisions DB for migrate-trips/delete (default: LIFER_HOME)."""
    from .home import lifer_home
    import sqlite3
    path = (Path(db_arg).expanduser() if db_arg
            else lifer_home() / "data" / "lifer.sqlite")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return path, conn


def _validate_trip_ids(trips: list[str]) -> None:
    for t in trips:
        if not t or "/" in t or t in (".", "..") or t.startswith("-"):
            raise SystemExit(f"invalid trip ID: {t!r}")


def _cmd_migrate_trips(args) -> None:
    from .runall import build_plan
    import tomllib
    from .home import find_regions_toml, lifer_home
    from .trips_ops import (apply_migrate_trips, migration_summary,
                            plan_migrate_trips)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    regions = tomllib.loads(find_regions_toml().read_text()).get("region", [])
    rows = build_plan(root, regions)
    home = lifer_home()
    db_path, conn = _open_decisions_db(args.db)
    try:
        if args.apply:
            res = apply_migrate_trips(home, conn, root, rows)
            print(f"migrate-trips (applied, db: {db_path})")
            print(migration_summary(res))
        else:
            res = plan_migrate_trips(home, conn, root, rows)
            print(f"migrate-trips (dry run, db: {db_path}; "
                  f"re-run with --apply to write the changes)")
            n = len(res["rekey"]) + len(res["map"])
            if not n and not res["unmapped"] and not res["conflicts"]:
                print("nothing to migrate")
            for e in res["rekey"]:
                print(f"  rekey {e['from']} -> {e['to']}: "
                      f"{e['decisions']} decisions, "
                      f"{e['crop_overrides']} crop fixes, "
                      f"{e['chosen_frames']} chosen frames")
                for c in e["conflicts"]:
                    print(f"    conflict (left in place): {c}")
                for m in e["mismatched"]:
                    print(f"    no longer matches {e['to']}'s candidates: {m}")
            for m in res["map"]:
                print(f"  map {m['from']}/{m['enc_id']} {m['frame']} "
                      f"-> {m['to_trip']}/{m['to_enc_id']} {m['to_frame']} "
                      f"({m['action']}{', ' + m['species'] if m['species'] else ''})")
            for u in res["unmapped"]:
                print(f"  unmapped (left in place): {u['trip']}/{u['enc_id']} "
                      f"{u['frame']} -- {u['reason']}")
            for c in res["conflicts"]:
                print(f"  conflict (left in place): {c['trip']}/{c['enc_id']} "
                      f"{c['frame']} -- {c['reason']}")
            if n:
                print(f"total: {n} to migrate, {len(res['unmapped'])} unmapped, "
                      f"{len(res['conflicts'])} conflicts")
    finally:
        conn.close()


def _cmd_delete(args) -> None:
    _validate_trip_ids(args.trips)
    from .home import lifer_home
    from .trips_ops import apply_delete, plan_delete

    home = lifer_home()
    db_path, conn = _open_decisions_db(args.db)
    try:
        if args.apply:
            res = apply_delete(home, conn, args.trips)
            print(f"deleted (db: {db_path}):")
            for t in res["trips"]:
                print(f"  {t['trip']}: {t['run_files']} run files, "
                      f"{t['cache_files']} cache files, {t['decisions']} "
                      f"decisions, {t['crop_overrides']} crop fixes, "
                      f"{t['chosen_frames']} chosen frames")
        else:
            res = plan_delete(home, conn, args.trips)
            print(f"delete (dry run, db: {db_path}; re-run with --apply):")
            for t in res["trips"]:
                print(f"  {t['trip']}: would remove {t['run_files']} run files, "
                      f"{t['cache_files']} cache files, {t['decisions']} "
                      f"decisions, {t['crop_overrides']} crop fixes, "
                      f"{t['chosen_frames']} chosen frames")
            print("  nothing under the photo archive is ever touched")
    finally:
        conn.close()


def _cmd_migrate_decisions(args) -> None:
    """Task 25: backfill decision frame lists, re-attach orphaned decisions."""
    from .decisions import apply_migrate_decisions, plan_migrate_decisions
    from .home import lifer_home

    home = lifer_home()
    db_path, conn = _open_decisions_db(args.db)
    try:
        if args.apply:
            res = apply_migrate_decisions(home, conn)
            print(f"migrate-decisions (applied, db: {db_path})")
        else:
            res = plan_migrate_decisions(home, conn)
            print(f"migrate-decisions (dry run, db: {db_path}; "
                  f"re-run with --apply to write the changes)")
        _print_migrate_decisions(res)
    finally:
        conn.close()


def _print_migrate_decisions(res: dict[str, Any]) -> None:
    if not res["n_decisions"]:
        print("no decisions in the DB")
        return
    by_folder: dict[str, list[dict[str, Any]]] = {}
    for e in res["entries"]:
        by_folder.setdefault(e["folder"], []).append(e)
    for folder in sorted(by_folder):
        entries = by_folder[folder]
        n_reatt = sum(1 for e in entries if e["changed"])
        n_backfill = sum(1 for e in entries if not e["had_frames"])
        print(f"  {folder}: {len(entries)} decision(s) to update "
              f"({n_reatt} reattached, {n_backfill} frame-list backfill)")
        for e in entries:
            if e["conflict"]:
                print(f"    CONFLICT (left in place): {e['enc_id']} {e['frame']} "
                      f"-> {e['new_enc_id']} {e['new_frame']} already occupied")
            elif e["changed"]:
                print(f"    reattach {e['enc_id']} {e['frame']} "
                      f"-> {e['new_enc_id']} {e['new_frame']} "
                      f"({e['action']} {e['species'] or '-'}, "
                      f"{len(e['new_frames'])} frames)")
            else:
                print(f"    backfill {e['enc_id']} {e['frame']} "
                      f"({e['action']} {e['species'] or '-'}, "
                      f"{len(e['new_frames'])} frames)")
    if res["ambiguous"]:
        print(f"  {len(res['ambiguous'])} ambiguous (left in place, no guessing):")
        for a in res["ambiguous"]:
            print(f"    {a['folder']}/{a['enc_id']} {a['frame']} "
                  f"({a['action']} {a['species'] or '-'}) -- {a['reason']}")
    total = len(res["entries"])
    applied = res.get("applied")
    if applied is not None:
        print(f"total: {res['n_decisions']} decisions, {res['ok']} already attached, "
              f"{applied} applied, {len(res['ambiguous'])} ambiguous")
    else:
        print(f"total: {res['n_decisions']} decisions, {res['ok']} already attached, "
              f"{total} to update, {len(res['ambiguous'])} ambiguous")


if __name__ == "__main__":
    main()
