"""lifer CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        prog="lifer",
        description="Find birds in raw photos, identify candidate species, feed a lifelist.",
    )
    p.add_argument("--home", default=None,
                   help="state directory (default $LIFER_HOME or ~/.local/share/lifer)")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the pipeline on one photo folder")
    run.add_argument("folder", help="folder of photos (NEFs, or JPEGs with no NEFs; read-only input)")
    run.add_argument("--root", default=None,
                     help="archive root the trip ID is relative to (default: the "
                          "folder's parent, i.e. the trip ID is the folder name)")
    run.add_argument("--gap", type=float, default=5.0, help="burst gap threshold in seconds (default 5)")
    run.add_argument("--alpha", type=float, default=0.3, help="seasonal-prior exponent (default 0.3)")
    run.add_argument("--yolo", default="yolo11m.pt", help="YOLO weights (default yolo11m.pt)")
    run.add_argument("--imgsz", type=int, default=1280, help="detection input size (default 1280)")
    run.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold (default 0.25)")
    run.add_argument("--temp", type=float, default=3.0,
                     help="BioCLIP temperature: logits/T before softmax (default 3.0)")

    runall = sub.add_parser(
        "run-all",
        help="process every photo folder under a root (the backlog)")
    runall.add_argument("root", help="archive root to scan for photo folders (read-only)")
    runall.add_argument("--dry-run", action="store_true",
                        help="print the plan without running anything")
    runall.add_argument("--gap", type=float, default=5.0, help="burst gap threshold in seconds (default 5)")
    runall.add_argument("--alpha", type=float, default=0.3, help="seasonal-prior exponent (default 0.3)")
    runall.add_argument("--yolo", default="yolo11m.pt", help="YOLO weights (default yolo11m.pt)")
    runall.add_argument("--imgsz", type=int, default=1280, help="detection input size (default 1280)")
    runall.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold (default 0.25)")
    runall.add_argument("--temp", type=float, default=3.0,
                        help="BioCLIP temperature: logits/T before softmax (default 3.0)")

    serve = sub.add_parser("serve", help="serve the review web app (default 127.0.0.1)")
    serve.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8737, help="port (default 8737)")
    serve.add_argument("--db", default="data/lifer.sqlite",
                       help="SQLite state file (default data/lifer.sqlite)")

    publish = sub.add_parser("publish", help="export the public lifelist into a website repo")
    publish.add_argument("--site", required=True, help="path to the Hugo website repo")
    publish.add_argument("--db", default="data/lifer.sqlite",
                         help="SQLite state file (default data/lifer.sqlite)")

    develop = sub.add_parser("develop", help="develop selected raw photos into edits")
    develop.add_argument("folder", help="folder of .NEF files already run (read-only input)")
    develop.add_argument("--root", default=None,
                         help="archive root the trip ID is relative to (default: "
                              "the folder's parent, i.e. the trip ID is the folder name)")
    develop.add_argument("--species", default=None,
                         help="species to pick frames for (adjusted top-1)")
    develop.add_argument("--n", type=int, default=5,
                         help="number of different encounters to develop (default 5)")
    develop.add_argument("--frames", default=None,
                         help="explicit comma-separated frame names (overrides auto-picking)")
    develop.add_argument("--style-sidecar", default=None,
                         help="sidecar .xmp of an edited photo defining the develop "
                              "style template (required)")
    develop.add_argument("--benchmark", action="store_true",
                         help="develop frames from a reference export folder and "
                              "compare lifer vs the reference "
                              "(writes edits/benchmark/compare.html)")
    develop.add_argument("--reference", default=None,
                         help="folder of reference darktable exports "
                              "(required for --benchmark)")
    develop.add_argument("--reference-sidecars", default=None,
                         help="folder with the reference's edited .xmp sidecars, "
                              "for the final crops (default: the --reference folder)")

    style = sub.add_parser("dt-style",
                           help="summarise a darktable develop style from edited sidecars")
    style.add_argument("--sidecars", default=None,
                       help="comma-separated edited sidecar .xmp paths to analyse")
    style.add_argument("--source", default=None,
                       help="folder containing edited .xmp sidecars "
                            "(one of --sidecars/--source is required)")

    migrate = sub.add_parser(
        "migrate",
        help="move the old state dirs into the lifer layout "
             "(default: dry-run, prints the plan and changes nothing)")
    migrate.add_argument("--dry-run", action="store_true",
                         help="print the plan without changing anything (default)")
    migrate.add_argument("--apply", action="store_true",
                         help="move the state dirs and rewrite the old paths")

    args = p.parse_args()
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
    elif args.cmd == "dt-style":
        from .dtstyle import run_dt_style
        run_dt_style(sidecars=args.sidecars, source=args.source)
    elif args.cmd == "migrate":
        from .migrate import run_migrate
        sys.exit(run_migrate(apply=args.apply))


if __name__ == "__main__":
    main()
