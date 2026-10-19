"""One-shot migration from the birdid state layout to the lifer layout.

The import name of the CLI and the home/cache dirs change (birdid -> lifer).
This command moves the existing state so nothing is lost:

- move    ~/.local/share/birdid  -> ~/.local/share/lifer
- move    ~/.cache/birdid        -> ~/.cache/lifer
- rename  <data>/birdid.sqlite   -> <data>/lifer.sqlite
- rewrite any absolute path containing the old directory to the new one
  everywhere it is stored (SQLite text columns, cache JSON, run outputs), so
  nothing points at the old location after the move.

The old state dirs are discovered the way the old code found them ($BIRDID_HOME,
then the XDG defaults), so a custom BIRDID_HOME migrates too; the new dirs are
found the way the new code finds them ($LIFER_HOME / XDG defaults). Paths
resolve against $HOME, so ``migrate --apply`` in a throwaway HOME works.

``--dry-run`` (the default) prints exactly what would happen and changes
nothing. ``--apply`` performs it and refuses if a target directory already
exists.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

# text-like files whose contents may hold absolute paths worth rewriting
_TEXT_SUFFIXES = {".json", ".csv", ".html", ".txt", ".toml", ".xml", ".xmp", ".md", ".ini"}


def _old_data() -> Path:
    h = os.environ.get("BIRDID_HOME")  # a custom old home, if any
    if h:
        return Path(h).expanduser()
    x = os.environ.get("XDG_DATA_HOME")
    base = Path(x).expanduser() if x else Path.home() / ".local" / "share"
    return base / "birdid"


def _new_data() -> Path:
    h = os.environ.get("LIFER_HOME")
    if h:
        return Path(h).expanduser()
    x = os.environ.get("XDG_DATA_HOME")
    base = Path(x).expanduser() if x else Path.home() / ".local" / "share"
    return base / "lifer"


def _old_cache() -> Path:
    x = os.environ.get("XDG_CACHE_HOME")
    base = Path(x).expanduser() if x else Path.home() / ".cache"
    return base / "birdid"


def _new_cache() -> Path:
    x = os.environ.get("XDG_CACHE_HOME")
    base = Path(x).expanduser() if x else Path.home() / ".cache"
    return base / "lifer"


def _show(p: Path) -> str:
    """Render a path with the home dir abbreviated to ~."""
    home = str(Path.home())
    s = str(p)
    return ("~" + s[len(home):]) if s.startswith(home) else s


def _rewrite_text(path: Path, pairs: list[tuple[str, str]]) -> int:
    """Replace old -> new path strings in a text file. Return 1 if written."""
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return 0
    try:
        data = path.read_bytes()
    except OSError:
        return 0
    if b"\x00" in data[:8192]:
        return 0  # binary; not a text file
    ndata = data
    for old, new in pairs:
        ndata = ndata.replace(old.encode(), new.encode())
    if ndata != data:
        path.write_bytes(ndata)
        return 1
    return 0


def _rewrite_sqlite(path: Path, pairs: list[tuple[str, str]]) -> int:
    """Replace old -> new path strings in every TEXT column. Return rows touched."""
    total = 0
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error:
        return 0
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
            for c in cols:
                for old, new in pairs:
                    cur = conn.execute(
                        f"UPDATE \"{t}\" SET \"{c}\" = REPLACE(\"{c}\", ?, ?) "
                        f"WHERE INSTR(\"{c}\", ?) > 0",
                        (old, new, old),
                    )
                    total += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return total


def _count_sqlite(path: Path, pairs: list[tuple[str, str]]) -> int:
    """Count (without modifying) the DB rows whose TEXT columns hold an old path."""
    total = 0
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return 0
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
            for c in cols:
                for old, _ in pairs:
                    n = conn.execute(
                        f"SELECT COUNT(*) FROM \"{t}\" WHERE INSTR(\"{c}\", ?) > 0",
                        (old,),
                    ).fetchone()[0]
                    total += n
    finally:
        conn.close()
    return total


def _plan_counts(base: Path, pairs: list[tuple[str, str]]) -> tuple[int, int]:
    """(files, db_rows) that would be rewritten under a not-yet-moved base."""
    files = db_rows = 0
    if not base.exists():
        return 0, 0
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".sqlite":
            db_rows += _count_sqlite(path, pairs)
        else:
            if path.suffix.lower() in _TEXT_SUFFIXES:
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                if b"\x00" in data[:8192]:
                    continue
                if any(old.encode() in data for old, _ in pairs):
                    files += 1
    return files, db_rows


def _apply(base: Path, pairs: list[tuple[str, str]]) -> tuple[int, int]:
    """Rewrite under an already-moved base. Return (files, db_rows)."""
    files = db_rows = 0
    if not base.exists():
        return 0, 0
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".sqlite":
            db_rows += _rewrite_sqlite(path, pairs)
        else:
            files += _rewrite_text(path, pairs)
    return files, db_rows


def run_migrate(apply: bool = False) -> int:
    old_data, new_data = _old_data(), _new_data()
    old_cache, new_cache = _old_cache(), _new_cache()
    pairs = [(str(old_data), str(new_data)), (str(old_cache), str(new_cache))]

    moves = []
    if old_data.exists():
        moves.append((old_data, new_data))
    if old_cache.exists():
        moves.append((old_cache, new_cache))

    if not moves:
        print("migrate: nothing to do; no birdid state directories found")
        return 0

    # the DB lives at <old_data>/data/birdid.sqlite before the move (for the
    # dry-run plan) and at <new_data>/data/birdid.sqlite after it (to rename)
    has_db = (old_data / "data" / "birdid.sqlite").is_file()
    db_dst = new_data / "data" / "lifer.sqlite"

    if apply:
        # refuse to clobber an existing target
        for _src, dst in moves:
            if dst.exists():
                print(f"migrate: refusing - target already exists: {_show(dst)}")
                return 1

    print("migrate: " + ("APPLY" if apply else "dry-run (no changes made)"))
    print(f"  move    {_show(old_data)} -> {_show(new_data)}"
          if old_data.exists() else f"  (no data dir at {_show(old_data)})")
    print(f"  move    {_show(old_cache)} -> {_show(new_cache)}"
          if old_cache.exists() else f"  (no cache dir at {_show(old_cache)})")
    if has_db:
        print("  rename  data/birdid.sqlite -> data/lifer.sqlite")

    # count rewrites (the apply source and the dry-run source hold the same bytes)
    counts = []
    for _src, _dst in moves:
        counts.append(_plan_counts(_src, pairs))
    files = sum(c[0] for c in counts)
    rows = sum(c[1] for c in counts)
    print(f"  rewrite {files} file(s) and {rows} DB row(s) that reference the old path")

    if not apply:
        return 0

    for src, dst in moves:
        if src != dst and src.exists():
            shutil.move(str(src), str(dst))
    db_moved = new_data / "data" / "birdid.sqlite"
    if db_moved.is_file():
        db_moved.rename(db_dst)

    applied = 0, 0
    for dst, _ in moves:
        applied = tuple(a + b for a, b in zip(applied, _apply(dst, pairs)))
    print(f"migrate: done - moved {len(moves)} dir(s), rewrote {applied[0]} file(s) "
          f"and {applied[1]} DB row(s)")
    return 0
