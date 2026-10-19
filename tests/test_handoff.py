"""Task 14: `lifer handoff`'s confirmed-encounter resolution and guardrails.

Runs on throwaway LIFER_HOME / DB -- never the real ~/.local/share/lifer or
~/.local/share/lifer, and never under ~/BIRDING.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lifer import handoff
from lifer import web

CAND_HEADER = ("enc_id,time_start,time_end,n_frames,best_frame,frames,det_conf,"
               "raw_top3,adj_top3,possible_lifer,lifer_lookalike,rare_here,"
               "rare_evidence")


def make_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(web.SCHEMA)
    conn.commit()
    return conn


def write_candidates(home: Path, folder: str, rows: list[tuple[str, str, list[str]]]):
    """rows: (enc_id, best_frame, frames)."""
    import csv as _csv

    run_dir = home / "runs" / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "candidates.csv").open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(CAND_HEADER.split(","))
        for enc_id, best, frames in rows:
            w.writerow([
                enc_id, "2026-01-01T00:00:00", "2026-01-01T00:00:05", str(len(frames)),
                best, ",".join(frames), "0.9", "None", "Eastern Screech-Owl (0.90)",
                "False", "", "False", "",
            ])
    (run_dir / "nodet.json").write_text("[]")
    (run_dir / "meta.json").write_text(json.dumps(
        {"trip_id": folder, "folder": str(home / "src")}))


def add_decision(conn, folder, enc_id, frame, frames,
                 action="confirm", species="Eastern Screech-Owl"):
    conn.execute(
        "INSERT INTO decisions (folder, enc_id, best_frame, action, species,"
        " scientific, note, frames, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (folder, enc_id, frame, action, species, "Megascops asio", "",
         json.dumps(frames), "2026-01-01T00:00:00+00:00"))
    conn.commit()


def test_confirmed_encounters_resolves(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("LIFER_HOME", str(home))
    # web's RUNS is a module constant resolved at import; point it at the
    # throwaway home so load_candidates reads the right candidates.csv.
    monkeypatch.setattr(web, "RUNS", home / "runs")
    folder = "2026-01-01-WOODS"
    write_candidates(home, folder, [
        ("enc_0001", "_DSC0001.NEF", ["_DSC0001.NEF", "_DSC0002.NEF"]),
        ("enc_0002", "_DSC0003.NEF", ["_DSC0003.NEF"]),
    ])
    conn = make_db(home / "data" / "lifer.sqlite")
    add_decision(conn, folder, "enc_0001", "_DSC0001.NEF",
                 ["_DSC0001.NEF", "_DSC0002.NEF"])            # confirmed
    add_decision(conn, folder, "enc_0002", "_DSC0003.NEF",
                 ["_DSC0003.NEF"], action="not_bird", species="")  # not a lifer

    encs = handoff.confirmed_encounters(conn, folder)
    assert [e.enc_id for e in encs] == ["enc_0001"]
    assert handoff.representative_frame(conn, folder,
                                        encs[0].frames, encs[0].best_frame) == "_DSC0001.NEF"
    conn.close()


def test_refuses_to_write_under_birding(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("LIFER_HOME", str(home))
    # point the module's BIRDING constant at the throwaway share
    monkeypatch.setattr(handoff, "BIRDING", tmp_path / "BIRDING")
    (tmp_path / "BIRDING").mkdir()
    folder = "2026-01-01-WOODS"
    write_candidates(home, folder, [("enc_0001", "_DSC0001.NEF", ["_DSC0001.NEF"])])
    conn = make_db(home / "data" / "lifer.sqlite")

    dest = tmp_path / "BIRDING" / "2026-01-01-WOODS"
    with pytest.raises(SystemExit):
        handoff.handoff(folder, dest=dest, db_path=home / "data" / "lifer.sqlite")
    assert not dest.exists()
    conn.close()


def test_is_within():
    assert handoff._is_within(Path("/a/b/c"), Path("/a/b"))
    assert not handoff._is_within(Path("/a/b"), Path("/a/b/c"))
    assert handoff._is_within(Path("/x/y"), Path("/x/y"))
