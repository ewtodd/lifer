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


# ---------------------------------------------------------------------------
# "Open in darktable" from the review app
# ---------------------------------------------------------------------------

def test_web_open_darktable_reopens_frames_already_handed_off(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("LIFER_HOME", str(home))
    db = home / "data" / "lifer.sqlite"
    make_db(db).close()
    dest = tmp_path / "Pictures" / "TRIP"
    dest.mkdir(parents=True)
    (dest / "_DSC0001.NEF").write_bytes(b"raw")     # handed off on an earlier click
    calls = {}

    def fake_handoff(folder, enc_id=None, db_path=None):
        calls["db_path"] = db_path
        return dest, [], ["_DSC0001.NEF", "_DSC0002.NEF"]   # 0002: raw missing

    monkeypatch.setattr(handoff, "handoff_frames_for", fake_handoff)
    monkeypatch.setattr(handoff, "open_darktable", lambda d: calls.setdefault("opened", d))
    app = web.create_app(db)
    app.config["TESTING"] = True
    r = app.test_client().post("/api/open_darktable", json={"folder": "TRIP"})
    assert r.status_code == 200, r.get_json()
    assert r.get_json() == {"ok": True, "dest": str(dest), "copied": 0,
                            "skipped": 1, "missing": 1}
    assert calls["opened"] == dest
    assert calls["db_path"] == db       # the app's DB, not the default one


def _launch(monkeypatch, tmp_path, dest, running=False):
    """open_darktable with Python standing in for darktable: ``python <dir>``
    runs <dir>/__main__.py, or exits 1 at once when there is none."""
    import sys
    monkeypatch.setenv("LIFER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(handoff.shutil, "which", lambda _name: sys.executable)
    monkeypatch.setattr(handoff, "_darktable_running", lambda: running)
    monkeypatch.setattr(handoff, "LAUNCH_GRACE_S", 0.5)
    return handoff.open_darktable(dest)


def test_open_darktable_reports_an_immediate_failure(monkeypatch, tmp_path):
    dest = tmp_path / "TRIP"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="exited with status 1"):
        _launch(monkeypatch, tmp_path, dest)
    assert (tmp_path / "home" / "darktable.log").is_file()


def test_open_darktable_hand_over_to_running_instance_is_ok(monkeypatch, tmp_path):
    dest = tmp_path / "TRIP"
    dest.mkdir()
    assert _launch(monkeypatch, tmp_path, dest, running=True) is True


def test_open_darktable_still_running_is_ok(monkeypatch, tmp_path):
    dest = tmp_path / "TRIP"
    dest.mkdir()
    (dest / "__main__.py").write_text("import time; time.sleep(2)\n")
    assert _launch(monkeypatch, tmp_path, dest) is True
