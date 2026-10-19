"""Task 25: decisions attach to encounters by frame overlap, not by enc_id.

- best-frame selection is stable when sharpness differences are tiny
- candidates.csv / nodet.json carry the encounter's frame list
- the web app resolves decisions by frame overlap (review, trips, sightings)
- `migrate-decisions` backfills frame lists and re-attaches orphaned
  decisions (like the enc_0014 American Golden-Plover), and never guesses
  on ambiguous matches
"""

import csv
import json
import sqlite3
from datetime import datetime
from typing import Any, cast

import pytest

import lifer.web as web
from lifer.bursts import FrameInfo, group_into_encounters
from lifer.decisions import (EncRef, apply_migrate_decisions,
                             encounters_from_cache, ensure_frames_column,
                             parse_frames, plan_migrate_decisions,
                             resolve_decision)
from lifer.outputs import Candidate, write_candidates_csv


def _frame(name: str, t: int, sharp: float) -> FrameInfo:
    return FrameInfo(name=name,
                     timestamp=datetime(2026, 1, 1, 12, 0, t),
                     n_dets=1 if sharp > 0 else 0,
                     best_conf=0.9, sharpness=sharp)


# ---------------------------------------------------------------------------
# best-frame determinism
# ---------------------------------------------------------------------------

def test_best_frame_stable_on_tiny_sharpness_difference():
    # the enc_0014 case: a 0.04% sharpness difference must not flip the
    # best frame between re-detections
    a = _frame("_DSC7500.NEF", 0, 1685.9126549502835)
    b = _frame("_DSC7507.NEF", 4, 1686.63934160996)
    e1 = group_into_encounters([a, b])[0]
    e2 = group_into_encounters([b, a])[0]
    assert e1.best_frame.name == e2.best_frame.name
    # near-tie: filename decides, not the 0.04% difference
    assert e1.best_frame.name == "_DSC7507.NEF"


def test_best_frame_real_difference_wins():
    blurry = _frame("_DSC0001.NEF", 0, 1000.0)
    sharp = _frame("_DSC0002.NEF", 1, 1700.0)
    enc = group_into_encounters([blurry, sharp])[0]
    assert enc.best_frame.name == "_DSC0002.NEF"


def test_grouping_stable_with_duplicate_timestamps():
    a = _frame("b.NEF", 0, 500.0)
    b = _frame("a.NEF", 0, 400.0)
    e1 = group_into_encounters([a, b])[0]
    e2 = group_into_encounters([b, a])[0]
    assert [f.name for f in e1.frames] == [f.name for f in e2.frames] == ["a.NEF", "b.NEF"]


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def _encs() -> list[EncRef]:
    return [
        EncRef("enc_0001", "_DSC0002.NEF", ["_DSC0001.NEF", "_DSC0002.NEF"]),
        EncRef("enc_0002", "_DSC0005.NEF", ["_DSC0004.NEF", "_DSC0005.NEF"]),
    ]


def test_resolve_by_best_frame():
    enc, status = resolve_decision("_DSC0001.NEF", [], _encs())
    assert enc is not None and enc.enc_id == "enc_0001"
    assert status == "best-frame"


def test_resolve_by_overlap():
    enc, status = resolve_decision(
        "_DSC9999.NEF", ["_DSC0004.NEF", "_DSC0005.NEF", "_DSC0006.NEF"], _encs())
    assert enc is not None and enc.enc_id == "enc_0002"
    assert status == "overlap"


def test_resolve_ambiguous_tie():
    # one frame shared with each of two encounters: no guessing
    enc, status = resolve_decision("_DSC9999.NEF",
                                   ["_DSC0001.NEF", "_DSC0004.NEF"], _encs())
    assert enc is None and status == "ambiguous"


def test_resolve_none():
    enc, status = resolve_decision("_DSC9999.NEF", [], _encs())
    assert enc is None and status == "none"


def test_parse_frames():
    assert parse_frames("") == []
    assert parse_frames(None) == []
    assert parse_frames('["_DSC1.NEF", "_DSC2.NEF"]') == ["_DSC1.NEF", "_DSC2.NEF"]
    assert parse_frames("not json") == []


def test_ensure_frames_column_idempotent():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE decisions (folder TEXT, enc_id TEXT, best_frame TEXT,"
        " action TEXT, species TEXT, scientific TEXT, note TEXT,"
        " updated_at TEXT, PRIMARY KEY (folder, enc_id, best_frame))")
    ensure_frames_column(conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(decisions)")]
    assert "frames" in cols
    ensure_frames_column(conn)  # second call is a no-op
    conn.close()


# ---------------------------------------------------------------------------
# migrate-decisions
# ---------------------------------------------------------------------------

def _make_home(tmp_path):
    """A trip whose re-detection moved the best frame (the enc_0014 case).

    enc_0001: _DSC0001 (old best, sharpness 1685.91) + _DSC0002 (new best,
    1686.64) -- a 0.04% difference; enc_0002: _DSC0005.
    candidates.csv is the OLD format (no frames column), as written before
    this change; the per-frame cache is complete.
    """
    run = tmp_path / "runs" / "TRIP"
    (run / "crops").mkdir(parents=True)
    (run / "candidates.csv").write_text(
        "enc_id,time_start,time_end,n_frames,best_frame,det_conf,raw_top3,"
        "adj_top3,possible_lifer,lifer_lookalike,rare_here,rare_evidence\n"
        "enc_0001,2026-01-01 12:00:00,2026-01-01 12:00:04,2,_DSC0002.NEF,0.9,"
        "Sparrow (0.100),Sparrow (0.100),False,,False,\n"
        "enc_0002,2026-01-01 12:00:30,2026-01-01 12:00:30,1,_DSC0005.NEF,0.9,"
        "Sparrow (0.100),Sparrow (0.100),False,,False,\n")
    cache = tmp_path / "cache" / "TRIP"
    cache.mkdir(parents=True)
    for name, t, sharp in [("_DSC0001.NEF", 0, 1685.9126549502835),
                           ("_DSC0002.NEF", 4, 1686.63934160996),
                           ("_DSC0005.NEF", 30, 2000.0)]:
        (cache / f"{name[:-4]}.json").write_text(json.dumps({
            "name": name, "timestamp": f"2026-01-01T12:00:{t:02d}",
            "sharpness": sharp, "dets": [[0, 0, 100, 100, 0.9, "Sparrow"]]}))
    return tmp_path


def _make_db(path, decisions) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE decisions (folder TEXT NOT NULL, enc_id TEXT NOT NULL,"
        " best_frame TEXT NOT NULL, action TEXT NOT NULL,"
        " species TEXT NOT NULL DEFAULT '', scientific TEXT NOT NULL DEFAULT '',"
        " note TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,"
        " PRIMARY KEY (folder, enc_id, best_frame))")
    for (folder, enc_id, frame, action, species) in decisions:
        conn.execute(
            "INSERT INTO decisions (folder, enc_id, best_frame, action, species,"
            " updated_at) VALUES (?,?,?,?,?,'t')",
            (folder, enc_id, frame, action, species))
    conn.commit()
    return conn


def test_encounters_from_cache(tmp_path):
    home = _make_home(tmp_path)
    encs = encounters_from_cache(home, "TRIP")
    assert [e.enc_id for e in encs] == ["enc_0001", "enc_0002"]
    assert encs[0].frames == ["_DSC0001.NEF", "_DSC0002.NEF"]
    assert encs[0].best_frame == "_DSC0002.NEF"  # deterministic on the near-tie
    assert encs[1].frames == ["_DSC0005.NEF"]


def test_migrate_backfills_and_reattaches(tmp_path):
    home = _make_home(tmp_path)
    conn = _make_db(tmp_path / "db", [
        ("TRIP", "enc_0001", "_DSC0001.NEF", "confirm", "Sparrow"),  # orphaned
        ("TRIP", "enc_0002", "_DSC0005.NEF", "confirm", "Sparrow"),  # attached
    ])
    plan = plan_migrate_decisions(home, conn)
    assert plan["n_decisions"] == 2 and plan["ok"] == 0
    entries = cast("list[Any]", plan["entries"])
    e1 = next(e for e in entries if e["frame"] == "_DSC0001.NEF")
    assert e1["changed"] is True
    assert (e1["new_enc_id"], e1["new_frame"]) == ("enc_0001", "_DSC0002.NEF")
    assert e1["new_frames"] == ["_DSC0001.NEF", "_DSC0002.NEF"]
    e2 = next(e for e in plan["entries"] if e["frame"] == "_DSC0005.NEF")
    assert e2["changed"] is False
    assert e2["new_frames"] == ["_DSC0005.NEF"]
    # dry run: nothing is written
    row = conn.execute(
        "SELECT frames FROM decisions WHERE best_frame='_DSC0001.NEF'").fetchone()
    assert row["frames"] == ""

    apply_migrate_decisions(home, conn)
    row = conn.execute(
        "SELECT * FROM decisions WHERE folder='TRIP' AND enc_id='enc_0001'"
    ).fetchone()
    assert row["best_frame"] == "_DSC0002.NEF"
    assert json.loads(row["frames"]) == ["_DSC0001.NEF", "_DSC0002.NEF"]
    row2 = conn.execute(
        "SELECT frames FROM decisions WHERE best_frame='_DSC0005.NEF'").fetchone()
    assert json.loads(row2["frames"]) == ["_DSC0005.NEF"]

    # a second pass is a no-op
    plan2 = plan_migrate_decisions(home, conn)
    assert plan2["entries"] == [] and plan2["ok"] == 2
    conn.close()


def test_migrate_ambiguous_listed_not_guessed(tmp_path):
    home = _make_home(tmp_path)
    conn = _make_db(tmp_path / "db", [
        ("TRIP", "enc_0009", "_DSC9999.NEF", "unsure", ""),  # frame in no encounter
    ])
    plan = plan_migrate_decisions(home, conn)
    assert plan["entries"] == []
    ambiguous = cast("list[Any]", plan["ambiguous"])
    assert len(ambiguous) == 1
    assert ambiguous[0]["frame"] == "_DSC9999.NEF"
    apply_migrate_decisions(home, conn)
    row = conn.execute(
        "SELECT * FROM decisions WHERE best_frame='_DSC9999.NEF'").fetchone()
    assert row["enc_id"] == "enc_0009" and row["frames"] == ""  # untouched
    conn.close()


def test_migrate_conflict_left_in_place(tmp_path):
    home = _make_home(tmp_path)
    conn = _make_db(tmp_path / "db", [
        ("TRIP", "enc_0001", "_DSC0001.NEF", "confirm", "Sparrow"),
        ("TRIP", "enc_0001", "_DSC0002.NEF", "not_bird", ""),  # occupies target
    ])
    res = apply_migrate_decisions(home, conn)
    rows = conn.execute(
        "SELECT best_frame FROM decisions WHERE folder='TRIP' AND enc_id="
        "'enc_0001' ORDER BY best_frame").fetchall()
    assert [r["best_frame"] for r in rows] == ["_DSC0001.NEF", "_DSC0002.NEF"]
    assert len([e for e in cast("list[Any]", res["entries"]) if e.get("conflict")]) == 1
    conn.close()


# ---------------------------------------------------------------------------
# web app
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """A review app over a re-detected trip with a legacy (frameless)
    decision pointing at the old best frame."""
    home = _make_home(tmp_path)
    (home / "data").mkdir()
    (home / "data" / "lifelist.csv").write_text(
        "species,scientific,name,date,place\n")
    # the current run's candidates.csv, new format with the frames column
    # (the comma-joined frame list is quoted, as csv.writer writes it)
    (home / "runs" / "TRIP" / "candidates.csv").write_text(
        "enc_id,time_start,time_end,n_frames,best_frame,frames,det_conf,"
        "raw_top3,adj_top3,possible_lifer,lifer_lookalike,rare_here,"
        "rare_evidence\n"
        "enc_0001,2026-01-01 12:00:00,2026-01-01 12:00:04,2,_DSC0002.NEF,"
        '"_DSC0001.NEF,_DSC0002.NEF",0.9,Sparrow (0.100),Sparrow (0.100),'
        "False,,False,\n")
    db = home / "data" / "lifer.sqlite"
    conn = _make_db(db, [("TRIP", "enc_0001", "_DSC0001.NEF", "confirm", "Sparrow")])
    conn.close()
    monkeypatch.setattr(web, "RUNS", home / "runs")
    monkeypatch.setattr(web, "LIFELIST_CSV", home / "data" / "lifelist.csv")
    monkeypatch.setattr(web, "lifer_home", lambda: home)
    app = web.create_app(db)
    app.config["TESTING"] = True
    return app.test_client(), db


def test_web_old_run_cache_fill(tmp_path, monkeypatch):
    """A trip whose candidates.csv predates the frames column: the frame
    lists come from the per-frame cache, so a legacy decision (pre-migration
    key) still attaches to its encounter."""
    home = _make_home(tmp_path)  # old-format CSV + full cache
    (home / "data").mkdir()
    (home / "data" / "lifelist.csv").write_text(
        "species,scientific,name,date,place\n")
    db = home / "data" / "lifer.sqlite"
    conn = _make_db(db, [("TRIP", "enc_0001", "_DSC0001.NEF", "confirm", "Sparrow")])
    conn.close()
    monkeypatch.setattr(web, "RUNS", home / "runs")
    monkeypatch.setattr(web, "LIFELIST_CSV", home / "data" / "lifelist.csv")
    monkeypatch.setattr(web, "lifer_home", lambda: home)
    app = web.create_app(db)
    app.config["TESTING"] = True
    client = app.test_client()
    trips = client.get("/api/trips").get_json()
    assert trips[0]["decided"] == 1
    data = client.get("/api/trip/TRIP").get_json()
    enc = data["groups"][0]["encs"][0]
    assert enc["decision"]["action"] == "confirm"
    assert enc["decision"]["species"] == "Sparrow"


def test_web_legacy_decision_reattaches(app_client):
    client, _db = app_client
    trips = client.get("/api/trips").get_json()
    assert trips[0]["decided"] == 1
    data = client.get("/api/trip/TRIP").get_json()
    enc = data["groups"][0]["encs"][0]
    assert enc["best_frame"] == "_DSC0002.NEF"
    assert enc["decision"]["action"] == "confirm"
    assert enc["decision"]["species"] == "Sparrow"


def test_web_sightings_resolve_by_frames(app_client):
    client, _db = app_client
    rows = client.get("/api/lifelist").get_json()
    row = next(r for r in rows if r["common"].lower() == "sparrow")
    assert row["count"] >= 1
    # the confirmed sighting attaches to the current encounter of the trip,
    # and its thumb is the representative frame's per-frame crop (task 12)
    assert row["thumb"] == "/media/TRIP/crops/frames/_DSC0002.NEF.jpg"


def test_representative_frame_prefers_choice():
    import sqlite3
    from lifer.boxes import representative_frame, set_frame_choice
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE frame_choice (folder TEXT NOT NULL, frames TEXT NOT NULL,"
        " chosen TEXT NOT NULL, updated_at TEXT NOT NULL,"
        " PRIMARY KEY (folder, frames))")
    frames = ["_DSC0001.NEF", "_DSC0002.NEF"]
    # default: the pipeline's best frame
    assert representative_frame(conn, "TRIP", frames, "_DSC0002.NEF") == "_DSC0002.NEF"
    # an owner choice wins over the best frame
    set_frame_choice(conn, "TRIP", frames, "_DSC0001.NEF")
    assert representative_frame(conn, "TRIP", frames, "_DSC0002.NEF") == "_DSC0001.NEF"
    # the choice is keyed by (folder, frame set), order-insensitive
    assert representative_frame(conn, "TRIP", ["_DSC0002.NEF", "_DSC0001.NEF"],
                                "_DSC0002.NEF") == "_DSC0001.NEF"
    # a chosen frame no longer in the encounter falls back to the best frame
    assert representative_frame(conn, "TRIP", ["_DSC0001.NEF", "_DSC0003.NEF"],
                                "_DSC0003.NEF") == "_DSC0003.NEF"
    conn.close()


def test_web_pick_representative_frame(app_client):
    client, db = app_client
    # enc_0001's pipeline best is _DSC0002.NEF; the owner picks _DSC0001.NEF
    r = client.post("/api/frame", json={
        "folder": "TRIP", "enc_id": "enc_0001", "frame": "_DSC0001.NEF"})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["chosen_frame"] == "_DSC0001.NEF"
    data = client.get("/api/trip/TRIP").get_json()
    enc = data["groups"][0]["encs"][0]
    assert enc["best_frame"] == "_DSC0002.NEF"          # pipeline best unchanged
    assert enc["repr_frame"] == "_DSC0001.NEF"          # the chosen photo wins
    assert enc["chosen_frame"] == "_DSC0001.NEF"
    # frames are sorted sharpest-first; _DSC0002 (1686.6) then _DSC0001 (1685.9)
    assert [f["name"] for f in enc["frames"]] == ["_DSC0002.NEF", "_DSC0001.NEF"]
    assert enc["frames"][1]["chosen"] is True
    # the card crop is the chosen frame's per-frame crop
    assert enc["crop"].endswith("/crops/frames/_DSC0001.NEF.jpg")
    # restarting over the same DB keeps the choice
    app2 = web.create_app(db)
    app2.config["TESTING"] = True
    client2 = app2.test_client()
    enc2 = client2.get("/api/trip/TRIP").get_json()["groups"][0]["encs"][0]
    assert enc2["repr_frame"] == "_DSC0001.NEF"
    # a frame that is not one of the encounter's frames is rejected
    r2 = client.post("/api/frame", json={
        "folder": "TRIP", "enc_id": "enc_0001", "frame": "_DSC9999.NEF"})
    assert r2.status_code == 400


def test_web_decision_post_stores_frames(app_client):
    client, db = app_client
    r = client.post("/api/decisions", json={
        "folder": "TRIP", "enc_id": "enc_0001", "best_frame": "_DSC0002.NEF",
        "action": "change", "species": "Sparrow",
        "scientific": "Passer domesticus",
    })
    assert r.get_json()["ok"]
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT frames FROM decisions WHERE best_frame='_DSC0002.NEF'").fetchone()
    conn.close()
    assert json.loads(row[0]) == ["_DSC0001.NEF", "_DSC0002.NEF"]


def test_candidates_csv_carries_frames(tmp_path):
    c = Candidate(enc_id="enc_0001", time_start="t", time_end="t", n_frames=2,
                  best_frame="_DSC0002.NEF", det_conf=0.9,
                  raw_top3=[("Sparrow", "Passer", 0.1)],
                  adj_top3=[("Sparrow", "Passer", 0.1)],
                  possible_lifer=False, lifer_lookalike=None,
                  frames=["_DSC0001.NEF", "_DSC0002.NEF"])
    p = tmp_path / "candidates.csv"
    write_candidates_csv(cands=[c], out_path=p)
    with p.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["frames"] == "_DSC0001.NEF,_DSC0002.NEF"
