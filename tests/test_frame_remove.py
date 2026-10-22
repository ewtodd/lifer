"""Per-frame remove in the review app: the best frame is re-picked by the
pipeline's rule, the owner's photo pick survives, the removal is recorded for
re-runs, and refusing to remove the last frame is a clean 400.

Runs on a throwaway LIFER_HOME and DB."""

from __future__ import annotations

import json
import sqlite3

import pytest

import lifer.web as web
from lifer.boxes import removed_frames

CAND_HEADER = ("enc_id,time_start,time_end,n_frames,best_frame,frames,det_conf,"
               "raw_top3,adj_top3,possible_lifer,lifer_lookalike,rare_here,"
               "rare_evidence\n")
# enc_0001: A, B (best, sharpest), C; enc_0002: just D
SHARPNESS = {"_A.NEF": 100.0, "_B.NEF": 300.0, "_C.NEF": 200.0, "_D.NEF": 50.0}


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path
    run = home / "runs" / "TRIP"
    run.mkdir(parents=True)
    (run / "candidates.csv").write_text(
        CAND_HEADER
        + 'enc_0001,2026-01-01 12:00:00,2026-01-01 12:00:04,3,_B.NEF,"_A.NEF,_B.NEF,_C.NEF",'
          "0.9,Sparrow (0.100),Sparrow (0.100),False,,False,\n"
        + "enc_0002,2026-01-01 12:01:00,2026-01-01 12:01:00,1,_D.NEF,_D.NEF,"
          "0.9,Sparrow (0.100),Sparrow (0.100),False,,False,\n")
    cache = home / "cache" / "TRIP"
    cache.mkdir(parents=True)
    for i, (name, sharp) in enumerate(SHARPNESS.items()):
        (cache / f"{name[:-4]}.json").write_text(json.dumps({
            "name": name, "timestamp": f"2026-01-01T12:00:0{i}",
            "sharpness": sharp, "dets": [[0, 0, 100, 100, 0.9, "Sparrow"]]}))
    (home / "data").mkdir()
    (home / "data" / "lifelist.csv").write_text("typed_name,common_name\n")
    db = home / "data" / "lifer.sqlite"
    monkeypatch.setattr(web, "RUNS", home / "runs")
    monkeypatch.setattr(web, "LIFELIST_CSV", home / "data" / "lifelist.csv")
    monkeypatch.setattr(web, "lifer_home", lambda: home)
    app = web.create_app(db)
    app.config["TESTING"] = True
    return app.test_client(), db


def _remove(c, frame, enc_id="enc_0001"):
    return c.post("/api/frame/remove",
                  json={"folder": "TRIP", "enc_id": enc_id, "frame": frame})


def _enc(c, enc_id="enc_0001"):
    groups = c.get("/api/trip/TRIP").get_json()["groups"]
    return next(e for g in groups for e in g["encs"] if e["id"] == enc_id)


def test_removing_the_best_frame_repicks_the_sharpest(client):
    c, db = client
    r = _remove(c, "_B.NEF")
    assert r.status_code == 200, r.get_json()
    # C (200) is sharper than A (100); not simply the first remaining frame
    assert r.get_json()["best_frame"] == "_C.NEF"
    enc = _enc(c)
    assert enc["best_frame"] == "_C.NEF"
    assert enc["n_frames"] == 2
    assert sorted(f["name"] for f in enc["frames"]) == ["_A.NEF", "_C.NEF"]
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    assert removed_frames(conn, "TRIP") == {"_B.NEF"}   # kept for re-runs
    conn.close()


def test_owner_pick_survives_removing_another_frame(client):
    c, _db = client
    assert c.post("/api/frame", json={
        "folder": "TRIP", "enc_id": "enc_0001", "frame": "_A.NEF"}).status_code == 200
    assert _remove(c, "_C.NEF").status_code == 200
    assert _enc(c)["repr_frame"] == "_A.NEF"


def test_removing_the_picked_frame_drops_the_pick(client):
    c, _db = client
    c.post("/api/frame", json={"folder": "TRIP", "enc_id": "enc_0001", "frame": "_A.NEF"})
    assert _remove(c, "_A.NEF").status_code == 200
    enc = _enc(c)
    assert enc["repr_frame"] == "_B.NEF"
    assert enc["chosen_frame"] is None


def test_cannot_remove_the_only_frame(client):
    c, _db = client
    r = _remove(c, "_D.NEF", enc_id="enc_0002")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False
    assert _enc(c, "enc_0002")["n_frames"] == 1
