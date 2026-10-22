"""Task 29b: several species in one burst in the review app and the gallery.

Extra birds render with their own boxes, "Add bird" / "Remove added bird",
the gallery's subject choice (owner > AF-nearest > primary) and migrate-decisions
re-attaching a decision to the extra bird its species moved to. Everything runs
on a throwaway LIFER_HOME and DB.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import lifer.gallery as gal
from lifer import web
from lifer.boxes import added_birds
from lifer.decisions import apply_migrate_decisions, plan_migrate_decisions

TRIP = "TRIP"
HEADER = ("enc_id,time_start,time_end,n_frames,best_frame,frames,bird,boxes,"
          "det_conf,raw_top3,adj_top3,possible_lifer,lifer_lookalike,rare_here,"
          "rare_evidence\n")
# enc_0001: Mallard, frames A + B, subject box far right of the frame;
# enc_0001-1: Gadwall in B only, box on the left
PRIMARY_BOX = [800, 500, 1100, 800]
EXTRA_BOX = [100, 300, 400, 600]


def _row(enc_id, frames, best, bird, boxes, species):
    return (f'{enc_id},2026-01-01 12:00:00,2026-01-01 12:00:01,{len(frames)},{best},'
            f'"{",".join(frames)}",{bird},'
            f'"{json.dumps(boxes).replace(chr(34), chr(34) * 2) if boxes else ""}",'
            f"0.9,{species} (0.900),{species} (0.900),False,,False,\n")


def _jpeg(w=1200, h=900) -> bytes:
    arr = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


STUB = {
    "top1": "Northern Pintail",
    "raw_top3": [("Northern Pintail", "Anas acuta", 0.9)],
    "adj_top3": [("Northern Pintail", "Anas acuta", 0.95)],
    "possible_lifer": False, "lifer_lookalike": None,
    "rare_here": False, "rare_evidence": "", "elapsed": 0.1,
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path
    monkeypatch.setenv("LIFER_HOME", str(home))
    monkeypatch.setattr(web, "RUNS", home / "runs")
    monkeypatch.setattr(web, "LIFELIST_CSV", home / "data" / "lifelist.csv")
    monkeypatch.setattr(web, "lifer_home", lambda: home)
    run = home / "runs" / TRIP
    run.mkdir(parents=True)
    src = home / "src"
    src.mkdir()
    (run / "meta.json").write_text(json.dumps({"folder": str(src)}))
    for name in ("_A.jpg", "_B.jpg"):
        (src / name).write_bytes(_jpeg())
    (run / "candidates.csv").write_text(
        HEADER
        + _row("enc_0001", ["_A.jpg", "_B.jpg"], "_A.jpg", 0, {}, "Mallard")
        + _row("enc_0001-1", ["_B.jpg"], "_B.jpg", 1, {"_B.jpg": EXTRA_BOX}, "Gadwall"))
    cache = home / "cache" / TRIP
    cache.mkdir(parents=True)
    for i, name in enumerate(("_A", "_B")):
        (cache / f"{name}.json").write_text(json.dumps({
            "name": f"{name}.jpg", "timestamp": f"2026-01-01T12:00:0{i}",
            "sharpness": 100.0 + i, "width": 1200, "height": 900,
            "af": [0.8, 0.6],            # next to the primary's box
            "dets": [PRIMARY_BOX + [0.9, "bird"]]}))
    (home / "data").mkdir()
    (home / "data" / "lifelist.csv").write_text("typed_name,common_name\n")
    monkeypatch.setattr("lifer.reid.identify_crop",
                        lambda folder, frame, crop: dict(STUB))
    db = home / "data" / "lifer.sqlite"
    app = web.create_app(db)
    app.config["TESTING"] = True
    return {"client": app.test_client(), "db": db, "home": home, "run": run}


def _enc(client, enc_id):
    groups = client.get(f"/api/trip/{TRIP}").get_json()["groups"]
    return next(e for g in groups for e in g["encs"] if e["id"] == enc_id)


def _confirm(client, enc_id, best, species):
    r = client.post("/api/decisions", json={
        "folder": TRIP, "enc_id": enc_id, "best_frame": best,
        "action": "confirm", "species": species})
    assert r.get_json()["ok"]


# ---------------------------------------------------------------------------
# extra birds render with their own boxes
# ---------------------------------------------------------------------------

def test_extra_bird_frames_use_its_boxes(env):
    c = env["client"]
    extra, primary = _enc(c, "enc_0001-1"), _enc(c, "enc_0001")
    f = extra["frames"][0]
    assert f["crop"] == f"/media/{TRIP}/crops/birds/enc_0001-1/_B.jpg.jpg"
    assert f["frame"] == f"/media/{TRIP}/frames/birds/enc_0001-1/_B.jpg.jpg"
    assert f["has_box"] and extra["override"] is None
    # the primary keeps get_box (the subject detection) and its own assets
    assert primary["frames"][0]["crop"].startswith(f"/media/{TRIP}/crops/frames/")
    # the crop is cut around the extra's box (300 px + 20% margins), not the
    # primary's subject box
    r = c.get(f["crop"])
    assert r.status_code == 200
    assert Image.open(BytesIO(r.data)).size == (420, 420)
    assert (env["run"] / "crops" / "birds" / "enc_0001-1" / "_B.jpg.jpg").is_file()
    assert c.get(f["frame"]).status_code == 200
    # burst mates, both ways
    assert extra["mates"] == [{"id": "enc_0001", "species": "Mallard"}]
    assert primary["mates"] == [{"id": "enc_0001-1", "species": "Gadwall"}]
    assert extra["bird"] == 1 and primary["bird"] == 0
    assert c.get(f"/media/{TRIP}/crops/birds/enc_0001-9/_B.jpg").status_code == 404
    assert c.get(f"/media/{TRIP}/crops/birds/enc_0001-1/_Z.jpg").status_code == 404


def test_extra_bird_has_no_fix_crop_or_frame_remove(env):
    c = env["client"]
    r = c.post("/api/crop", json={"folder": TRIP, "enc_id": "enc_0001-1",
                                  "box": [1, 1, 50, 50]})
    assert r.status_code == 400
    r = c.post("/api/frame/remove", json={"folder": TRIP, "enc_id": "enc_0001-1",
                                          "frame": "_B.jpg"})
    assert r.status_code == 400
    html = c.get(f"/trip/{TRIP}").get_data(as_text=True)
    assert 'data-goto="enc_0001-1"' in html and 'data-goto="enc_0001"' in html
    assert html.count("data-rm-frame=") == 2        # the primary's two frames only


# ---------------------------------------------------------------------------
# Add bird / Remove added bird
# ---------------------------------------------------------------------------

def test_add_bird_then_remove(env):
    c = env["client"]
    box = [100, 100, 300, 300]
    r = c.post("/api/bird/add", json={"folder": TRIP, "enc_id": "enc_0001",
                                      "frame": "_A.jpg", "box": box})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["enc_id"] == "enc_0001-2"
    row = next(x for x in web.load_candidates(TRIP) if x["enc_id"] == "enc_0001-2")
    assert row["bird"] == 2 and row["boxes"] == {"_A.jpg": box}
    assert row["frames"] == ["_A.jpg"] and row["adj_top3"][0][0] == "Northern Pintail"
    assert _added(_conn(env)) == [("_A.jpg", [float(v) for v in box])]
    # its card shows in its own species group, flagged removable
    groups = c.get(f"/api/trip/{TRIP}").get_json()["groups"]
    g = next(g for g in groups if g["species"] == "Northern Pintail")
    assert [e["id"] for e in g["encs"]] == ["enc_0001-2"] and g["encs"][0]["added"]
    assert not _enc(c, "enc_0001-1")["added"]       # found by the detector
    assert {m["id"] for m in _enc(c, "enc_0001")["mates"]} == {"enc_0001-1", "enc_0001-2"}

    # the same box again, or a box over the primary's subject, is refused
    r = c.post("/api/bird/add", json={"folder": TRIP, "enc_id": "enc_0001",
                                      "frame": "_A.jpg", "box": box})
    assert r.status_code == 409
    r = c.post("/api/bird/add", json={"folder": TRIP, "enc_id": "enc_0001",
                                      "frame": "_A.jpg", "box": PRIMARY_BOX})
    assert r.status_code == 409

    # a decision about the added bird goes with it; the primary's stays
    _confirm(c, "enc_0001-2", "_A.jpg", "Northern Pintail")
    _confirm(c, "enc_0001", "_A.jpg", "Mallard")
    assert _enc(c, "enc_0001-2")["decision"]["species"] == "Northern Pintail"
    assert c.post("/api/bird/remove", json={"folder": TRIP, "enc_id": "enc_0001-2"}
                  ).get_json() == {"ok": True, "decisions_removed": 1}
    assert [x["enc_id"] for x in web.load_candidates(TRIP)] == ["enc_0001", "enc_0001-1"]
    assert _added(_conn(env)) == []
    assert _enc(c, "enc_0001")["decision"]["species"] == "Mallard"
    left = sqlite3.connect(env["db"]).execute("SELECT enc_id FROM decisions").fetchall()
    assert left == [("enc_0001",)]
    # a detector-found extra can't be removed this way
    r = c.post("/api/bird/remove", json={"folder": TRIP, "enc_id": "enc_0001-1"})
    assert r.status_code == 400


def _added(conn):
    return added_birds(conn, TRIP)


def test_add_bird_on_a_run_without_bird_columns(env):
    run = env["run"]
    (run / "candidates.csv").write_text(
        "enc_id,time_start,time_end,n_frames,best_frame,frames,det_conf,raw_top3,"
        "adj_top3,possible_lifer,lifer_lookalike,rare_here,rare_evidence\n"
        '"enc_0001",2026-01-01 12:00:00,2026-01-01 12:00:01,2,_A.jpg,"_A.jpg,_B.jpg",'
        "0.9,Mallard (0.900),Mallard (0.900),False,,False,\n")
    r = env["client"].post("/api/bird/add", json={
        "folder": TRIP, "enc_id": "enc_0001", "frame": "_A.jpg",
        "box": [100, 100, 300, 300]})
    assert r.status_code == 200, r.get_json()
    rows = {x["enc_id"]: x for x in web.load_candidates(TRIP)}
    assert rows["enc_0001"]["bird"] == 0 and rows["enc_0001-1"]["bird"] == 1


# ---------------------------------------------------------------------------
# gallery subject
# ---------------------------------------------------------------------------

def test_gallery_subject_endpoint_and_flags(env):
    c = env["client"]
    body = {"folder": TRIP, "enc_id": "enc_0001-1", "frame": "_B.jpg"}
    assert c.post("/api/gallery_subject", json=body).status_code == 400  # unconfirmed
    _confirm(c, "enc_0001-1", "_B.jpg", "Gadwall")
    _confirm(c, "enc_0001", "_A.jpg", "Mallard")
    r = c.post("/api/gallery_subject", json=body)
    assert r.get_json()["ok"]
    bb = json.loads(r.get_json()["bird_box"])
    assert bb[0] == "_B.jpg" and bb[1:] == [float(v) for v in EXTRA_BOX]
    flags = lambda e: {f["name"]: f["gallery"] for f in _enc(c, e)["frames"]}
    assert flags("enc_0001-1") == {"_B.jpg": True}
    assert flags("enc_0001") == {"_A.jpg": False, "_B.jpg": False}
    # choosing the primary for that frame moves the flag (one bird per frame)
    r = c.post("/api/gallery_subject", json={**body, "enc_id": "enc_0001"})
    assert r.get_json() == {"ok": True, "bird_box": ""}
    assert flags("enc_0001-1") == {"_B.jpg": False}
    assert flags("enc_0001") == {"_A.jpg": False, "_B.jpg": True}
    c.post("/api/gallery_subject", json={**body, "enc_id": "enc_0001", "clear": True})
    assert flags("enc_0001") == {"_A.jpg": False, "_B.jpg": False}


def _birds(env_):
    """The two confirmed birds of frame _B as the gallery sees them."""
    refs = web._enc_refs(TRIP, web.load_candidates(TRIP), [])
    return [{"folder": TRIP, "ref": r} for r in refs if "_B.jpg" in r.frames]


def _conn(env_):
    conn = sqlite3.connect(env_["db"])
    conn.row_factory = sqlite3.Row
    return conn


def test_pick_subject_owner_then_af_then_primary(env):
    birds = _birds(env)
    conn = _conn(env)
    # AF (0.8, 0.6 -> px 960, 540) sits inside the primary's box
    assert gal.pick_subject(birds, TRIP, "_B.jpg", conn)["ref"].enc_id == "enc_0001"
    # move the AF point onto the extra bird: nearest the AF wins
    cache = env["home"] / "cache" / TRIP / "_B.json"
    d = json.loads(cache.read_text())
    d["af"] = [0.2, 0.5]
    cache.write_text(json.dumps(d))
    assert gal.pick_subject(birds, TRIP, "_B.jpg", conn)["ref"].enc_id == "enc_0001-1"
    # the owner's choice beats the AF point
    conn.execute("INSERT INTO gallery_subject VALUES (?,?,?,?)",
                 (TRIP, "_B.jpg", "", "now"))
    assert gal.pick_subject(birds, TRIP, "_B.jpg", conn)["ref"].enc_id == "enc_0001"
    conn.execute("UPDATE gallery_subject SET bird_box=?", (
        json.dumps(["_B.jpg", *EXTRA_BOX]),))
    assert gal.pick_subject(birds, TRIP, "_B.jpg", conn)["ref"].enc_id == "enc_0001-1"
    # no AF point recorded and no choice: the primary
    conn.execute("DELETE FROM gallery_subject")
    d.pop("af")
    cache.write_text(json.dumps(d))
    assert gal.pick_subject(birds, TRIP, "_B.jpg", conn)["ref"].enc_id == "enc_0001"


def test_publish_names_the_photo_after_the_chosen_bird(env, monkeypatch):
    c = env["client"]
    _confirm(c, "enc_0001-1", "_B.jpg", "Gadwall")
    _confirm(c, "enc_0001", "_A.jpg", "Mallard")
    root = env["home"] / "Pictures"
    root.mkdir()
    (root / "_B.NEF.xmp").write_text(
        '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="5" '
        'xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/" xmpMM:DerivedFrom="_B.jpg"/>')
    (root / "darktable_exported").mkdir()
    (root / "darktable_exported" / "_B.jpg").write_bytes(b"jpeg")
    monkeypatch.setattr(gal, "_roots", lambda cfg: [root])
    monkeypatch.setattr(gal, "region_name", lambda folder: "Somewhere")
    monkeypatch.setattr(gal, "_capture_time",
                        lambda p: datetime(2026, 1, 1, 12, 0, 1))
    boxes_used = []

    def fake_render(export, folder, frame, out_full, out_thumb, conn, bird_box=None):
        boxes_used.append(bird_box)
        out_full.parent.mkdir(parents=True, exist_ok=True)
        out_full.write_bytes(b"jpg")
        out_thumb.write_bytes(b"t")
        return (2400, 1600)
    monkeypatch.setattr(gal, "_render", fake_render)

    def species(site):
        gal.publish_gallery(site, env["db"])
        return json.loads((site / "data" / "gallery.json").read_text())["photos"][0]["species"]

    # the AF point is on the Mallard (primary)
    assert species(env["home"] / "site1") == "Mallard"
    assert boxes_used[-1] is None
    # the owner names the Gadwall
    assert c.post("/api/gallery_subject", json={
        "folder": TRIP, "enc_id": "enc_0001-1", "frame": "_B.jpg"}).get_json()["ok"]
    assert species(env["home"] / "site2") == "Gadwall"
    assert boxes_used[-1] == [float(v) for v in EXTRA_BOX]


# ---------------------------------------------------------------------------
# migrate-decisions: a decision follows its species when a burst splits
# ---------------------------------------------------------------------------

def test_migrate_moves_a_decision_to_the_extra_with_its_species(env):
    c = env["client"]
    # decided before the burst split: Gadwall, on the primary's frames
    _confirm(c, "enc_0001", "_A.jpg", "Gadwall")
    conn = _conn(env)
    conn.execute("UPDATE decisions SET frames=?", (json.dumps(["_A.jpg", "_B.jpg"]),))
    conn.commit()
    plan = plan_migrate_decisions(env["home"], conn)
    [e] = [x for x in plan["entries"] if x.get("to_extra")]
    assert (e["new_enc_id"], e["new_frame"]) == ("enc_0001-1", "_B.jpg")
    assert json.loads(e["bird_box"]) == ["_B.jpg", *[float(v) for v in EXTRA_BOX]]
    # a dry run touches nothing
    assert conn.execute("SELECT bird_box FROM decisions").fetchone()[0] == ""
    res = apply_migrate_decisions(env["home"], conn)
    assert res["applied"] == 1
    row = conn.execute("SELECT enc_id, best_frame, bird_box FROM decisions").fetchone()
    assert row["enc_id"] == "enc_0001-1" and row["bird_box"] == e["bird_box"]
    # idempotent, and the review app now shows it on the extra bird
    again = plan_migrate_decisions(env["home"], conn)
    assert again["entries"] == [] and again["ok"] == 1
    assert _enc(c, "enc_0001-1")["decision"]["species"] == "Gadwall"
    assert _enc(c, "enc_0001")["decision"] is None


def test_migrate_leaves_decisions_that_still_match_the_primary(env):
    _confirm(env["client"], "enc_0001", "_A.jpg", "Mallard")
    plan = plan_migrate_decisions(env["home"], _conn(env))
    assert not [x for x in plan["entries"] if x.get("to_extra")]
