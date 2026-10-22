"""Task 29a: several species in one burst.

Grouping rules (lookalike threshold, subject-only drop, cap, added birds),
candidates.csv with/without the new columns, decision resolution with/without
a bird_box, the per-detection classification cache, and a confirmed extra
bird becoming a sighting of a second species on the same frames.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import lifer.web as web
from lifer.birds import (BirdDet, bird_boxes, bird_frames, average_probs,
                         group_extra_birds, moved_frames)
from lifer.boxes import add_bird, added_birds, open_db, remove_added_bird
from lifer.decisions import (EncRef, bird_box_json, encounters_from_outputs,
                             ensure_bird_box_column, resolve_decision)
from lifer.detect import Detection, FrameDetections
from lifer.outputs import Candidate, write_candidates_csv
from lifer.pipeline import FrameRecord, classify_dets

P, A, B, C, D, E = 1, 2, 3, 4, 5, 6  # taxon ids; P is the primary's species
FRAMES = [f"_DSC{i:04d}.NEF" for i in range(1, 7)]


def _det(frame, species, prob, *, subject=False, added=False, sharp=10.0,
         box=(0, 0, 100, 100), conf=0.9):
    """A detection whose adjusted top-1 is `species` with probability `prob`
    (the rest of the mass spread thin over ten other taxa)."""
    adj = {species: prob, **{100 + i: (1.0 - prob) / 10 for i in range(10)}}
    return BirdDet(frame, box, conf, sharp, dict(adj), adj, subject, added)


PRIMARY = {P: 0.6, A: 0.25, 99: 0.15}  # P's top 3 are P, A, 99


def _groups(dets, primary=PRIMARY):
    return group_extra_birds(dets, primary, FRAMES)


# ---------------------------------------------------------------------------
# grouping rules
# ---------------------------------------------------------------------------

def test_other_species_needs_half_probability():
    assert [b.species for b in _groups([_det(FRAMES[0], B, 0.55)])] == [B]
    assert _groups([_det(FRAMES[0], B, 0.45)]) == []


def test_one_lucky_crop_is_not_enough():
    # best crop clears 0.5, but the group's best three average below it
    ds = [_det(FRAMES[0], B, 0.6), _det(FRAMES[1], B, 0.42), _det(FRAMES[2], B, 0.41)]
    assert _groups(ds) == []


def test_weak_yolo_detection_is_not_an_extra_bird():
    assert _groups([_det(FRAMES[0], B, 0.7, conf=0.4)]) == []
    assert [b.species for b in _groups([_det(FRAMES[0], B, 0.7, conf=0.5)])] == [B]


def test_added_birds_skip_the_filters():
    d = _det(FRAMES[0], B, 0.3, added=True, conf=1.0)
    assert [b.species for b in _groups([d])] == [B]


def test_primary_species_is_never_an_extra_bird():
    assert _groups([_det(FRAMES[0], P, 0.95)]) == []


def test_lookalike_of_the_primary_needs_more_evidence():
    # A is in P's adjusted top 3: 0.7 is not enough, 0.85 is
    assert _groups([_det(FRAMES[0], A, 0.7)]) == []
    assert [b.species for b in _groups([_det(FRAMES[0], A, 0.85)])] == [A]


def test_subject_only_group_in_primary_top3_is_the_primary_misread():
    assert _groups([_det(FRAMES[0], A, 0.95, subject=True),
                    _det(FRAMES[1], A, 0.9, subject=True)]) == []
    # one non-subject detection makes it a second bird
    kept = _groups([_det(FRAMES[0], A, 0.95, subject=True),
                    _det(FRAMES[1], A, 0.9)])
    assert [b.species for b in kept] == [A]
    # an unrelated species may well be the subject (the re-aim case)
    assert [b.species for b in _groups([_det(FRAMES[0], B, 0.9, subject=True)])] == [B]


def test_at_most_three_extra_birds_most_probable_first():
    dets = [_det(FRAMES[i], s, p) for i, (s, p) in
            enumerate([(B, 0.6), (C, 0.9), (D, 0.7), (E, 0.8)])]
    kept = {b.species for b in _groups(dets)}
    assert kept == {C, D, E}  # B has the least summed probability


def test_added_birds_always_kept_outside_the_cap():
    dets = [_det(FRAMES[i], s, 0.9) for i, s in enumerate([B, C, D])]
    # an added box of the primary's own species, and a very uncertain one
    dets += [_det(FRAMES[3], P, 0.9, added=True),
             _det(FRAMES[4], E, 0.2, added=True)]
    birds = _groups(dets)
    assert len(birds) == 5
    assert sum(b.added for b in birds) == 2


def test_two_added_boxes_of_one_species_form_one_bird():
    dets = [_det(FRAMES[0], B, 0.9, added=True, box=(0, 0, 100, 100)),
            _det(FRAMES[1], B, 0.8, added=True, box=(5, 5, 105, 105))]
    [bird] = _groups(dets)
    assert bird.added and len(bird.dets) == 2


def test_frames_boxes_and_moved_frames():
    dets = [_det(FRAMES[1], B, 0.9, subject=True, box=(1, 1, 90, 90)),
            _det(FRAMES[2], B, 0.8, box=(2, 2, 80, 80)),
            _det(FRAMES[2], B, 0.6, box=(3, 3, 70, 70))]  # worse, same frame
    [bird] = _groups(dets)
    assert bird_frames(bird, FRAMES) == [FRAMES[1], FRAMES[2]]
    assert bird_boxes(bird) == {FRAMES[1]: [1, 1, 90, 90],
                                FRAMES[2]: [2, 2, 80, 80]}
    # only the frame where the bird was the subject leaves the primary
    assert moved_frames([bird]) == {FRAMES[1]}
    assert bird.best.frame == FRAMES[1]


def test_averages_use_the_best_three_detections():
    dets = [_det(FRAMES[i], B, p) for i, p in
            enumerate([0.9, 0.8, 0.7, 0.51])]
    [bird] = _groups(dets)
    assert abs(average_probs(bird, "adj")[B] - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# candidates.csv
# ---------------------------------------------------------------------------

def _cand(enc_id="enc_0001", **kw):
    return Candidate(enc_id=enc_id, time_start="2021-12-03 10:00:00",
                     time_end="2021-12-03 10:00:04", n_frames=2,
                     best_frame="_DSC0001.NEF", det_conf=0.9,
                     raw_top3=[("Mallard", "Anas platyrhynchos", 0.9)],
                     adj_top3=[("Mallard", "Anas platyrhynchos", 0.9)],
                     possible_lifer=False, lifer_lookalike=None,
                     frames=["_DSC0001.NEF", "_DSC0002.NEF"], **kw)


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("LIFER_HOME", str(tmp_path))
    monkeypatch.setattr(web, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(web, "lifer_home", lambda: tmp_path)
    (tmp_path / "runs" / "TRIP").mkdir(parents=True)
    return tmp_path


def test_candidates_csv_round_trip_with_extra_bird(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    box = {"_DSC0002.NEF": [10.5, 20.0, 110.5, 120.0]}
    write_candidates_csv(
        [_cand(), _cand("enc_0001-2", bird=2, boxes=box)],
        home / "runs" / "TRIP" / "candidates.csv")
    rows = {c["enc_id"]: c for c in web.load_candidates("TRIP")}
    assert rows["enc_0001"]["bird"] == 0 and rows["enc_0001"]["boxes"] == {}
    assert rows["enc_0001-2"]["bird"] == 2
    assert rows["enc_0001-2"]["boxes"] == box
    refs = {e.enc_id: e for e in encounters_from_outputs(home, "TRIP")}
    assert refs["enc_0001-2"].bird == 2 and refs["enc_0001-2"].boxes == box
    assert refs["enc_0001"].bird == 0


def test_old_candidates_csv_without_the_columns(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    (home / "runs" / "TRIP" / "candidates.csv").write_text(
        "enc_id,time_start,time_end,n_frames,best_frame,det_conf,raw_top3,"
        "adj_top3,possible_lifer,lifer_lookalike,rare_here,rare_evidence\n"
        "enc_0001,2021-12-03 10:00:00,2021-12-03 10:00:04,1,_DSC0001.NEF,0.9,"
        "Mallard (0.900),Mallard (0.900),False,,False,\n")
    [c] = web.load_candidates("TRIP")
    assert c["bird"] == 0 and c["boxes"] == {}
    assert encounters_from_outputs(home, "TRIP")[0].bird == 0


# ---------------------------------------------------------------------------
# resolve_decision
# ---------------------------------------------------------------------------

def _refs():
    return [
        EncRef("enc_0011", "_DSC0003.NEF", ["_DSC0001.NEF", "_DSC0003.NEF"]),
        EncRef("enc_0011-2", "_DSC0004.NEF", ["_DSC0003.NEF", "_DSC0004.NEF"],
               bird=2, boxes={"_DSC0003.NEF": [0, 0, 100, 100],
                              "_DSC0004.NEF": [50, 50, 150, 150]}),
        EncRef("enc_0011-3", "_DSC0003.NEF", ["_DSC0003.NEF"],
               bird=3, boxes={"_DSC0003.NEF": [500, 500, 600, 600]}),
    ]


def test_decision_without_bird_box_only_sees_primary_refs():
    # _DSC0003 is in all three refs; the old rule used to call that ambiguous
    enc, status = resolve_decision("_DSC0003.NEF", ["_DSC0003.NEF"], _refs())
    assert enc.enc_id == "enc_0011" and status == "best-frame"
    # frames only in an extra bird do not resolve a plain decision
    enc, _ = resolve_decision("_DSC0004.NEF", ["_DSC0004.NEF"], _refs())
    assert enc is None


def test_decision_with_bird_box_resolves_by_box():
    bb = bird_box_json("_DSC0003.NEF", [505, 495, 610, 600])
    enc, status = resolve_decision("_DSC0003.NEF", ["_DSC0003.NEF"], _refs(), bb)
    assert enc.enc_id == "enc_0011-3" and status == "bird-box"
    # no box in that frame overlaps enough: none, never the primary
    bb = bird_box_json("_DSC0003.NEF", [3000, 3000, 3100, 3100])
    assert resolve_decision("_DSC0003.NEF", [], _refs(), bb)[0] is None
    # a frame the extra birds don't have a box in
    bb = bird_box_json("_DSC0001.NEF", [0, 0, 100, 100])
    assert resolve_decision("_DSC0001.NEF", [], _refs(), bb)[0] is None


def test_bird_box_column_migrates_in_place():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE decisions (folder TEXT, frames TEXT)")
    ensure_bird_box_column(conn)
    ensure_bird_box_column(conn)  # idempotent
    assert "bird_box" in [r[1] for r in conn.execute("PRAGMA table_info(decisions)")]


# ---------------------------------------------------------------------------
# added_birds table
# ---------------------------------------------------------------------------

def test_added_birds_table_round_trip(tmp_path):
    conn = open_db(tmp_path / "t.sqlite")
    add_bird(conn, "TRIP", "_DSC0003.NEF", [1, 2, 3, 4])
    add_bird(conn, "TRIP", "_DSC0003.NEF", [1, 2, 3, 4])  # idempotent
    add_bird(conn, "OTHER", "_DSC0003.NEF", [9, 9, 9, 9])
    assert added_birds(conn, "TRIP") == [("_DSC0003.NEF", [1, 2, 3, 4])]
    remove_added_bird(conn, "TRIP", "_DSC0003.NEF", [1, 2, 3, 4])
    assert added_birds(conn, "TRIP") == []
    assert web.SCHEMA.count("added_birds")  # the serve app creates it too


# ---------------------------------------------------------------------------
# per-detection classification cache
# ---------------------------------------------------------------------------

class FakeBioCLIP:
    def __init__(self):
        self.calls: list[int] = []

    def probabilities(self, crops, txt, temperature):
        self.calls.append(len(crops))
        return torch.tensor([[0.9, 0.1]] * len(crops))


def _record(name: str, tmp_path: Path) -> FrameRecord:
    arr = np.random.default_rng(1).integers(0, 255, (900, 1200, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return FrameRecord(path=tmp_path / name, name=name,
                       timestamp=datetime(2021, 12, 3, 10, 0, 0),
                       dets=FrameDetections(name, [Detection(0, 0, 1, 1, .9)]),
                       sharpness=1.0, _jpeg=buf.getvalue())


def test_classify_dets_caches_per_box(tmp_path):
    rec = _record("f1.NEF", tmp_path)
    fake = FakeBioCLIP()
    args = (fake, torch.zeros(2), [5, 6], tmp_path, "hash", 3.0)
    boxes = {"f1.NEF": [((100.2, 100, 500, 500), 0.8, False),
                        ((600, 200, 900, 600), 0.7, False)]}
    first = classify_dets(boxes, {"f1.NEF": rec}, *args)
    assert len(first) == 2 and fake.calls == [2]
    assert abs(first[0]["raw"][5] - 0.9) < 1e-6
    # a re-run, with the box shifted by a fraction of a pixel, reads no photo
    rec._jpeg = None
    boxes["f1.NEF"][0] = ((100.4, 100, 500, 500), 0.8, False)
    again = classify_dets(boxes, {"f1.NEF": rec}, *args)
    assert fake.calls == [2]
    assert [round(r["raw"][5], 3) for r in again] == [0.9, 0.9]
    # a new box is classified alone; a new temperature invalidates everything
    boxes["f1.NEF"].append(((10, 10, 300, 300), 0.6, True))
    rec._jpeg = _record("f1.NEF", tmp_path)._jpeg
    classify_dets(boxes, {"f1.NEF": rec}, *args)
    assert fake.calls == [2, 1]
    classify_dets(boxes, {"f1.NEF": rec}, fake, torch.zeros(2), [5, 6],
                  tmp_path, "hash", 2.0)
    assert fake.calls == [2, 1, 3]


# ---------------------------------------------------------------------------
# a confirmed extra bird is a sighting of its own species
# ---------------------------------------------------------------------------

CAND_HEADER = ("enc_id,time_start,time_end,n_frames,best_frame,frames,bird,boxes,"
               "det_conf,raw_top3,adj_top3,possible_lifer,lifer_lookalike,"
               "rare_here,rare_evidence\n")


def _sightings_db(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    boxes = json.dumps({"_DSC0002.NEF": [500, 500, 700, 700]}).replace('"', '""')
    (home / "runs" / "TRIP" / "candidates.csv").write_text(
        CAND_HEADER
        + "enc_0001,2021-12-03 10:00:00,2021-12-03 10:00:04,2,_DSC0001.NEF,"
          "\"_DSC0001.NEF,_DSC0002.NEF\",0,,0.9,Little Blue Heron (0.900),"
          "Little Blue Heron (0.900),False,,False,\n"
        + "enc_0001-2,2021-12-03 10:00:02,2021-12-03 10:00:02,1,_DSC0002.NEF,"
          f"_DSC0002.NEF,2,\"{boxes}\",0.9,Wilson's Snipe (0.900),"
          "Wilson's Snipe (0.900),True,,False,\n")
    (home / "data").mkdir()
    (home / "data" / "lifelist.csv").write_text(
        "typed_name,common_name,scientific_name,taxon_id,date,location,match_score\n")
    monkeypatch.setattr(web, "LIFELIST_CSV", home / "data" / "lifelist.csv")
    monkeypatch.setattr(web, "region_name", lambda folder: "Leon County, FL")
    monkeypatch.setattr(web, "aves_taxa", lambda: [
        (1, "Little Blue Heron", "Egretta caerulea"),
        (2, "Wilson's Snipe", "Gallinago delicata")])
    conn = sqlite3.connect(str(home / "data" / "lifer.sqlite"))
    conn.row_factory = sqlite3.Row
    conn.executescript(web.SCHEMA)
    return conn


def test_confirmed_extra_bird_adds_a_second_species_on_the_same_frames(
        tmp_path, monkeypatch):
    conn = _sightings_db(tmp_path, monkeypatch)
    frames = json.dumps(["_DSC0001.NEF", "_DSC0002.NEF"])
    # the heron decision predates extra birds: no bird_box
    conn.execute(
        "INSERT INTO decisions (folder, enc_id, best_frame, action, species,"
        " frames, updated_at) VALUES ('TRIP','enc_0001','_DSC0001.NEF',"
        "'confirm','Little Blue Heron',?,'t')", (frames,))
    # the snipe decision: its frames are the heron's too, only the box differs
    conn.execute(
        "INSERT INTO decisions (folder, enc_id, best_frame, action, species,"
        " frames, bird_box, updated_at) VALUES ('TRIP','enc_0001-2',"
        "'_DSC0002.NEF','confirm',?,?,?,'t')",
        ("Wilson's Snipe", json.dumps(["_DSC0002.NEF"]),
         bird_box_json("_DSC0002.NEF", [505, 505, 705, 705])))
    conn.commit()
    recs = web.build_sightings(conn)
    assert [s["folder"] for s in recs["little blue heron"]["sightings"]] == ["TRIP"]
    [snipe] = recs["wilson's snipe"]["sightings"]
    assert snipe["folder"] == "TRIP" and snipe["enc"]["enc_id"] == "enc_0001-2"
    assert snipe["date"] == "2021-12-03"


def test_api_decision_on_extra_bird_stores_its_box_server_side(tmp_path, monkeypatch):
    conn = _sightings_db(tmp_path, monkeypatch)
    conn.close()
    client = web.create_app(tmp_path / "data" / "lifer.sqlite").test_client()
    r = client.post("/api/decisions", json={
        "folder": "TRIP", "enc_id": "enc_0001-2", "best_frame": "_DSC0002.NEF",
        "action": "confirm", "species": "Wilson's Snipe"})
    assert r.json["ok"]
    r = client.post("/api/decisions", json={
        "folder": "TRIP", "enc_id": "enc_0001", "best_frame": "_DSC0001.NEF",
        "action": "confirm", "species": "Little Blue Heron"})
    db = sqlite3.connect(str(tmp_path / "data" / "lifer.sqlite"))
    rows = dict(db.execute("SELECT enc_id, bird_box FROM decisions"))
    assert json.loads(rows["enc_0001-2"]) == ["_DSC0002.NEF", 500, 500, 700, 700]
    assert rows["enc_0001"] == ""
    # the trip page still renders both as ordinary encounters in their groups
    trip = client.get("/api/trip/TRIP").json
    ids = {e["id"]: e["decision"]["action"] if e["decision"] else None
           for g in trip["groups"] for e in g["encs"]}
    assert ids == {"enc_0001": "confirm", "enc_0001-2": "confirm"}
