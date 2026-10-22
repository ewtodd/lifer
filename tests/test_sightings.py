"""build_sightings: an imported LIFELIST.ods row merges into a confirmed trip
only when it records that same visit (same place, dated within the trip)."""

from __future__ import annotations

import json
import sqlite3

import lifer.web as web

LIFELIST_HEADER = "typed_name,common_name,scientific_name,taxon_id,date,location,match_score\n"
CAND_HEADER = ("enc_id,time_start,time_end,n_frames,best_frame,frames,det_conf,"
               "raw_top3,adj_top3,possible_lifer,lifer_lookalike,rare_here,"
               "rare_evidence\n")
PLACE = "Wakulla County, FL"


def _sightings(tmp_path, monkeypatch, ods_rows):
    home = tmp_path
    run = home / "runs" / "TRIP"
    run.mkdir(parents=True)
    (run / "candidates.csv").write_text(
        CAND_HEADER
        + "enc_0001,2022-12-29 09:00:00,2022-12-29 09:00:04,1,_DSC0001.NEF,"
          "_DSC0001.NEF,0.9,Ring-necked Duck (0.900),Ring-necked Duck (0.900),False,,False,\n"
        + "enc_0002,2022-12-30 10:00:00,2022-12-30 10:00:02,1,_DSC0002.NEF,"
          "_DSC0002.NEF,0.9,Mallard (0.900),Mallard (0.900),False,,False,\n")
    (home / "data").mkdir()
    lifelist = home / "data" / "lifelist.csv"
    lifelist.write_text(LIFELIST_HEADER + "".join(
        f'{name},{name},{sci},{tid},{d},"{place}",100\n'
        for name, sci, tid, d, place in ods_rows))
    db = home / "data" / "lifer.sqlite"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.executescript(web.SCHEMA)
    for enc_id, frame, species in [("enc_0001", "_DSC0001.NEF", "Ring-necked Duck"),
                                   ("enc_0002", "_DSC0002.NEF", "Mallard")]:
        conn.execute(
            "INSERT INTO decisions (folder, enc_id, best_frame, action, species,"
            " frames, updated_at) VALUES ('TRIP',?,?,'confirm',?,?,'t')",
            (enc_id, frame, species, json.dumps([frame])))
    conn.commit()
    monkeypatch.setattr(web, "RUNS", home / "runs")
    monkeypatch.setattr(web, "LIFELIST_CSV", lifelist)
    monkeypatch.setattr(web, "lifer_home", lambda: home)
    monkeypatch.setattr(web, "region_name", lambda folder: PLACE)
    monkeypatch.setattr(web, "aves_taxa", lambda: [
        (1, "Ring-necked Duck", "Aythya collaris"),
        (2, "Mallard", "Anas platyrhynchos")])
    try:
        return web.build_sightings(conn)
    finally:
        conn.close()


def test_ods_row_from_another_visit_stays_separate(tmp_path, monkeypatch):
    recs = _sightings(tmp_path, monkeypatch, [
        ("Ring-necked Duck", "Aythya collaris", 1, "11/13/21", PLACE)])
    s = recs["ring-necked duck"]["sightings"]
    # the 2022 photo keeps its own date; the 2021 lifelist row is a sighting too
    assert [(x["source"], x["date"]) for x in s] == [
        ("trip", "2022-12-29"), ("ods", "2021-11-13")]
    assert recs["ring-necked duck"]["_has_ods"]


def test_ods_row_from_the_same_visit_folds_in(tmp_path, monkeypatch):
    recs = _sightings(tmp_path, monkeypatch, [
        # typed up a day before the trip's first photo: within the slack
        ("Mallard", "Anas platyrhynchos", 2, "12/28/22", PLACE)])
    s = recs["mallard"]["sightings"]
    assert len(s) == 1 and s[0]["source"] == "trip"
    assert s[0]["date"] == "2022-12-28"     # the earlier lifelist date wins
    assert recs["mallard"]["_has_ods"]      # still not a new species


def test_ods_row_at_another_place_stays_separate(tmp_path, monkeypatch):
    recs = _sightings(tmp_path, monkeypatch, [
        ("Mallard", "Anas platyrhynchos", 2, "12/30/22", "Leon County, FL")])
    assert [x["source"] for x in recs["mallard"]["sightings"]] == ["trip", "ods"]
