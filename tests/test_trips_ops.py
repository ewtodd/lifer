"""Task 24: raws beat derived copies, RAW-bucket parent trip IDs, stale-run
listing, migrate-trips, and delete (CLI + web).

Everything runs on throwaway LIFER_HOME / DB paths -- never the real
~/.local/share/lifer or ~/BIRDING.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lifer import web
from lifer.home import effective_trip_id, is_raw_bucket
from lifer.runall import build_plan
from lifer.trips_ops import (apply_delete, apply_migrate_trips, plan_delete,
                             plan_migrate_trips, raw_frame_for)

ALL_REGIONS = [{"glob": "*", "name": "Test", "short_name": "T", "place_id": 1}]

CAND_HEADER = ("enc_id,time_start,time_end,n_frames,best_frame,det_conf,"
               "raw_top3,adj_top3,possible_lifer,lifer_lookalike,rare_here,"
               "rare_evidence")


def make_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(web.SCHEMA)
    conn.commit()
    return conn


def add_decision(conn, folder, enc_id, frame,
                 action="confirm", species="Red-winged Blackbird"):
    conn.execute(
        "INSERT INTO decisions (folder, enc_id, best_frame, action, species,"
        " scientific, note, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (folder, enc_id, frame, action, species, "Agelaius phoeniceus", "",
         "2026-01-01T00:00:00+00:00"))
    conn.commit()


def add_crop(conn, folder, frame):
    conn.execute(
        "INSERT INTO crop_boxes (folder, frame, x1, y1, x2, y2, reid_old,"
        " reid_new, snapshot, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (folder, frame, 1.0, 1.0, 2.0, 2.0, "", "", "",
         "2026-01-01T00:00:00+00:00"))
    conn.commit()


def write_candidates(run_dir: Path, rows: list[tuple[str, str]]):
    """rows: (enc_id, best_frame); the rest of the columns are filler."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [CAND_HEADER]
    for enc_id, frame in rows:
        top3 = "Red-winged Blackbird (0.9)"
        lines.append(f"{enc_id},2022-12-29T10:00:00,2022-12-29T10:00:05,1,"
                     f"{frame},0.9,{top3},{top3},False,,False,")
    (run_dir / "candidates.csv").write_text("\n".join(lines) + "\n")


def write_cache(home: Path, trip: str, name: str, ts: str,
                sharpness: float = 50.0):
    d = home / "cache" / trip
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{Path(name).stem}.json").write_text(json.dumps({
        "name": name, "timestamp": ts,
        "dets": [[0, 0, 100, 100, 0.9]], "sharpness": sharpness}))


def make_archive(root: Path) -> Path:
    """ST-MARKS: a raw outing (top JPEGs + WATERMARKED + RAW bucket);
    GREENE-VALLEY: a JPEG-only outing (no NEFs anywhere in the tree)."""
    tree = root / "2022-12-29-ST-MARKS"
    raw = tree / "RAW"
    raw.mkdir(parents=True)
    (tree / "_DSC0001_lzn.jpg").write_bytes(b"j")
    wm = tree / "WATERMARKED"
    wm.mkdir()
    (wm / "_DSC0001WM.jpg").write_bytes(b"j")
    for n in (1, 2, 3):
        (raw / f"_DSC{n:04d}.NEF").write_bytes(b"n")
    gv = root / "2026-10-18-GREENE-VALLEY"
    gv.mkdir()
    (gv / "a.jpg").write_bytes(b"j")
    (gv / "b.jpg").write_bytes(b"j")
    return tree


# ---------------------------------------------------------------------------
# Rule 1 + 2: the plan
# ---------------------------------------------------------------------------

def test_plan_raws_beat_derived_copies(tmp_path):
    root = tmp_path / "BIRDING"
    tree = make_archive(root)
    rows = build_plan(root, ALL_REGIONS)
    by_folder = {r.folder: r for r in rows}

    # The top-level folder's own JPEGs: derived copy (tree has NEFs).
    top = by_folder[tree]
    assert top.status == "skip" and "derived copy" in top.reason
    # A sibling with no NEF ancestor, but NEFs elsewhere in the tree.
    wm = by_folder[tree / "WATERMARKED"]
    assert wm.status == "skip" and "derived copy" in wm.reason
    # The RAW bucket runs, under the parent's trip ID (rule 2).
    raw = by_folder[tree / "RAW"]
    assert raw.status == "run" and raw.reason == ""
    assert raw.trip_id == "2022-12-29-ST-MARKS"
    # A JPEG-only tree keeps the task-20 behaviour.
    gv = by_folder[root / "2026-10-18-GREENE-VALLEY"]
    assert gv.status == "run" and gv.trip_id == "2026-10-18-GREENE-VALLEY"


def test_raw_bucket_names(tmp_path):
    root = tmp_path / "B"
    for name in ("raw", "RAW", "NEF"):
        d = root / "X" / name
        d.mkdir(parents=True)
        (d / "a.NEF").write_bytes(b"")
        assert is_raw_bucket(d)
        assert effective_trip_id(d, root) == "X"
    # A folder named RAW with no NEFs is not a bucket.
    d = root / "Y" / "raw"
    d.mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"")
    assert not is_raw_bucket(d)
    assert effective_trip_id(d, root) == "Y--raw"
    # A top-level folder named RAW has no parent outing.
    d = root / "RAW"
    d.mkdir()
    (d / "a.NEF").write_bytes(b"")
    assert effective_trip_id(d, root) == "RAW"


def test_duplicate_subset_still_skipped(tmp_path):
    root = tmp_path / "B"
    d = root / "2026-10-18-GREENE-VALLEY"
    pics = d / "pics"
    pics.mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"j")
    (d / "b.jpg").write_bytes(b"j")
    (pics / "a.jpg").write_bytes(b"j")
    rows = {r.folder: r for r in build_plan(root, ALL_REGIONS)}
    assert rows[d].status == "run"
    assert rows[pics].status == "skip"
    assert "duplicate subset" in rows[pics].reason


# ---------------------------------------------------------------------------
# Rule 3: run-all lists stale runs and orphaned decisions
# ---------------------------------------------------------------------------

def test_runall_lists_stale_runs(tmp_path, monkeypatch, capsys):
    root = tmp_path / "BIRDING"
    tree = make_archive(root)
    home = tmp_path / "home"
    monkeypatch.setenv("LIFER_HOME", str(home))
    home.mkdir()
    (home / "regions.toml").write_text(
        '[[region]]\nglob = "*"\nname = "Test"\nshort_name = "T"\n'
        "place_id = 1\n")
    conn = make_db(home / "data" / "lifer.sqlite")

    # Stale raw-bucket run with DB references + chosen frames.
    stale = "2022-12-29-ST-MARKS--RAW"
    write_candidates(home / "runs" / stale, [("enc_0001", "_DSC0001.NEF")])
    add_decision(conn, stale, "enc_0001", "_DSC0001.NEF")
    add_crop(conn, stale, "_DSC0001.NEF")
    (home / "develop" / stale / "src").mkdir(parents=True)
    (home / "develop" / stale / "src" / "f.NEF").write_bytes(b"")

    # Old top-level JPEG run: same trip ID as the RAW bucket row, but the
    # source folder changed (JPEGs -> NEFs), so its decisions orphan.
    old = home / "runs" / "2022-12-29-ST-MARKS"
    write_candidates(old, [("enc_0001", "_DSC0001_lzn.jpg")])
    (old / "meta.json").write_text(json.dumps(
        {"trip_id": "2022-12-29-ST-MARKS", "folder": str(tree)}))
    add_decision(conn, "2022-12-29-ST-MARKS", "enc_0001", "_DSC0001_lzn.jpg")
    conn.close()

    from lifer.runall import run_all
    run_all(str(root), dry_run=True)
    out = capsys.readouterr().out

    assert "stale runs / decisions to migrate" in out
    assert stale in out
    assert "1 decisions, 1 crop fixes, 1 chosen frames" in out
    assert "source changed" in out
    assert "_DSC0001_lzn.jpg" in out
    # Dry run: nothing was deleted.
    assert (home / "runs" / stale).is_dir()
    assert conn is not None


# ---------------------------------------------------------------------------
# migrate-trips
# ---------------------------------------------------------------------------

def test_raw_frame_for_stems():
    assert raw_frame_for("_DSC0082_lzn.jpg") == "_DSC0082.NEF"
    assert raw_frame_for("_DSC0127_lzn-1.jpg") == "_DSC0127.NEF"
    assert raw_frame_for("_DSC0246_lznWM.jpg") == "_DSC0246.NEF"
    assert raw_frame_for("_DSC0399WM.jpg") == "_DSC0399.NEF"
    assert raw_frame_for("_DSC0159_01.jpg") == "_DSC0159.NEF"
    assert raw_frame_for("_DSC0001.NEF") == "_DSC0001.NEF"


def test_migrate_rekeys_raw_bucket(tmp_path):
    root = tmp_path / "BIRDING"
    tree = make_archive(root)
    home = tmp_path / "home"
    conn = make_db(home / "data" / "lifer.sqlite")
    old, new = "2022-12-29-ST-MARKS--RAW", "2022-12-29-ST-MARKS"
    write_candidates(home / "runs" / old, [("enc_0001", "_DSC0001.NEF")])
    # The re-run: same encounter in the parent trip.
    write_candidates(home / "runs" / new, [("enc_0001", "_DSC0001.NEF")])
    add_decision(conn, old, "enc_0001", "_DSC0001.NEF")
    add_crop(conn, old, "_DSC0001.NEF")
    (home / "develop" / old / "src").mkdir(parents=True)
    (home / "develop" / old / "src" / "f.NEF").write_bytes(b"")

    rows = build_plan(root, ALL_REGIONS)
    plan = plan_migrate_trips(home, conn, root, rows)
    assert len(plan["rekey"]) == 1
    e = plan["rekey"][0]
    assert (e["from"], e["to"]) == (old, new)
    assert e["decisions"] == 1 and e["crop_overrides"] == 1
    assert e["chosen_frames"] == 1 and not e["mismatched"]

    apply_migrate_trips(home, conn, root, rows)
    assert conn.execute(
        "SELECT folder FROM decisions WHERE enc_id='enc_0001'"
    ).fetchone()["folder"] == new
    assert conn.execute(
        "SELECT folder FROM crop_boxes WHERE frame='_DSC0001.NEF'"
    ).fetchone()["folder"] == new
    assert not (home / "develop" / old).exists()
    assert (home / "develop" / new / "src" / "f.NEF").is_file()
    # The stale run dir is NOT deleted by a migration (`lifer delete` does
    # that); only the state was rekeyed.
    assert (home / "runs" / old).is_dir()


def test_migrate_rekey_reports_mismatched(tmp_path):
    root = tmp_path / "BIRDING"
    tree = make_archive(root)
    home = tmp_path / "home"
    conn = make_db(home / "data" / "lifer.sqlite")
    old, new = "2022-12-29-ST-MARKS--RAW", "2022-12-29-ST-MARKS"
    write_candidates(home / "runs" / old, [("enc_0001", "_DSC0001.NEF")])
    # The re-run produced different encounter numbering.
    write_candidates(home / "runs" / new, [("enc_0001", "_DSC0002.NEF")])
    add_decision(conn, old, "enc_0001", "_DSC0001.NEF")
    rows = build_plan(root, ALL_REGIONS)
    plan = plan_migrate_trips(home, conn, root, rows)
    assert plan["rekey"][0]["mismatched"] == ["enc_0001 _DSC0001.NEF"]


def test_migrate_maps_jpg_decisions_onto_raw(tmp_path):
    root = tmp_path / "BIRDING"
    tree = make_archive(root)
    (tree / "_DSC9999_gone.jpg").write_bytes(b"")  # no matching NEF
    home = tmp_path / "home"
    conn = make_db(home / "data" / "lifer.sqlite")
    trip = "2022-12-29-ST-MARKS"
    # The old top-level JPEG run: the trip now runs from the RAW bucket.
    old = home / "runs" / trip
    write_candidates(old, [("enc_0001", "_DSC0001_lzn.jpg")])
    (old / "meta.json").write_text(json.dumps(
        {"trip_id": trip, "folder": str(tree)}))
    add_decision(conn, trip, "enc_0001", "_DSC0001_lzn.jpg")
    add_decision(conn, trip, "enc_0002", "_DSC9999_gone.jpg")
    # The raw trip's per-frame cache: two encounters, 20 s apart.
    write_cache(home, trip, "_DSC0001.NEF", "2026-01-01T10:00:00")
    write_cache(home, trip, "_DSC0002.NEF", "2026-01-01T10:00:20")
    write_candidates(home / "runs" / trip,
                     [("enc_0001", "_DSC0001.NEF"),
                      ("enc_0002", "_DSC0002.NEF")])

    rows = build_plan(root, ALL_REGIONS)
    plan = plan_migrate_trips(home, conn, root, rows)
    assert len(plan["map"]) == 1
    m = plan["map"][0]
    assert (m["to_trip"], m["to_enc_id"], m["to_frame"]) == \
        (trip, "enc_0001", "_DSC0001.NEF")
    assert len(plan["unmapped"]) == 1
    assert plan["unmapped"][0]["frame"] == "_DSC9999_gone.jpg"

    apply_migrate_trips(home, conn, root, rows)
    decs = conn.execute(
        "SELECT folder, enc_id, best_frame FROM decisions ORDER BY enc_id"
    ).fetchall()
    assert [(d["enc_id"], d["best_frame"]) for d in decs] == [
        ("enc_0001", "_DSC0001.NEF"), ("enc_0002", "_DSC9999_gone.jpg")]
    assert all(d["folder"] == trip for d in decs)


def test_migrate_conflict_kept_in_place(tmp_path):
    root = tmp_path / "BIRDING"
    tree = make_archive(root)
    home = tmp_path / "home"
    conn = make_db(home / "data" / "lifer.sqlite")
    trip = "2022-12-29-ST-MARKS"
    old = home / "runs" / trip
    write_candidates(old, [("enc_0001", "_DSC0001_lzn.jpg")])
    (old / "meta.json").write_text(json.dumps(
        {"trip_id": trip, "folder": str(tree)}))
    add_decision(conn, trip, "enc_0001", "_DSC0001_lzn.jpg")
    write_cache(home, trip, "_DSC0001.NEF", "2026-01-01T10:00:00")
    write_candidates(home / "runs" / trip, [("enc_0001", "_DSC0001.NEF")])
    # The destination already has a decision for that encounter.
    add_decision(conn, trip, "enc_0001", "_DSC0001.NEF", action="change",
                 species="Barn Swallow")

    rows = build_plan(root, ALL_REGIONS)
    plan = plan_migrate_trips(home, conn, root, rows)
    assert not plan["map"] and len(plan["conflicts"]) == 1
    apply_migrate_trips(home, conn, root, rows)
    n = conn.execute(
        "SELECT COUNT(*) n FROM decisions WHERE folder=?", (trip,)
    ).fetchone()["n"]
    assert n == 2  # nothing moved, nothing dropped


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_plan_then_apply(tmp_path):
    home = tmp_path / "home"
    conn = make_db(home / "data" / "lifer.sqlite")
    tid = "2022-12-29-ST-MARKS"
    (home / "runs" / tid).mkdir(parents=True)
    (home / "runs" / tid / "candidates.csv").write_text(CAND_HEADER + "\n")
    (home / "cache" / tid / "bioclip").mkdir(parents=True)
    (home / "cache" / tid / "bioclip" / "a.json").write_text("{}")
    (home / "develop" / tid / "src").mkdir(parents=True)
    (home / "develop" / tid / "src" / "f.NEF").write_bytes(b"")
    (home / "edits" / tid).mkdir(parents=True)
    (home / "edits" / tid / "f.jpg").write_bytes(b"")
    add_decision(conn, tid, "enc_0001", "_DSC0001.NEF")
    add_crop(conn, tid, "_DSC0001.NEF")

    plan = plan_delete(home, conn, [tid])
    t = plan["trips"][0]
    assert (t["run_files"], t["cache_files"], t["decisions"],
            t["crop_overrides"], t["chosen_frames"]) == (1, 1, 1, 1, 2)
    assert (home / "runs" / tid).is_dir()  # plan touches nothing

    apply_delete(home, conn, [tid])
    assert not (home / "runs" / tid).exists()
    assert not (home / "cache" / tid).exists()
    assert not (home / "develop" / tid).exists()
    assert not (home / "edits" / tid).exists()
    assert conn.execute("SELECT COUNT(*) n FROM decisions").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) n FROM crop_boxes").fetchone()["n"] == 0


def test_delete_refuses_paths_outside_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    conn = make_db(home / "data" / "lifer.sqlite")
    with pytest.raises(SystemExit):
        apply_delete(home, conn, ["../outside"])
    assert not (tmp_path / "outside").exists()


# ---------------------------------------------------------------------------
# web: /api/delete/plan + /api/delete
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("LIFER_HOME", str(home))
    monkeypatch.setattr(web, "RUNS", home / "runs")
    monkeypatch.setattr(web, "LIFELIST_CSV", home / "data" / "lifelist.csv")
    db = home / "data" / "lifer.sqlite"
    app = web.create_app(db)
    return app.test_client(), home


def test_web_delete_flow(app_client):
    client, home = app_client
    tid = "2022-12-29-ST-MARKS"
    write_candidates(home / "runs" / tid, [("enc_0001", "_DSC0001.NEF")])
    conn = sqlite3.connect(str(home / "data" / "lifer.sqlite"))
    conn.row_factory = sqlite3.Row
    add_decision(conn, tid, "enc_0001", "_DSC0001.NEF")
    add_crop(conn, tid, "_DSC0001.NEF")
    (home / "edits" / tid).mkdir(parents=True)
    (home / "edits" / tid / "f.jpg").write_bytes(b"")
    conn.close()

    # The plan endpoint reports what would be removed.
    j = client.get(f"/api/delete/plan?folder={tid}").get_json()
    assert j["ok"]
    t = j["trips"][0]
    assert (t["run_files"], t["decisions"], t["crop_overrides"],
            t["chosen_frames"]) == (1, 1, 1, 1)

    # Bad trip IDs are rejected.
    assert client.get("/api/delete/plan?folder=../etc").status_code == 400
    assert client.get("/api/delete/plan?folder=a/b").status_code == 400

    # Confirmation must echo the trip ID.
    r = client.post("/api/delete", json={"folder": tid, "confirm": "WRONG"})
    assert r.status_code == 400
    assert (home / "runs" / tid).is_dir()

    # Typed confirmation deletes.
    r = client.post("/api/delete", json={"folder": tid, "confirm": tid})
    assert r.get_json()["ok"]
    assert not (home / "runs" / tid).exists()
    assert not (home / "edits" / tid).exists()
    conn = sqlite3.connect(str(home / "data" / "lifer.sqlite"))
    assert conn.execute("SELECT COUNT(*) n FROM decisions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) n FROM crop_boxes").fetchone()[0] == 0


def test_web_pages_offer_delete(app_client):
    client, home = app_client
    tid = "2022-12-29-ST-MARKS"
    write_candidates(home / "runs" / tid, [("enc_0001", "_DSC0001.NEF")])
    html = client.get("/").get_data(as_text=True)
    assert f'data-delete="{tid}"' in html
    assert "delmodal" in html
    html = client.get(f"/trip/{tid}?delete=1").get_data(as_text=True)
    assert f'data-delete="{tid}"' in html
    assert f'data-trip="{tid}"' in html
