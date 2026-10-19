"""Task 23: encounters with no usable detection get no species.

Covers the classifier's crop guard (empty / zero-area / near-uniform crops),
no-detection encounters in the pipeline, and the review app's "No bird
detected" section with its fix-crop flow (draw a box -> crop + re-identify ->
the encounter becomes a real candidate; Undo reverts it).
"""
import json
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from lifer import web
from lifer.boxes import open_db
from lifer.bursts import Encounter, FrameInfo
from lifer.detect import Detection, FrameDetections, usable_crop
from lifer.pipeline import FrameRecord, classify_encounter

FOLDER = "2023-02-04-ST-MARKS"  # matches the *ST-MARKS* glob in regions.toml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _jpeg(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _noise_jpeg(w: int = 1200, h: int = 900) -> bytes:
    arr = np.random.default_rng(42).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return _jpeg(Image.fromarray(arr))


def _black_jpeg(w: int = 1200, h: int = 900) -> bytes:
    return _jpeg(Image.new("RGB", (w, h), (0, 0, 0)))


def _det(x1=100, y1=100, x2=700, y2=700, conf=0.9) -> Detection:
    return Detection(x1, y1, x2, y2, conf)


def _frame(name: str, jpeg: bytes, dets,
           ts: str = "2021-08-09T16:40:03") -> FrameRecord:
    fd = FrameDetections(frame_name=name, dets=dets)
    return FrameRecord(
        path=Path("/nonexistent") / name, name=name,
        timestamp=datetime.fromisoformat(ts), dets=fd,
        sharpness=-1.0 if not dets else 100.0, _jpeg=jpeg,
    )


def _ts(i: int) -> datetime:
    return datetime(2021, 8, 9, 16, 40, i)


class FakeBioCLIP:
    """Stands in for the model: deterministic probabilities, call counting."""

    def __init__(self, n_classes: int):
        self.n = n_classes
        self.calls: list[int] = []

    def probabilities(self, crops, txt, temperature):
        self.calls.append(len(crops))
        row = [0.0] * self.n
        row[0] = 0.97
        row[1] = 0.03
        return [row[:] for _ in crops]


def _classes() -> dict:
    return {5: SimpleNamespace(common="Species A", scientific="a b"),
            6: SimpleNamespace(common="Species B", scientific="b c")}


def _classify(tmp_path, enc: Encounter, recs: dict, fake: FakeBioCLIP):
    conn = open_db(tmp_path / "t.sqlite")
    return classify_encounter(enc, recs, _classes(), fake, torch.zeros(2),
                              tmp_path, "hash", 0.01, folder="TEST", conn=conn)


# ---------------------------------------------------------------------------
# usable_crop
# ---------------------------------------------------------------------------

def test_usable_crop_rejects_empty_and_blank():
    assert not usable_crop(Image.new("RGB", (0, 0)))
    assert not usable_crop(Image.new("RGB", (1, 1), 0))
    assert not usable_crop(Image.new("RGB", (64, 64), (0, 0, 0)))
    # near-uniform: values within 1 grey level (variance ~0.25, below 1.0)
    tight = np.full((32, 32), 128, dtype=np.uint8)
    tight += np.random.default_rng(2).integers(0, 2, (32, 32)).astype("uint8")
    assert not usable_crop(Image.fromarray(np.stack([tight] * 3, -1)))


def test_usable_crop_accepts_texture():
    arr = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert usable_crop(Image.fromarray(arr))


# ---------------------------------------------------------------------------
# classify_encounter
# ---------------------------------------------------------------------------

def test_classify_no_det_frames_returns_zeros(tmp_path):
    fake = FakeBioCLIP(2)
    enc = Encounter(frames=[FrameInfo("f1.NEF", _ts(1), 0, 0.0, -1.0)])
    recs = {"f1.NEF": _frame("f1.NEF", _black_jpeg(), [])}
    vec = _classify(tmp_path, enc, recs, fake)
    assert vec == [0.0, 0.0]
    assert fake.calls == []  # nothing was ever fed to the model


def test_classify_black_crop_returns_zeros(tmp_path):
    # a detected frame whose image is blank: the guard must skip the crop
    fake = FakeBioCLIP(2)
    enc = Encounter(frames=[FrameInfo("f1.NEF", _ts(1), 1, 0.9, 100.0)])
    recs = {"f1.NEF": _frame("f1.NEF", _black_jpeg(), [_det()])}
    vec = _classify(tmp_path, enc, recs, fake)
    assert vec == [0.0, 0.0]
    assert fake.calls == []


def test_classify_zero_area_box_returns_zeros(tmp_path):
    fake = FakeBioCLIP(2)
    enc = Encounter(frames=[FrameInfo("f1.NEF", _ts(1), 1, 0.9, 100.0)])
    recs = {"f1.NEF": _frame("f1.NEF", _noise_jpeg(), [_det(x1=100, y1=100,
                                                            x2=100, y2=100)])}
    vec = _classify(tmp_path, enc, recs, fake)
    assert vec == [0.0, 0.0]
    assert fake.calls == []


def test_classify_textured_crop_classifies(tmp_path):
    fake = FakeBioCLIP(2)
    enc = Encounter(frames=[FrameInfo("f1.NEF", _ts(1), 1, 0.9, 100.0)])
    recs = {"f1.NEF": _frame("f1.NEF", _noise_jpeg(), [_det()])}
    vec = _classify(tmp_path, enc, recs, fake)
    assert vec[0] > 0.5 and sum(vec) == pytest.approx(1.0)
    assert fake.calls == [1]


def test_classify_mixed_encounter_uses_only_det_frames(tmp_path):
    # the no-det frame of a mixed encounter must not be classified (task 23:
    # "mixing detection and non-detection frames classifies only the
    # detected frames")
    fake = FakeBioCLIP(2)
    enc = Encounter(frames=[
        FrameInfo("f1.NEF", _ts(1), 1, 0.9, 100.0),
        FrameInfo("f2.NEF", _ts(2), 0, 0.0, -1.0),
    ])
    recs = {
        "f1.NEF": _frame("f1.NEF", _noise_jpeg(), [_det()]),
        "f2.NEF": _frame("f2.NEF", _noise_jpeg(), []),
    }
    vec = _classify(tmp_path, enc, recs, fake)
    assert vec[0] > 0.5
    assert fake.calls == [1]  # exactly the one detected crop


# ---------------------------------------------------------------------------
# web app: load_nodet / build_trip
# ---------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LIFER_HOME", str(tmp_path))
    monkeypatch.setattr(web, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(web, "LIFELIST_CSV", tmp_path / "data" / "lifelist.csv")
    (tmp_path / "runs").mkdir()
    return tmp_path


def _make_run(home: Path) -> Path:
    d = home / "runs" / FOLDER
    (d / "crops" / "nodet").mkdir(parents=True)
    (d / "candidates.csv").write_text(
        "enc_id,time_start,time_end,n_frames,best_frame,det_conf,raw_top3,"
        "adj_top3,possible_lifer,lifer_lookalike,rare_here,rare_evidence\n"
        "enc_0001,2023-02-04 09:00:00,2023-02-04 09:00:01,2,_DSC0002.NEF,0.800,"
        "American Crow (0.900),American Crow (0.900),False,,False,\n",
        encoding="utf-8")
    (d / "nodet.json").write_text(json.dumps([{
        "enc_id": "enc_0002",
        "time_start": "2023-02-04 10:00:00",
        "time_end": "2023-02-04 10:00:02",
        "n_frames": 3,
        "best_frame": "_DSC0001.jpg",
    }]) + "\n", encoding="utf-8")
    return d


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(web.SCHEMA)
    return conn


def test_load_nodet_prefers_json(home):
    d = _make_run(home)
    (d / "crops" / "nodet" / "legacy.jpg").write_bytes(b"x")
    entries = web.load_nodet(FOLDER)
    assert entries == [{"enc_id": "enc_0002",
                        "time_start": "2023-02-04 10:00:00",
                        "time_end": "2023-02-04 10:00:02",
                        "n_frames": 3,
                        "best_frame": "_DSC0001.jpg"}]


def test_load_nodet_legacy_fallback(home):
    d = home / "runs" / FOLDER
    (d / "crops" / "nodet").mkdir(parents=True)
    (d / "crops" / "nodet" / "a.jpg").write_bytes(b"x")
    (d / "crops" / "nodet" / "b.jpg").write_bytes(b"x")
    assert web.load_nodet(FOLDER) == [{"file": "a.jpg"}, {"file": "b.jpg"}]


def test_build_trip_nodet_enc_entries(home):
    _make_run(home)
    t = web.build_trip(FOLDER, _conn())
    assert len(t["nodet"]) == 1
    n = t["nodet"][0]
    assert n["id"] == "enc_0002"
    assert n["enc"] is True
    assert n["thumb"] == f"/media/{FOLDER}/crops/nodet/enc_0002.jpg"
    assert n["time"] == "10:00"
    assert n["n_frames"] == 3
    # the no-detection encounter is not a candidate: no species group holds it
    enc_ids = [e["id"] for g in t["groups"] for e in g["encs"]]
    assert "enc_0002" not in enc_ids
    assert "enc_0001" in enc_ids


# ---------------------------------------------------------------------------
# web app: fix-crop on a no-detection encounter
# ---------------------------------------------------------------------------

def test_api_crop_nodet_fix_and_undo(home, monkeypatch):
    d = _make_run(home)
    # source photo (read-only stand-in for the BIRDING share)
    src = home / "BIRDING" / FOLDER
    src.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"folder": str(src)}),
                                 encoding="utf-8")
    (src / "_DSC0001.jpg").write_bytes(_noise_jpeg())
    # the pipeline's frame cache: size known, no detections
    cache = home / "cache" / FOLDER
    cache.mkdir(parents=True)
    (cache / "_DSC0001.json").write_text(json.dumps({
        "name": "_DSC0001.jpg", "width": 1200, "height": 900, "dets": [],
    }), encoding="utf-8")

    stub = {
        "top1": "American Crow",
        "raw_top3": [["American Crow", "Corvus brachyrhynchos", 0.9]],
        "adj_top3": [["American Crow", "Corvus brachyrhynchos", 0.95]],
        "possible_lifer": True, "lifer_lookalike": None,
        "rare_here": False, "rare_evidence": "", "elapsed": 0.1,
    }
    monkeypatch.setattr("lifer.reid.identify_crop",
                        lambda folder, frame, crop: dict(stub))

    db = home / "data" / "lifer.sqlite"
    client = web.create_app(db).test_client()

    # cropboxes: the encounter is known (via nodet.json), has no detector box
    r = client.get(f"/api/cropboxes?folder={FOLDER}&enc_id=enc_0002")
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] and j["frame"] == "_DSC0001.jpg"
    assert j["boxes"] == [] and j["override"] is None

    # fix: draw a box -> crop + re-identify -> real candidate
    r = client.post("/api/crop", json={"folder": FOLDER, "enc_id": "enc_0002",
                                       "box": [100, 100, 700, 700]})
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["ok"] and j["overridden"] is True
    assert j["reid"] == {"old": "", "new": "American Crow"}
    row = next(c for c in web.load_candidates(FOLDER)
               if c["enc_id"] == "enc_0002")
    assert row["adj_top3"][0][0] == "American Crow"
    assert row["possible_lifer"] is True
    assert web.load_nodet(FOLDER) == []  # left the no-bird section
    assert (d / "crops" / "enc_0002.jpg").is_file()

    # undo: no detector box to fall back to -> back to "No bird detected"
    r = client.post("/api/crop", json={"folder": FOLDER, "enc_id": "enc_0002",
                                       "undo": True})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] and j["nodet"] is True
    assert not any(c["enc_id"] == "enc_0002"
                   for c in web.load_candidates(FOLDER))
    assert [e["enc_id"] for e in web.load_nodet(FOLDER)] == ["enc_0002"]
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM crop_boxes WHERE folder = ?",
        (FOLDER,)).fetchone()["n"] == 0
