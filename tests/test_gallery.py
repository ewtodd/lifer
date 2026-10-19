"""Task 27: the finished-photo gallery (lifer/gallery.py).

Uses throwaway LIFER_HOME and fake edit roots / source folders - never the
real BIRDING share or the real DB.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

import lifer.gallery as gal
from lifer.web import _init_db


def _xmp(rating: int, derived: str | None = None) -> str:
    """A minimal darktable sidecar with a rating and optional DerivedFrom."""
    df = f' xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/" xmpMM:DerivedFrom="{derived}"' \
        if derived else ""
    return (f"<?xpacket?><x:xmpmeta><rdf:RDF><rdf:Description "
            f'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="{rating}"{df}/>'
            f"</rdf:RDF></x:xmpmeta>")


def _sidecar(root: Path, name: str, rating: int, derived: str | None = None) -> Path:
    p = root / name
    p.write_text(_xmp(rating, derived))
    return p


def _export(root: Path, name: str, mtime: int = 1000) -> Path:
    d = root / "darktable_exported"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"jpeg")
    import os
    os.utime(p, (mtime, mtime))
    return p


# ---------------- pure functions: rating / base raw / export ----------------

def test_rating_parses_attr_and_element(tmp_path):
    assert gal._sidecar_rating(_sidecar(tmp_path, "a.NEF.xmp", 5)) == 5
    assert gal._sidecar_rating(_sidecar(tmp_path, "b.NEF.xmp", 3)) == 3
    # element form
    p = tmp_path / "c.NEF.xmp"
    p.write_text("<xmp:Rating>5</xmp:Rating>")
    assert gal._sidecar_rating(p) == 5


def test_rating_absent_returns_none(tmp_path):
    p = tmp_path / "plain.NEF.xmp"
    p.write_text("<rdf:Description/>")
    assert gal._sidecar_rating(p) is None


def test_base_raw_uses_derived_from_for_duplicates(tmp_path):
    # a darktable duplicate points at its base raw
    sc = _sidecar(tmp_path, "_DSC8819_01.NEF.xmp", 5, derived="_DSC8819.NEF")
    assert gal._base_raw(sc) == "_DSC8819.NEF"
    # a plain sidecar is its own raw
    sc2 = _sidecar(tmp_path, "_DSC8623.NEF.xmp", 5, derived="_DSC8623.NEF")
    assert gal._base_raw(sc2) == "_DSC8623.NEF"


def test_find_export_picks_base_and_duplicates_newest(tmp_path):
    _sidecar(tmp_path, "_DSC8819.NEF.xmp", 5, derived="_DSC8819.NEF")
    _export(tmp_path, "_DSC8819.jpg", mtime=100)          # older base export
    newest = _export(tmp_path, "_DSC8819_01.jpg", mtime=900)  # newer duplicate
    found = gal.find_export(tmp_path / "_DSC8819.NEF.xmp")
    assert found is not None
    assert found.name == newest.name, "newest mtime must win"


def test_find_export_duplicate_sidecar(tmp_path):
    sc = _sidecar(tmp_path, "_DSC8819_01.NEF.xmp", 5, derived="_DSC8819.NEF")
    _export(tmp_path, "_DSC8819_01.jpg")
    found = gal.find_export(sc)
    assert found is not None and found.name == "_DSC8819_01.jpg"


def test_find_export_missing_returns_none(tmp_path):
    _sidecar(tmp_path, "_DSC9002.NEF.xmp", 5, derived="_DSC9002.NEF")
    assert gal.find_export(tmp_path / "_DSC9002.NEF.xmp") is None


# ---------------- integration: matching + counts + idempotency ----------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake edit root, source folder, LIFER_HOME and a confirmed decision."""
    home = tmp_path / "home"
    (home / "runs" / "tripA").mkdir(parents=True)
    (home / "runs" / "tripA" / "meta.json").write_text(
        '{"trip_id": "tripA", "folder": "%s"}' % (tmp_path / "src"))
    (home / "runs" / "tripA" / "candidates.csv").write_text(
        "enc_id,time_start,time_end,n_frames,best_frame\n"
        "enc_0001,2025-05-26 08:32:44,2025-05-26 08:32:47,1,_DSC9001.NEF\n")
    monkeypatch.setenv("LIFER_HOME", str(home))

    root = tmp_path / "Pictures"
    root.mkdir()
    src = tmp_path / "src"
    src.mkdir()

    monkeypatch.setattr(gal, "_roots", lambda cfg: [root])
    monkeypatch.setattr(gal, "source_folder_for_trip", lambda trip: src)
    monkeypatch.setattr(gal, "region_name", lambda folder: "Nichols Arboretum")
    # capture time is keyed by filename so a sidecar and its source frame agree
    times = {"_DSC9001.NEF": datetime(2025, 5, 26, 8, 32, 44)}
    monkeypatch.setattr(gal, "_capture_time",
                        lambda p: times.get(p.name))

    def fake_render(export, folder, frame, out_full, out_thumb, conn):
        out_full.parent.mkdir(parents=True, exist_ok=True)
        out_full.write_bytes(b"jpg")
        out_thumb.write_bytes(b"thumb")
        return (2400, 1600)
    monkeypatch.setattr(gal, "_render", fake_render)

    db = tmp_path / "db.sqlite"
    _init_db(db)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO decisions (folder, enc_id, best_frame, action, species, "
        "scientific, note, frames, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("tripA", "enc_0001", "_DSC9001.NEF", "confirm",
         "Mourning Warbler", "Setophaga philadelphia", "",
         '["_DSC9001.NEF"]', "2025-05-26 08:32:44"))
    conn.commit()
    conn.close()

    return {"root": root, "src": src, "db": db, "home": home, "tmp": tmp_path}


def test_publish_matches_and_reports_counts(env):
    root = env["root"]
    # 5-star with a confirmed decision + export -> in the gallery
    _sidecar(root, "_DSC9001.NEF.xmp", 5, derived="_DSC9001.NEF")
    _export(root, "_DSC9001.jpg")
    # 5-star but no export -> rated-not-exported
    _sidecar(root, "_DSC9002.NEF.xmp", 5, derived="_DSC9002.NEF")
    # 5-star with an export but no confirmed decision -> rated-not-confirmed
    _sidecar(root, "_DSC9003.NEF.xmp", 5, derived="_DSC9003.NEF")
    _export(root, "_DSC9003.jpg")
    # 4-star -> ignored entirely
    _sidecar(root, "_DSC9004.NEF.xmp", 4, derived="_DSC9004.NEF")

    site = env["tmp"] / "site"
    res = gal.publish_gallery(site, env["db"])

    assert res == {"in_gallery": 1, "rated_not_exported": 1,
                   "rated_not_confirmed": 1}

    json_path = site / "data" / "gallery.json"
    assert json_path.is_file()
    import json
    payload = json.loads(json_path.read_text())
    assert len(payload["photos"]) == 1
    ph = payload["photos"][0]
    assert ph["species"] == "Mourning Warbler"
    assert ph["scientific"] == "Setophaga philadelphia"
    assert ph["date"] == "2025-05-26"
    assert ph["place"] == "Nichols Arboretum"
    assert ph["file"].startswith("/birds/gallery/")
    assert ph["thumb"].startswith("/birds/gallery/")
    assert ph["width"] == 2400 and ph["height"] == 1600

    out = site / "static" / "birds" / "gallery"
    assert any(out.glob("*_DSC9001.jpg")), "the gallery file should be written"


def test_publish_idempotent_removes_lower_rated(env):
    root = env["root"]
    sc = _sidecar(root, "_DSC9001.NEF.xmp", 5, derived="_DSC9001.NEF")
    _export(root, "_DSC9001.jpg")
    site = env["tmp"] / "site2"
    assert gal.publish_gallery(site, env["db"])["in_gallery"] == 1
    out = site / "static" / "birds" / "gallery"
    assert any(out.glob("*_DSC9001.jpg"))

    # owner lowers the rating: the photo leaves the gallery and files are removed
    sc.write_text(_xmp(4, "_DSC9001.NEF"))
    res = gal.publish_gallery(site, env["db"])
    assert res["in_gallery"] == 0
    assert res["rated_not_exported"] == 0
    assert res["rated_not_confirmed"] == 0
    assert not any(out.glob("*_DSC9001*")), "stale gallery file must be removed"


def test_publish_time_mismatch_does_not_match(env):
    root = env["root"]
    # 9005 is in the DB decision set too, but its source capture time is unknown
    _sidecar(root, "_DSC9005.NEF.xmp", 5, derived="_DSC9005.NEF")
    _export(root, "_DSC9005.jpg")
    db = env["db"]
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO decisions (folder, enc_id, best_frame, action, species, "
        "scientific, note, frames, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("tripA", "enc_0002", "_DSC9005.NEF", "confirm",
         "Blue Jay", "Cyanocitta cristata", "", '["_DSC9005.NEF"]',
         "2025-05-26 08:35:00"))
    conn.commit()
    conn.close()
    site = env["tmp"] / "site3"
    # _capture_time returns None for _DSC9005.NEF (not in the times dict), so
    # the decision's frame can't be matched to the sidecar by time.
    res = gal.publish_gallery(site, env["db"])
    assert res["in_gallery"] == 0
    assert res["rated_not_confirmed"] == 1
