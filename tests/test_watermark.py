"""Task 14: the owner's watermark placement (lifer/watermark.py).

Uses throwaway LIFER_HOME and a fake ~/BIRDING share - never the real things.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.ImageDraw as D
import pytest
from PIL import Image

import lifer.watermark as wm
from lifer.watermark import Watermark, apply_watermark, ensure_assets, map_box


@pytest.fixture
def marks(tmp_path, monkeypatch):
    """A fake ~/BIRDING share with the two (opaque) marks and a throwaway LIFER_HOME."""
    birding = tmp_path / "BIRDING"
    birding.mkdir()
    Image.new("RGBA", (1500, 650), (255, 255, 255, 255)).save(birding / "watermarkfull.png")
    Image.new("RGBA", (719, 650), (255, 255, 255, 255)).save(birding / "watermark.png")
    monkeypatch.setattr(wm, "BIRDING", birding)
    monkeypatch.setenv("LIFER_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return birding


def _changed_bbox(img: Image.Image, out: Image.Image):
    """Bounding box of pixels the watermark changed."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    b = np.asarray(out.convert("RGB"), dtype=np.int16)
    changed = (np.abs(a - b).sum(axis=2) != 0)
    if not changed.any():
        return None
    ys, xs = np.nonzero(changed)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _overlaps(box, rect):
    x1, y1, x2, y2 = box
    r1, r2, r3, r4 = rect
    return not (x2 <= r1 or r3 <= x1 or y2 <= r2 or r4 <= y1)


def test_assets_copied_once(marks):
    home = marks.parent / "home"
    paths = ensure_assets()
    assert set(paths) == {"watermarkfull.png", "watermark.png"}
    assert (home / "assets" / "watermarkfull.png").is_file()
    assert (home / "assets" / "watermark.png").is_file()


def test_bottom_right_bird_moves_mark(marks):
    """The mark must not sit on the bird; a bottom-right bird pushes it away."""
    img = Image.new("RGB", (1600, 1200), (120, 140, 110))
    d = D.Draw(img)
    # This box overlaps the default bottom-right mark rectangle (inset 24, w 160,
    # h 69): (1416, 1107, 1576, 1176), so the mark has to move.
    bird = (1450, 1100, 1590, 1190)
    d.rectangle(bird, fill=(60, 90, 40))
    out = apply_watermark(img, bird, Watermark())
    box = _changed_bbox(img, out)
    assert box is not None
    assert not _overlaps(box, bird), f"mark {box} overlaps bird {bird}"


def test_centered_bird_keeps_default_corner(marks):
    """A centred bird leaves the mark in the default bottom-right corner."""
    img = Image.new("RGB", (1600, 1200), (120, 140, 110))
    d = D.Draw(img)
    d.rectangle((700, 500, 850, 630), fill=(60, 90, 40))
    out = apply_watermark(img, (700, 500, 850, 630), Watermark())
    box = _changed_bbox(img, out)
    assert box is not None
    # default corner: bottom-right
    assert box[0] > img.width // 2 and box[1] > img.height // 2


def test_disabled_does_not_mark(marks):
    img = Image.new("RGB", (1600, 1200), (120, 140, 110))
    out = apply_watermark(img, None, Watermark(enabled=False))
    assert _changed_bbox(img, out) is None


def test_tight_crop_uses_monogram(marks):
    """A crop where the signature can't fit falls back to the monogram."""
    ensure_assets()
    cfg = Watermark()
    img = Image.new("RGB", (100, 100), (120, 140, 110))
    # signature at min_width (120) is wider than the 100px frame, so the
    # monogram (also width 120, but the clamp path) is used instead.
    path, w, h = wm._pick_mark(ensure_assets(), cfg, img, inset=2)
    assert path is not None
    assert Path(path).name == "watermark.png"
    assert w <= 100 and h <= 100


def test_map_box_scales_through_crop():
    # bird at (3000,2000)-(3200,2200) in a 6000x4000 frame, cropped to the
    # bottom-right quarter (origin 3000,2000), then scaled to 800x533.
    crop = (3000, 2000, 6000, 4000)
    box = map_box((5000, 3500, 5200, 3700), crop, 800, 533)
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 800 and 0 <= y1 < y2 <= 533
    # the box should sit in the lower-right of the output
    assert x1 > 400 and y1 > 266
