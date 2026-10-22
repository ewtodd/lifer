"""Task 28: AF-point subject rule, detection merging, GPU-resize clamp."""
import pytest
import torch

from lifer.boxes import get_box, open_db
from lifer.detect import Detection, FrameDetections, merge_dets, resize_for_yolo
from lifer.subject import af_label_xy, subject_det, subject_from_cache


def test_af_label_grid():
    assert af_label_xy("C6") == pytest.approx((0.50, 0.51))
    assert af_label_xy("B5") == pytest.approx((0.455, 0.445))
    assert af_label_xy("D7") == pytest.approx((0.545, 0.575))


def test_af_label_multi_point_is_mean():
    x, y = af_label_xy("C6, C7")
    assert (x, y) == pytest.approx((0.5225, 0.51))


@pytest.mark.parametrize("v", [None, "", "(none)", "none", "garbage"])
def test_af_label_missing_is_centre(v):
    assert af_label_xy(v) == (0.5, 0.5)


def test_subject_is_nearest_to_af_not_most_confident():
    left = [100, 100, 300, 300, 0.87]     # confident, far from the AF point
    right = [4800, 1800, 5200, 2200, 0.86]
    # AF at the right-hand bird
    assert subject_det([left, right], (0.85, 0.5), 6000, 4000) is right
    # no frame size known: highest confidence
    assert subject_det([left, right], (0.85, 0.5), None, None) is left


def test_subject_distance_zero_inside_ties_by_conf():
    big = [0, 0, 3000, 3000, 0.6]
    small = [1400, 1400, 1600, 1600, 0.9]
    # AF inside both boxes: distance 0 for both, higher conf wins
    assert subject_det([big, small], (1500 / 6000, 1500 / 4000), 6000, 4000) is small
    assert subject_det([], (0.5, 0.5), 6000, 4000) is None


def test_subject_accepts_objects_and_frame_best():
    a, b = Detection(0, 0, 10, 10, 0.9), Detection(5900, 3900, 5990, 3990, 0.5)
    fd = FrameDetections("x", [a, b], af=(0.99, 0.99), width=6000, height=4000)
    assert fd.best is b
    assert FrameDetections("x", [a, b]).best is a  # no size: max conf
    assert FrameDetections("x").best is None


def test_subject_from_cache():
    cache = {"dets": [[0, 0, 10, 10, 0.9], [5900, 3900, 5990, 3990, 0.5]],
             "af": [0.99, 0.99], "width": 6000, "height": 4000}
    assert subject_from_cache(cache)[4] == 0.5
    assert subject_from_cache({}) is None
    assert subject_from_cache(None) is None


def test_get_box_dets_path_uses_af(tmp_path):
    conn = open_db(tmp_path / "t.sqlite")
    dets = [[0, 0, 10, 10, 0.9], [5900, 3900, 5990, 3990, 0.5]]
    box = get_box("F", "a.NEF", conn, dets=dets, af=(0.99, 0.99), size=(6000, 4000))
    assert box.conf == 0.5
    assert get_box("F", "a.NEF", conn, dets=dets).conf == 0.9
    conn.close()


def test_merge_dedupes_overlap_keeping_higher_conf():
    a = Detection(100, 100, 200, 200, 0.6)
    b = Detection(105, 105, 205, 205, 0.8)   # same bird, other pass
    c = Detection(500, 500, 600, 600, 0.3)   # distinct
    out = merge_dets([a, c], [b])
    assert [(d.conf) for d in out] == [0.8, 0.3]
    # touching but barely overlapping boxes stay separate
    d = Detection(190, 100, 290, 200, 0.5)
    assert len(merge_dets([a], [d])) == 2


def test_resize_clamps_and_pads_to_stride():
    # a bright frame: antialiased resize overshoots 1.0 without the clamp
    x = torch.zeros(1, 3, 400, 600)
    x[..., ::2, :] = 1.0
    x[..., :, 300:] = 1.0
    y, (left, top) = resize_for_yolo(x, 0.37)
    assert y.max() <= 1.0 and y.min() >= 0.0
    assert y.shape[-2] % 32 == 0 and y.shape[-1] % 32 == 0
    assert resize_for_yolo(x, 1.0)[0].shape[-1] == 608
    # 148x222 -> padded to 160x224, split evenly (top 6, left 1)
    assert tuple(y.shape[-2:]) == (160, 224) and (left, top) == (1, 6)
    assert y[0, 0, 0, 0] == pytest.approx(114 / 255)
