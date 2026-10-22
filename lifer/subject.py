"""Which detection is the subject: the one under the camera's autofocus point.

Torch-free on purpose (the review app imports it). The D5600 records the AF
point it focused on in every NEF; the owner focuses on the bird, so the
detection nearest that point is the subject. Picking the highest confidence
instead was a coin flip whenever a frame held several birds (28% of frames).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

CENTRE = (0.5, 0.5)
_LABEL = re.compile(r"\b([A-E])(\d{1,2})\b")


def af_label_xy(value) -> tuple[float, float]:
    """Normalised (x, y) of an exiftool AFPointsUsed value ("C6", "B5, B6").

    A linear grid fitted to where birds sit per label across the archive:
    columns step 0.045 of the frame width around 6, rows 0.065 of the height
    around row C. Several points -> their mean; absent/"(none)" -> centre.
    """
    pts = [(0.50 + 0.045 * (int(c) - 6), 0.51 + 0.065 * (ord(r) - ord("C")))
           for r, c in _LABEL.findall(str(value or ""))]
    if not pts:
        return CENTRE
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def read_af_points(folder: Path) -> dict[str, tuple[float, float]]:
    """AF point per NEF in a folder, with ONE exiftool call (a call per file
    costs ~0.2 s each). Files without a point are simply absent."""
    exiftool = shutil.which("exiftool")
    if exiftool is None or not any(folder.glob("*.NEF")):
        return {}
    out = subprocess.run(
        [exiftool, "-q", "-j", "-ext", "NEF", "-AFPointsUsed", str(folder)],
        capture_output=True, text=True, check=False).stdout
    try:
        rows = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        return {}
    return {Path(r["SourceFile"]).name: af_label_xy(r.get("AFPointsUsed"))
            for r in rows if "SourceFile" in r}


def box_distance(det, px: float, py: float) -> float:
    """Distance from a point to a box (x1, y1, x2, y2, ...); 0 inside."""
    dx = max(det[0] - px, 0.0, px - det[2])
    dy = max(det[1] - py, 0.0, py - det[3])
    return (dx * dx + dy * dy) ** 0.5


def iou(a, b) -> float:
    """Intersection over union of two [x1, y1, x2, y2, ...] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - inter)
    return inter / union if union > 0 else 0.0


def subject_det(dets, af, width: float | None, height: float | None):
    """The detection nearest the AF point (0 when inside; ties -> higher
    conf). `dets` hold [x1, y1, x2, y2, conf] lists or objects with those
    attributes; `af` is normalised. Without a frame size there is no AF
    position in pixels, so fall back to the highest confidence."""
    if not dets:
        return None
    conf = (lambda d: d[4] if isinstance(d, (list, tuple)) else d.conf)
    if not (width and height):
        return max(dets, key=conf)
    px, py = (af or CENTRE)[0] * width, (af or CENTRE)[1] * height
    seq = (lambda d: d) if isinstance(dets[0], (list, tuple)) else (
        lambda d: (d.x1, d.y1, d.x2, d.y2, d.conf))
    return min(dets, key=lambda d: (box_distance(seq(d), px, py), -conf(d)))


def subject_from_cache(cache: dict | None):
    """subject_det over a per-frame cache entry (dets, af, width, height)."""
    c = cache or {}
    return subject_det(c.get("dets") or [], c.get("af"),
                       c.get("width"), c.get("height"))
