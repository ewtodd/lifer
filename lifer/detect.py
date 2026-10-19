"""Bird detection on previews via an off-the-shelf COCO detector (YOLO `bird`).

Birds are often ~1% of the frame, so we run the detector at a high input
resolution (imgsz>=1280) on the full-size preview. Each detection is cropped to
a square with ~20% margin for classification. Frames with no detection are kept
and recorded, not dropped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .home import weights_dir

BIRD_CLASS = 14  # COCO class id for "bird"


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


@dataclass
class FrameDetections:
    frame_name: str
    dets: list[Detection] = field(default_factory=list)

    @property
    def best(self) -> Detection | None:
        return max(self.dets, key=lambda d: d.conf) if self.dets else None


def _weights_fingerprint(path: Path) -> str:
    """Path-independent ID for a weights file: name + size + content hash.

    The detection cache is keyed on this instead of the weights' absolute
    path, so moving the cache dir (or renaming the state dirs) doesn't
    invalidate cached detections: the same file at a new location still yields
    the same key.
    """
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return f"{path.name}:{path.stat().st_size}:{h.hexdigest()[:16]}"


class BirdDetector:
    weights_key: str

    def __init__(self, weights: str = "yolo11m.pt", imgsz: int = 1280, conf: float = 0.25):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; lifer requires the GPU")
        self.imgsz = imgsz
        self.conf = conf
        w = Path(weights).expanduser()
        if not w.is_absolute():
            w = weights_dir() / w  # default lives in ~/.cache/lifer
        self.weights_key = _weights_fingerprint(w)
        self.model = YOLO(str(w))

    def detect(self, image: Image.Image) -> FrameDetections:
        """Run the detector on one preview, return bird detections."""
        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[BIRD_CLASS],
            device=0,
            verbose=False,
        )
        dets: list[Detection] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c)))
        return FrameDetections(frame_name="", dets=dets)


def square_crop(image: Image.Image, det: Any, margin: float = 0.20) -> Image.Image:
    """Crop a detection to a square with ~margin on each side, clamped to frame."""
    W, H = image.size
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    w, h = x2 - x1, y2 - y1
    s = max(w, h) * (1.0 + 2.0 * margin)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # clamp side so the square fits inside the frame
    s = min(s, W, H)
    x0 = min(max(cx - s / 2.0, 0.0), W - s)
    y0 = min(max(cy - s / 2.0, 0.0), H - s)
    box = (int(round(x0)), int(round(y0)), int(round(x0 + s)), int(round(y0 + s)))
    return image.crop(box)


def sharpness(crop: Image.Image) -> float:
    """Variance of Laplacian on the (bird) crop — a focus/sharpness proxy."""
    import cv2

    arr = np.asarray(crop.convert("L"))
    if arr.shape[0] > 512:  # downscale large crops for a comparable measure
        arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def usable_crop(crop: Image.Image, min_variance: float = 1.0) -> bool:
    """Whether a crop is worth classifying: non-empty and not near-uniform.

    A zero-area box or a blank/black frame (e.g. a missing embedded JPEG)
    must not be fed to BioCLIP: with nothing in the image the encoder picks
    an arbitrary "top species". Near-uniform = grayscale pixel variance
    below `min_variance`, measured on a downsampled copy (uniformity does
    not need full resolution; real bird crops measure in the hundreds).
    """
    if crop is None or crop.width < 2 or crop.height < 2:
        return False
    small = crop.convert("L")
    if max(small.size) > 256:
        small = small.resize((min(256, small.width), min(256, small.height)))
    return float(np.asarray(small).var()) >= min_variance
