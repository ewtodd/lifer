"""Bird detection on previews via an off-the-shelf COCO detector (YOLO `bird`).

Birds are often ~1% of the frame, so the detector runs on the GPU at two
resolutions on the full-size preview (plus a zoomed pass around the AF point
when nothing is found there). Each detection is cropped to a square with ~20%
margin for classification. Frames with no detection are kept and recorded, not
dropped.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg
from ultralytics import YOLO

from .home import weights_dir
from .subject import box_distance, iou, subject_det

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
    af: tuple[float, float] = (0.5, 0.5)  # normalised AF point
    width: int = 0
    height: int = 0

    @property
    def best(self) -> Detection | None:
        """The frame's subject: the detection at the AF point (see subject.py)."""
        return subject_det(self.dets, self.af, self.width, self.height)


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


def _xyxy(d: Detection) -> tuple[float, float, float, float]:
    return (d.x1, d.y1, d.x2, d.y2)


def merge_dets(*groups: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Union of several passes' detections: a box overlapping a more
    confident one (IoU >= iou_thr) is the same bird seen twice, so only the
    more confident one is kept."""
    kept: list[Detection] = []
    for d in sorted((d for g in groups for d in g), key=lambda d: -d.conf):
        if all(iou(_xyxy(d), _xyxy(k)) < iou_thr for k in kept):
            kept.append(d)
    return kept


def resize_for_yolo(x: torch.Tensor, scale: float
                    ) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize a (1, 3, H, W) tensor in [0, 1] by `scale` and pad it to a
    multiple of 32 (the network stride) with YOLO's grey. Returns the tensor
    and the (left, top) padding, which boxes must subtract.

    The padding is split evenly like ultralytics' letterbox. Padding only the
    bottom/right shifted the image a few pixels and made the network lose
    clear birds (conf 0.59 -> 0.06), so this matters.

    Antialiased interpolation overshoots 1.0 slightly, and ultralytics then
    takes the tensor for 0-255 data and divides by 255 again (a black image,
    no detections), so the result is clamped to [0, 1].
    """
    h, w = x.shape[-2:]
    if scale != 1.0:
        x = F.interpolate(x, size=(round(h * scale), round(w * scale)),
                          mode="bilinear", antialias=True, align_corners=False)
        x = x.clamp(0.0, 1.0)
    h, w = x.shape[-2:]
    dw, dh = -w % 32, -h % 32
    left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
    x = F.pad(x, (left, dw - left, top, dh - top), value=114 / 255)
    return x, (left, top)


class BirdDetector:
    """Bird detector on the GPU: JPEG bytes -> nvjpeg -> two full-frame
    passes (1280 and `imgsz`) plus, when nothing sits at the AF point, a
    zoomed pass on a window around it. See `detect`."""

    weights_key: str
    # AF-window pass: a 2000 px square around the AF point, run at 1280 with a
    # lower threshold. It only runs when the full-frame passes found nothing
    # at the AF point (recovers small or hard birds the owner focused on).
    WINDOW = 2000
    WINDOW_IMGSZ = 1280
    WINDOW_CONF = 0.15
    # "found something at the AF point" = a box within this fraction of the
    # frame's long side
    AF_NEAR = 0.06

    def __init__(self, weights: str = "yolo11m.pt", imgsz: int = 2560, conf: float = 0.25):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; lifer requires the GPU")
        self.imgsz = imgsz
        self.conf = conf
        w = Path(weights).expanduser()
        if not w.is_absolute():
            w = weights_dir() / w  # default lives in ~/.cache/lifer
        self.weights_key = _weights_fingerprint(w)
        self.model = YOLO(str(w))

    @property
    def recipe(self) -> dict:
        """Everything besides weights/conf that shapes the cached detections;
        part of the per-frame cache key, so changing it re-detects."""
        return {"passes": sorted({1280, self.imgsz}),
                "window": [self.WINDOW, self.WINDOW_IMGSZ, self.WINDOW_CONF],
                "af_near": self.AF_NEAR, "v": 2}

    @staticmethod
    def decode(jpeg: bytes) -> torch.Tensor:
        """JPEG bytes -> (3, H, W) uint8 tensor on the GPU (nvjpeg; PIL for
        the odd file nvjpeg refuses)."""
        try:
            return decode_jpeg(torch.frombuffer(bytearray(jpeg), dtype=torch.uint8),
                               device="cuda", mode=ImageReadMode.RGB)
        except RuntimeError:
            arr = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            return torch.from_numpy(arr.copy()).permute(2, 0, 1).cuda()

    def _predict(self, x: torch.Tensor, scale: float, conf: float,
                 dx: float = 0.0, dy: float = 0.0) -> list[Detection]:
        """One YOLO pass over `x` at `scale`. Ultralytics reports boxes in the
        coordinates of the tensor it was given, so scale them back to `x`'s
        pixels and offset them by (dx, dy)."""
        inp, (left, top) = resize_for_yolo(x, scale)
        res = self.model.predict(inp, conf=conf, classes=[BIRD_CLASS],
                                 device=0, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return []
        xyxy = (res[0].boxes.xyxy.cpu().numpy()
                - np.array([left, top, left, top])) / scale
        confs = res[0].boxes.conf.cpu().numpy()
        return [Detection(float(a) + dx, float(b) + dy, float(c) + dx,
                          float(d) + dy, float(p))
                for (a, b, c, d), p in zip(xyxy, confs)]

    def detect(self, frame: torch.Tensor, af=(0.5, 0.5)) -> FrameDetections:
        """Birds in one decoded frame ((3, H, W) uint8 on the GPU), boxes in
        full-frame pixels. The 1280 pass stays beside the high-res one because
        a bird filling much of the frame can be missed at 2560."""
        h, w = frame.shape[-2:]
        x = frame.unsqueeze(0).float().div_(255.0)
        long_side = max(h, w)
        # never upscale: a small image gets one pass, not two identical ones
        scales = sorted({min(1.0, s / long_side) for s in (1280, self.imgsz)})
        dets = merge_dets(*(self._predict(x, s, self.conf) for s in scales))

        ax, ay = af[0] * w, af[1] * h
        if not any(box_distance(_xyxy(d), ax, ay) <= self.AF_NEAR * long_side
                   for d in dets):
            side_x, side_y = min(self.WINDOW, w), min(self.WINDOW, h)
            x0 = int(min(max(ax - side_x / 2, 0), w - side_x))
            y0 = int(min(max(ay - side_y / 2, 0), h - side_y))
            win = x[..., y0:y0 + side_y, x0:x0 + side_x]
            scale = min(1.0, self.WINDOW_IMGSZ / max(side_x, side_y))
            dets = merge_dets(dets, self._predict(win, scale, self.WINDOW_CONF,
                                                  x0, y0))
        return FrameDetections(frame_name="", dets=dets, af=tuple(af),
                               width=w, height=h)


def square_crop(image: Image.Image, det: Any, margin: float = 0.20) -> Image.Image:
    """Crop a detection to a square with ~margin on each side, clamped to frame."""
    return image.crop(square_box(image.size, det, margin))


def square_box(size: tuple[int, int], det: Any,
               margin: float = 0.20) -> tuple[int, int, int, int]:
    """The (left, top, right, bottom) square that square_crop cuts."""
    W, H = size
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    w, h = x2 - x1, y2 - y1
    s = max(w, h) * (1.0 + 2.0 * margin)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # clamp side so the square fits inside the frame
    s = min(s, W, H)
    x0 = min(max(cx - s / 2.0, 0.0), W - s)
    y0 = min(max(cy - s / 2.0, 0.0), H - s)
    return (int(round(x0)), int(round(y0)), int(round(x0 + s)), int(round(y0 + s)))


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
