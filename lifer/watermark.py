"""The owner's watermark on shared and published lifer photos.

The marks live on the read-only BIRDING share (``watermarkfull.png``, the
handwritten ``ETodd`` signature, and ``watermark.png``, the ``ET`` monogram).
lifer copies them once into ``LIFER_HOME/assets/`` and reads placement from
``LIFER_HOME/config.toml``'s ``[watermark]`` section. The mark is applied by
``lifer publish`` to site photos (*not* the <= 400 px thumbnails) and by
``lifer develop`` to the share copy of each edit, and is never drawn over the
bird.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from .config import load_config
from .home import lifer_home

BIRDING = Path.home() / "BIRDING"
ASSET_NAMES = ("watermarkfull.png", "watermark.png")

# Corner order. We start at the configured (preferred) corner and, when the
# mark would sit on the bird, walk to the opposite side, then the top, then
# the last one before falling back to the flattest corner.
_CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")


@dataclass
class Watermark:
    """Placement for the owner's mark; defaults match the agreed values."""

    enabled: bool = True
    file: str = "watermarkfull.png"
    monogram: str = "watermark.png"
    width_frac: float = 0.10
    min_width: int = 120
    opacity: float = 0.7
    corner: str = "bottom-right"
    inset_frac: float = 0.02

    @classmethod
    def from_config(cls) -> "Watermark":
        cfg = load_config().get("watermark", {})
        fields = cls.__dataclass_fields__
        # Coerce each configured value to the field's type (config.toml is
        # untyped; basedpyright needs the constructor to see concrete types).
        def coerce(k: str, v):
            t = getattr(fields[k].type, "__name__", str(fields[k].type))
            try:
                if t == "bool":
                    return bool(v)
                if t == "int":
                    return int(v)
                if t == "float":
                    return float(v)
                if t == "str":
                    return str(v)
            except (TypeError, ValueError):
                return None
            return None

        kwargs = {k: coerced for k, v in cfg.items()
                  if k in fields and (coerced := coerce(k, v)) is not None}
        return cls(**kwargs)


def ensure_assets() -> dict[str, Path]:
    """Copy the marks from the read-only share into LIFER_HOME/assets/ once.

    Reads from ~/BIRDING (never writes there); a mark already copied is left
    alone so the owner can swap it. Returns {asset-name: path} for those that
    are available (missing assets are just skipped).
    """
    assets = lifer_home() / "assets"
    out: dict[str, Path] = {}
    for name in ASSET_NAMES:
        dst = assets / name
        if not dst.is_file():
            src = BIRDING / name
            if src.is_file():
                assets.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        if dst.is_file():
            out[name] = dst
    return out


def map_box(box: "Sequence[float] | None",
            crop: "Sequence[float]",
            out_w: int, out_h: int) -> tuple[float, float, float, float] | None:
    """Map a full-frame bird box into an image cropped to ``crop`` then scaled.

    ``crop`` is (x1, y1, x2, y2) in the full-frame pixel space (the same space
    ``get_box`` returns) and ``out_w``/``out_h`` are the size of the image the
    crop was scaled to (aspect preserved, so one uniform scale). Returns the
    box in the output image's pixel coords, clamped to it, or None.
    """
    if box is None:
        return None
    cx1, cy1, cx2, cy2 = crop
    cw = max(cx2 - cx1, 1.0)
    ch = max(cy2 - cy1, 1.0)
    sx = out_w / cw
    sy = out_h / ch
    bx1, by1, bx2, by2 = box
    x1 = max(0.0, min((bx1 - cx1) * sx, float(out_w)))
    y1 = max(0.0, min((by1 - cy1) * sy, float(out_h)))
    x2 = max(0.0, min((bx2 - cx1) * sx, float(out_w)))
    y2 = max(0.0, min((by2 - cy1) * sy, float(out_h)))
    return (x1, y1, x2, y2)


def _rect(corner: str, w: int, h: int, image: Image.Image, inset: int):
    """(x1, y1, x2, y2) pixel rect for a w x h mark at a corner."""
    W, H = image.size
    x = inset if "left" in corner else W - inset - w
    y = inset if "top" in corner else H - inset - h
    return (x, y, x + w, y + h)


def _overlaps(rect, bird) -> bool:
    """True if two (x1, y1, x2, y2) rects intersect (bird may be None)."""
    if bird is None:
        return False
    x1, y1, x2, y2 = rect
    bx1, by1, bx2, by2 = bird
    return not (x2 <= bx1 or bx2 <= x1 or y2 <= by1 or by2 <= y1)


def _gradient_energy(image: Image.Image, rect) -> float:
    """Mean Sobel magnitude inside ``rect``: lower means a flatter background."""
    x1, y1, x2, y2 = (int(max(v, 0)) for v in rect)
    x2 = min(x2, image.width)
    y2 = min(y2, image.height)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return float("inf")
    region = image.convert("L").crop((x1, y1, x2, y2))
    import numpy as np
    a = np.asarray(region, dtype=np.float32)
    gx = float(np.abs(np.diff(a, axis=1)).mean())
    gy = float(np.abs(np.diff(a, axis=0)).mean())
    return gx + gy


def _pick_mark(assets: dict[str, Path], cfg: Watermark, image: Image.Image,
               inset: int):
    """Choose (signature|monogram, width, height) that fits, monogram if not."""
    sig = assets.get(cfg.file) or assets.get("watermarkfull.png")
    mono = assets.get(cfg.monogram) or assets.get("watermark.png")

    def size(path: Path) -> tuple[int, int]:
        mark = Image.open(path).convert("RGBA")
        w = int(round(cfg.width_frac * image.width))
        w = max(w, cfg.min_width)
        return (w, max(1, int(round(w * mark.height / mark.width))))

    def fits(w: int, h: int) -> bool:
        return (w + 2 * inset <= image.width) and (h + 2 * inset <= image.height)

    for path in (sig, mono):
        if path is not None:
            w, h = size(path)
            if fits(w, h):
                return path, w, h
    # Neither fits: clamp the monogram to the frame.
    path = mono or sig
    if path is None:
        return None, 0, 0
    mark = Image.open(path).convert("RGBA")
    w = max(cfg.min_width // 2, int(round(cfg.width_frac * image.width)))
    w = min(w, image.width - 2 * inset)
    h = max(1, int(round(w * mark.height / mark.width)))
    h = min(h, image.height - 2 * inset)
    return path, max(1, w), max(1, h)


def apply_watermark(image: Image.Image, bird, cfg: Watermark | None = None) -> Image.Image:
    """Return ``image`` with the owner's mark placed to avoid ``bird``.

    ``bird`` is the bird box in the image's own pixel coords (after the crop),
    or None when unknown. The signature is used unless it is too wide/tall to
    fit, in which case the monogram is. If every corner would sit on the bird,
    the flattest corner (lowest gradient energy) is chosen.
    """
    cfg = cfg or Watermark.from_config()
    if not cfg.enabled:
        return image
    assets = ensure_assets()
    if not assets:
        return image

    inset = int(round(cfg.inset_frac * min(image.size)))
    path, mw, mh = _pick_mark(assets, cfg, image, inset)
    if path is None:
        return image

    if cfg.corner not in _CORNERS:
        cfg.corner = "bottom-right"
    order = _CORNERS[_CORNERS.index(cfg.corner):] + _CORNERS[:_CORNERS.index(cfg.corner)]

    rect: tuple[int, int, int, int] | None = None
    for corner in order:
        cand = _rect(corner, mw, mh, image, inset)
        if not _overlaps(cand, bird):
            rect = cand
            break
    if rect is None:
        rect = min((_rect(c, mw, mh, image, inset) for c in _CORNERS),
                   key=lambda r: _gradient_energy(image, r))

    mark = Image.open(path).convert("RGBA").resize((mw, mh), Image.LANCZOS)
    alpha = mark.getchannel("A").point(lambda a: int(a * cfg.opacity))
    mark = mark.copy()
    mark.putalpha(alpha)

    base = image.convert("RGBA")
    base.paste(mark, (rect[0], rect[1]), mark)
    return base.convert("RGB")
