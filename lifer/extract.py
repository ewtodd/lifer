"""Extract the embedded full-size JPEG (JpgFromRaw) and capture time from NEFs.

rawpy 0.27 exposes the embedded preview via ``RawPy.extract_thumb()`` (a
``Thumbnail`` with ``.data``) and EXIF-ish metadata via ``RawPy.other``.
Everything stays in memory; nothing is written to disk.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import rawpy
from PIL import Image


@dataclass
class Frame:
    path: Path
    name: str
    timestamp: datetime
    jpeg: bytes  # full-size embedded preview


def load_frame(path: Path) -> Frame:
    """Read one NEF: embedded JPEG bytes + DateTimeOriginal."""
    with rawpy.imread(str(path)) as raw:
        thumb = raw.extract_thumb()
        if thumb is None or not thumb.data:
            raise RuntimeError(f"no embedded JPEG in {path}")
        jpeg = bytes(thumb.data)
        ts = raw.other.timestamp

    # Fail loudly on a corrupt/empty preview rather than propagating garbage.
    with Image.open(io.BytesIO(jpeg)) as im:
        im.load()
        if im.size[0] < 512:  # a real JpgFromRaw on a D5600 is 6000x4000
            raise RuntimeError(f"embedded preview in {path} is only {im.size}")

    if ts is None:
        ts = datetime.fromtimestamp(path.stat().st_mtime)
    return Frame(path=path, name=path.name, timestamp=ts, jpeg=jpeg)


def jpeg_to_image(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def _exif_original_datetime(im: Image.Image) -> datetime | None:
    """EXIF DateTimeOriginal (else DateTime), or None."""
    try:
        exif = im.getexif()
    except Exception:
        return None
    value = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def load_frame_jpg(path: Path) -> Frame:
    """Read a standalone JPEG photo (camera dump, no NEF): file bytes +
    EXIF DateTimeOriginal. Same contract as load_frame."""
    jpeg = path.read_bytes()
    with Image.open(io.BytesIO(jpeg)) as im:
        im.load()
        if im.size[0] < 512:  # refuse thumbnails / watermarks as sources
            raise RuntimeError(f"JPEG {path} is only {im.size}")
        ts = _exif_original_datetime(im)
    if ts is None:
        ts = datetime.fromtimestamp(path.stat().st_mtime)
    return Frame(path=path, name=path.name, timestamp=ts, jpeg=jpeg)


def load_photo(path: Path) -> Image.Image:
    """Full-size image of a source photo, whatever its kind: the NEF's
    embedded JPEG, or the JPEG file itself. Use this anywhere a source
    photo is re-read (never assume which kind a run's frames are)."""
    if path.suffix.lower() == ".nef":
        return jpeg_to_image(load_frame(path).jpeg)
    with Image.open(io.BytesIO(path.read_bytes())) as im:
        return im.convert("RGB")
