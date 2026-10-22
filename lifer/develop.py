"""Editing half of lifer: `lifer develop` (+ `lifer develop --benchmark`).

Takes a run folder already processed by `lifer run` and develops the raw
photos into final edits with darktable-cli, in a style taken from an edited
sidecar (the ``--style-sidecar`` template).  Each chosen NEF is copied under
LIFER_HOME first so darktable never touches the source folder; a per-image
darktable XMP sidecar is generated (style template + global exposure metered
on the bird box + a bird-local lift, crop from the detection box) and
darktable-cli renders it.  The generated .xmp is kept beside each output.

The benchmark mode develops frames against a set of reference exports,
framing each with the reference's own crop so the three panels
(camera JPEG | lifer | reference) line up, then writes a compare page with
measured bird/background luminance ratio and chroma (CIELAB) for lifer vs
the reference.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import sqlite3
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .boxes import get_box, open_db, representative_frame
from .subject import subject_from_cache
from .extract import load_frame
from .home import effective_trip_id, lifer_home
from .dxmp import (NO_MASK_BLENDOP, Module, crop_params, decode_params,
                   exposure_params, toneequal_params, write_sidecar)
from .dtstyle import build_template
from .watermark import Watermark, apply_watermark, map_box

# Soft-crop threshold: Laplacian variance of the bird crop below this is
# treated as out-of-focus / motion blur and skipped (from runs/cache).
SHARPEN_SOFT = 80.0

# Bird occupies this fraction of the developed frame's width.
BIRD_FRAC = 0.40

# Per-image EV range we allow ourselves (brakes, not a target) and the
# middle-grey luminance (linear) we drive the bird toward.  BIRD_TARGET is
# calibrated so a normally-exposed bird meters to ~0.7 EV (darktable's
# scene-referred default); a shaded bird gets more, a bright one less.
EV_MIN = -0.5
EV_MAX = 1.2
BIRD_TARGET = 0.086

# Bird-local lift, in EV, applied to the shadow/midtone bands by the tone
# equalizer (content-aware: it lifts the shaded bird, not the sunlit bg).
BIRD_LIFT = 0.5

# Output framing.
LONG_EDGE = 3000
JPEG_Q = 92


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _cache_frame(cache_dir: Path, name: str) -> dict | None:
    """The per-frame cache JSON for a NEF name, or None if missing."""
    stem = name[:-4] if name.lower().endswith(".nef") else name
    p = cache_dir / f"{stem}.json"
    if not p.is_file():
        return None
    with p.open() as f:
        return json.load(f)


def _as_nef(name: str) -> str:
    """Return ``name`` with a ``.NEF`` suffix (source files are uppercase)."""
    return name if name.lower().endswith(".nef") else f"{name}.NEF"


def _adj_top1(adj_top3: str) -> tuple[str, float]:
    """Split "Name (0.996); Name2" into (top name, top prob)."""
    first = adj_top3.split(";")[0].strip()
    name = first.rsplit("(", 1)[0].strip()
    prob = float(first.rsplit("(", 1)[1].rstrip(")"))
    return name, prob


def _pick_frame_names(
    candidates: list[dict], cache_dir: Path, species: str, n: int,
    folder: str = "", conn: sqlite3.Connection | None = None,
) -> list[str]:
    """Sharpest representatives: one representative_frame per encounter.

    Goes through ``representative_frame`` (task 12) so a photo the owner picked
    for an encounter is developed instead of whatever the detector scored as
    sharpest; the candidate rows are per-encounter.
    """
    conn = conn or open_db()
    scored: list[tuple[float, float, str]] = []
    for c in candidates:
        top1, prob = _adj_top1(c["adj_top3"])
        if top1.lower() != species.lower():
            continue
        frames = [x for x in (c.get("frames") or "").split(",") if x]
        frame = representative_frame(conn, folder, frames, c["best_frame"])
        data = _cache_frame(cache_dir, frame)
        if data is None:
            continue
        sharp = float(data.get("sharpness", -1.0))
        if sharp < 0:
            continue
        scored.append((prob, sharp, frame))
    scored.sort(key=lambda t: (-t[0], -t[1]))

    picked: list[str] = []
    for _prob, sharp, frame in scored:
        if sharp < SHARPEN_SOFT:
            continue
        picked.append(frame)
        if len(picked) >= n:
            break
    return picked


def _best_box(cache: dict | None) -> Box | None:
    """The subject detection box (nearest the AF point) from a cache entry."""
    best = subject_from_cache(cache)
    return Box(best[0], best[1], best[2], best[3]) if best else None


def _compute_crop(box: Box, img_w: float, img_h: float) -> Box:
    """Crop rectangle (in image pixel coords) that frames the whole bird."""
    portrait = box.h > box.w * 1.05
    aspect = (4.0 / 5.0) if portrait else (3.0 / 2.0)
    frame_w = box.w / BIRD_FRAC
    frame_h = frame_w / aspect
    need_h = box.h * 1.30
    if frame_h < need_h:
        frame_h = need_h
        frame_w = frame_h * aspect
    if frame_w > img_w:
        frame_h *= img_w / frame_w
        frame_w = img_w
    if frame_h > img_h:
        frame_w *= img_h / frame_h
        frame_h = img_h
    left = min(max(box.cx - frame_w / 2.0, 0.0), img_w - frame_w)
    top = min(max(box.cy - frame_h / 2.0, 0.0), img_h - frame_h)
    return Box(left, top, left + frame_w, top + frame_h)


def _linear_mean(im: Image.Image) -> float:
    """Mean luminance (linear, 0-1) of an image."""
    im = im.convert("L")
    h = im.histogram()
    total = sum(h)
    if total == 0:
        return 0.0
    s = sum(i * c for i, c in enumerate(h)) / total / 255.0
    return s if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4


def _exposure_ev(before: Image.Image, bird: Box) -> float:
    """EV offset so the bird sits near middle grey, metered on the bird box."""
    b = before.convert("RGB")
    bird_crop = b.crop((int(bird.x1), int(bird.y1), int(bird.x2), int(bird.y2)))
    bird_mean = _linear_mean(bird_crop)
    ev = math.log2(BIRD_TARGET / max(bird_mean, 1e-4))
    # dampen toward the scene-referred default so a back-lit bird does not
    # blow out the whole frame; the bird-local lift handles it instead
    ev = 0.7 + (ev - 0.7) * 0.5
    return max(EV_MIN, min(ev, EV_MAX))


def _bird_lift(before: Image.Image, bird: Box) -> float:
    """Adaptive bird-local lift (EV) for the tone equalizer.

    A lift is only useful on a strongly back-lit bird (a bright background
    against a shaded subject), so it is gated on the bird-vs-background
    contrast of the camera JPEG: no lift for a normal scene, more the darker
    the bird is relative to the background.
    """
    b = before.convert("RGB")
    bird_lum = _linear_mean(b.crop((int(bird.x1), int(bird.y1),
                                    int(bird.x2), int(bird.y2))))
    bg = _linear_mean(b)
    contrast = math.log2(bg / max(bird_lum, 1e-4))
    return max(0.0, min(0.6 * (contrast - 1.2), 1.5))


def _iso_nr_luma(iso: int) -> float:
    if iso <= 400:
        return 8.0
    if iso <= 1600:
        return 8.0 + (iso - 400) / 1200.0 * 20.0
    return 28.0 + min((iso - 1600) / 6400.0, 1.0) * 45.0


def _read_iso(nef: Path) -> int:
    """ISO from EXIF (exiftool is on the package's PATH via the flake)."""
    exif = shutil.which("exiftool")
    if not exif:
        return 0
    out = subprocess.run([exif, "-s", "-S", "-ISO", str(nef)],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return 0


def _fractional_crop(im: Image.Image, frac: Box) -> Image.Image:
    """Crop `im` using a box given in normalized [0,1] coordinates."""
    w, h = im.size
    x1 = max(0, int(round(frac.x1 * w)))
    y1 = max(0, int(round(frac.y1 * h)))
    x2 = min(w, int(round(frac.x2 * w)))
    y2 = min(h, int(round(frac.y2 * h)))
    return im.crop((x1, y1, x2, y2))


def _resize_long_edge(im: Image.Image, long_edge: int) -> Image.Image:
    w, h = im.size
    long_side = max(w, h)
    if long_side <= long_edge:
        return im
    scale = long_edge / long_side
    return im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                     Image.LANCZOS)


def _extract_jpeg(nef: Path) -> Image.Image:
    """The camera's embedded full-size JPEG for a NEF (the 'before' image)."""
    frame = load_frame(nef)
    return Image.open(io.BytesIO(frame.jpeg)).convert("RGB")


# --- darktable develop -------------------------------------------------------

def _per_image_mods(template: list[Module], crop: Box, ev: float,
                    lift: float = BIRD_LIFT) -> list[Module]:
    """Template + per-image exposure, bird-local lift and crop."""
    noblend = decode_params(NO_MASK_BLENDOP)
    base = [m for m in template
            if m.operation not in ("bilat", "colorbalancergb", "colorzones",
                                   "sharpen")]
    style = [m for m in template
             if m.operation in ("bilat", "colorbalancergb", "colorzones",
                                "sharpen")]
    out = list(base)
    out.append(Module("exposure", 6, exposure_params(ev), blendop_params=noblend))
    out.append(Module("toneequal", 1, toneequal_params(lift),
                      blendop_params=noblend))
    # darktable crop params are (left, top, right, bottom) normalized corners.
    out.append(Module("crop", 1, crop_params(crop.x1, crop.y1, crop.x2, crop.y2)))
    out.extend(style)
    return out


def _develop_darktable(nef: Path, xmp: Path, out: Path,
                       modules: list[Module]) -> None:
    """Render one NEF with the given modules and write the sidecar beside out."""
    write_sidecar(xmp, modules, [], derived_from=nef.name)
    cli = shutil.which("darktable-cli") or "darktable-cli"
    subprocess.run(
        [cli, str(nef), str(xmp), str(out),
         "--core", "--library", ":memory:",
         "--conf", "write_sidecar_files=never"],
        check=True, capture_output=True, text=True)


def _render_frame(nef: Path, xmp: Path, out_full: Path, crop: Box, ev: float,
                  lift: float = BIRD_LIFT, template: list[Module] | None = None) -> Path:
    """Build the per-image XMP, render via darktable-cli; return the full crop."""
    if template is None:
        template = build_template()
    mods = _per_image_mods(template, crop, ev, lift)
    _develop_darktable(nef, xmp, out_full, mods)
    return out_full


# --- benchmark / measurement -------------------------------------------------

def _srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(rgb):
    lin = _srgb_to_lin(rgb)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    d65 = np.array([0.95047, 1.0, 1.08883])
    f = xyz / d65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(f > eps, np.cbrt(f), (kappa * f + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _lab_stats(p: Path, bird_box: tuple[float, float, float, float]):
    """Mean Lab L and C* over the bird box and the rest of the image (bg)."""
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    h, w = a.shape[:2]
    x0, x1, y0, y1 = (int(bird_box[0] * w), int(bird_box[1] * w),
                      int(bird_box[2] * h), int(bird_box[3] * h))
    L, aa, bb = _rgb_to_lab(a)
    C = np.sqrt(aa ** 2 + bb ** 2)
    bird = np.zeros((h, w), bool)
    bird[y0:y1, x0:x1] = True
    lb, cb = float(L[bird].mean()), float(C[bird].mean())
    lbg, cbg = float(L[~bird].mean()), float(C[~bird].mean())
    return {"bird_L": lb, "bg_L": lbg, "ratio": lb / max(lbg, 1e-3),
            "bird_C": cb, "bg_C": cbg}


def _write_benchmark_html(bench: Path, rows: list[dict]) -> None:
    """Three-panel (camera | lifer | reference) compare page with the numbers."""
    row_html = []
    for r in rows:
        row_html.append(f"""<tr>
  <td>{r["frame"]}</td>
  <td><img src="{r["cam_img"]}" title="camera JPEG"></td>
  <td><img src="{r["lifer_img"]}" title="lifer"></td>
  <td><img src="{r["ref_img"]}" title="reference"></td>
  <td class="n">{r["b"]["ratio"]:.2f}</td>
  <td class="n">{r["o"]["ratio"]:.2f}</td>
  <td class="n">{r["b"]["bird_C"]:.1f}</td>
  <td class="n">{r["o"]["bird_C"]:.1f}</td>
  <td class="{r["class"]}">{r["pass"]}</td>
</tr>""")
    html = """<!doctype html><html><head><meta charset="utf-8">
<title>lifer develop benchmark</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#ddd;margin:1.5rem}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #333;padding:.4rem;text-align:left}
td img{display:block;max-width:16rem;height:auto}
th.n,td.n{text-align:right}
.pass{color:#7c7}
.fail{color:#c77}
</style></head><body>
<h1>lifer develop &mdash; benchmark vs reference darktable exports</h1>
<p>Each row: the same raw developed by lifer's pipeline (crop = the
reference's own crop so the panels line up) next to the reference export.
The metric is the bird/background luminance ratio (CIELAB L) and mean bird
chroma (C*); a row passes when both are within 25% of the reference's.</p>
<table>
<tr><th>frame</th><th>camera</th><th>lifer</th><th>reference</th>
<th>ratio b</th><th>ratio r</th><th>chroma b</th><th>chroma r</th><th>?</th></tr>
<!--ROWS-->
</table>
</body></html>"""
    html = html.replace("<!--ROWS-->", "\n".join(row_html))
    (bench / "compare.html").write_text(html)


def _write_compare_html(out_dir: Path, entries: list[dict]) -> None:
    """Static offline before/after compare page with a drag slider."""
    pair_html = []
    for e in entries:
        pair_html.append(f"""
<section class="pair"><h3>{e["frame"]}</h3>
  <div class="slider">
    <div class="stack">
      <img class="after" src="{e["after"]}" alt="developed">
      <div class="beforewrap">
        <img class="before" src="{e["before"]}" alt="camera jpeg">
      </div>
      <div class="handle"></div>
    </div>
    <input type="range" class="slider" min="0" max="100" value="50" aria-label="before after">
  </div>
  <p class="set">{e["settings"]}</p>
</section>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>lifer develop &mdash; {entries[0]["folder"] if entries else ""}</title>
<style>
body{{font-family:system-ui, sans-serif;background:#111;color:#ddd;margin:1.5rem}}
h1{{font-size:1.3rem}}
h3{{margin:1.6rem 0 .4rem}}
.pair{{margin-bottom:2rem}}
.set{{color:#9a9a9a;font-size:.85rem;max-width:60rem;white-space:pre-wrap}}
.slider{{position:relative;display:inline-block;max-width:100%}}
.stack{{position:relative;overflow:hidden;max-width:70rem}}
.stack img{{display:block}}
.stack .after{{width:100%;height:auto}}
.beforewrap{{position:absolute;inset:0;clip-path:inset(0 100% 0 0)}}
.before{{display:block;height:100%;width:auto;max-width:none}}
input[type=range]{{width:100%;margin-top:.4rem}}
</style></head><body>
<h1>lifer develop &mdash; {entries[0]["folder"] if entries else ""}</h1>
<p>Drag the slider: left is the camera's embedded JPEG, right is the developed RAW.</p>
{''.join(pair_html)}
<script>
document.querySelectorAll('.slider').forEach(el => {{
  if (el.tagName === 'INPUT') {{
    const stack = el.previousElementSibling.querySelector('.stack');
    const beforewrap = stack.querySelector('.beforewrap');
    const before = stack.querySelector('.before');
    const after = stack.querySelector('.after');
    before.onload = () => {{
      beforewrap.style.clipPath = 'inset(0 ' + (100 - el.value) + '% 0 0)';
    }};
    before.src = before.getAttribute('src');
    after.onload = () => {{
      before.style.height = after.height + 'px';
      const ar = before.naturalWidth / before.naturalHeight;
      before.style.width = (after.height * ar) + 'px';
      beforewrap.style.clipPath = 'inset(0 ' + (100 - el.value) + '% 0 0)';
    }};
    el.addEventListener('input', () => {{
      beforewrap.style.clipPath = 'inset(0 ' + (100 - el.value) + '% 0 0)';
    }});
  }}
}});
</script>
</body></html>
"""
    (out_dir / "compare.html").write_text(html)


def _bird_det_box(cache_dir: Path, name: str) -> tuple[float, float, float, float] | None:
    """Normalized [x0,x1,y0,y1] bird box for a frame (full-frame coords)."""
    data = _cache_frame(cache_dir, name)
    box = _best_box(data)
    if box is not None:
        return (box.x1 / 6000, box.x2 / 6000, box.y1 / 4000, box.y2 / 4000)
    return None


def _find_reference_export(ref_dir: Path, stem: str) -> Path | None:
    """The reference export for a frame; handles `_DSC8819_01.jpg`-style names."""
    direct = ref_dir / f"{stem}.jpg"
    if direct.is_file():
        return direct
    matches = sorted(ref_dir.glob(f"{stem}*.jpg"),
                     key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _reference_crops(reference_sidecars: Path) -> dict[str, tuple[float, float, float, float]]:
    """Final-crop (left, top, right, bottom) per frame from the reference sidecars."""
    from .dtstyle import load_sidecar
    out: dict[str, tuple[float, float, float, float]] = {}
    for x in sorted(reference_sidecars.glob("*.xmp")):
        try:
            modules, _masks, _m = load_sidecar(x)
        except Exception:
            continue
        crops = [m for m in modules if m.operation == "crop"]
        if not crops:
            continue
        c = crops[-1]  # the last (final) crop
        vals = struct.unpack("<4f", c.params[:16])
        out[x.name.removesuffix(".NEF.xmp")] = tuple(vals)
    return out


def develop(source_folder: str, species: str = "", n: int = 5,
            frames: list[str] | None = None, lift: float = BIRD_LIFT,
            style_sidecar: Path | None = None,
            root: str | Path | None = None,
            conn: sqlite3.Connection | None = None) -> int:
    """Develop a set of frames with darktable-cli, in the style of ``style_sidecar``.

    Frames and crops go through the shared task-11/12 functions
    (``representative_frame`` for the frame to develop, ``get_box`` for the
    crop) so an owner's fixed crop or chosen photo is what gets developed, and
    each edit gets a watermarked ``share/`` copy alongside the clean master.
    """
    if style_sidecar is None:
        raise SystemExit("error: a style sidecar is required "
                         "(--style-sidecar, an edited .xmp)")
    home = lifer_home()
    source = Path(source_folder).expanduser()
    folder = effective_trip_id(source, root)
    conn = conn or open_db()
    run_dir = home / "runs" / folder
    cache_dir = home / "cache" / folder
    src_dir = home / "develop" / folder / "src"
    out_dir = home / "edits" / folder
    before_dir = out_dir / "before"
    xmp_dir = out_dir / "xmp"
    share_dir = out_dir / "share"
    for d in (src_dir, out_dir, before_dir, xmp_dir, share_dir):
        d.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "candidates.csv"
    if not csv_path.is_file():
        raise SystemExit(f"error: no run results at {run_dir}/candidates.csv")
    with open(csv_path, newline="") as f:
        candidates = list(csv.DictReader(f))

    if frames:
        for name in frames:
            if _cache_frame(cache_dir, name) is None:
                raise SystemExit(f"error: frame '{name}' not found in {cache_dir}")
        selected = list(frames)
    else:
        selected = _pick_frame_names(candidates, cache_dir, species, n,
                                     folder=folder, conn=conn)
        if not selected:
            raise SystemExit(f"error: no sharp frames of '{species}' found in {folder}")
        print(f"selected {len(selected)} frame(s): {', '.join(selected)}")

    # normalise to the on-disk .NEF name (validation above accepts either form)
    selected = [_as_nef(n) for n in selected]

    template = build_template(style_sidecar)
    entries: list[dict] = []
    for name in selected:
        src_nef = src_dir / name
        if not src_nef.is_file():
            shutil.copy2(source / name, src_nef)

        data = _cache_frame(cache_dir, name) or {}
        det_box = get_box(folder, name, conn, dets=data.get("dets", []),
                          af=data.get("af"),
                          size=(data.get("width"), data.get("height")))
        if det_box is None:
            print(f"skip {name}: no detection")
            continue
        box = Box(det_box.x1, det_box.y1, det_box.x2, det_box.y2)

        iso = _read_iso(src_nef)
        before_full = _extract_jpeg(src_nef)
        bw, bh = before_full.size
        crop = _compute_crop(box, bw, bh)
        ev = _exposure_ev(before_full, box)
        lift = _bird_lift(before_full, box)

        stem = Path(name).stem
        xmp_path = xmp_dir / f"{stem}.xmp"
        full = out_dir / f"{stem}._full.jpg"
        # darktable crop params are normalized; convert the pixel crop box
        frac = Box(crop.x1 / bw, crop.y1 / bh, crop.x2 / bw, crop.y2 / bh)
        _render_frame(src_nef, xmp_path, full, frac, ev, lift, template)

        # developed output: same fractional crop, long edge 3000, q92
        after = _resize_long_edge(Image.open(full), LONG_EDGE)
        jpg_path = out_dir / f"{stem}.jpg"
        after.convert("RGB").save(jpg_path, "JPEG", quality=JPEG_Q)
        full.unlink(missing_ok=True)

        # watermarked share copy beside the clean master (task 14): the same
        # edit, but with the owner's mark placed to keep it off the bird.
        bird = map_box((box.x1, box.y1, box.x2, box.y2),
                       (crop.x1, crop.y1, crop.x2, crop.y2),
                       after.width, after.height)
        share_img = apply_watermark(after, bird, Watermark.from_config())
        share_path = share_dir / f"{stem}.jpg"
        share_img.save(share_path, "JPEG", quality=JPEG_Q)

        before = _resize_long_edge(_fractional_crop(before_full, frac), LONG_EDGE)
        before_path = before_dir / f"{stem}.jpg"
        before.convert("RGB").save(before_path, "JPEG", quality=90)

        aspect = "4:5" if crop.h > crop.w else "3:2"
        settings = (f"ISO {iso}; WB camera; {ev:+.2f} EV; "
                    f"bird lift {lift:+.2f}; style template; "
                    f"crop {crop.w:.0f}x{crop.h:.0f} ({aspect})")
        entries.append({
            "frame": name,
            "before": f"before/{stem}.jpg",
            "after": f"{stem}.jpg",
            "settings": settings,
            "folder": folder,
        })
        print(f"developed {name} -> {jpg_path} ({settings})")

    if entries:
        _write_compare_html(out_dir, entries)
    print(f"compare -> {out_dir}/compare.html")
    return 0


def benchmark(source_folder: str, reference_dir: Path,
              frames: list[str] | None = None,
              reference_sidecars: Path | None = None,
              lift: float = BIRD_LIFT,
              style_sidecar: Path | None = None,
              root: str | Path | None = None) -> int:
    """Develop frames against a reference export set and compare lifer vs it.

    Uses the reference's own crop for framing parity so the three panels
    align; measures bird/background Lab-L ratio and bird chroma (C*).
    """
    if style_sidecar is None:
        raise SystemExit("error: a style sidecar is required "
                         "(--style-sidecar, an edited .xmp)")
    home = lifer_home()
    source = Path(source_folder).expanduser()
    folder = effective_trip_id(source, root)
    cache_dir = home / "cache" / folder
    src_dir = home / "develop" / folder / "src"
    bench = home / "edits" / "benchmark"
    for d in (src_dir, bench):
        d.mkdir(parents=True, exist_ok=True)
    if frames is None:
        # default: benchmark the frames that have a reference export
        stems = sorted({Path(p.name).stem for p in reference_dir.glob("*.jpg")})
        if not stems:
            raise SystemExit(f"error: no reference exports in {reference_dir}")
        # darktable export names may carry a taken-name suffix (_DSC8819_01)
        frames = [re.sub(r"_\d+$", "", s) + ".NEF" for s in stems]
    ref_crops = _reference_crops(reference_sidecars or reference_dir)
    template = build_template(style_sidecar)

    rows: list[dict] = []
    for name in frames:
        stem = Path(name).stem
        src_nef = src_dir / name
        if not src_nef.is_file():
            shutil.copy2(source / name, src_nef)

        before_full = _extract_jpeg(src_nef)
        bw, bh = before_full.size

        cro = ref_crops.get(name)
        if cro is None:
            cro = ref_crops.get(stem)
        if cro is None:
            raise SystemExit(f"error: no reference crop for {name} (checked "
                             f"{reference_sidecars or reference_dir})")
        # reference crop params are (left, top, right, bottom) normalized.
        left, top, right, bottom = cro
        crop = Box(left * bw, top * bh, right * bw, bottom * bh)

        data = _cache_frame(cache_dir, name)
        box = _best_box(data)
        if box is None:
            ev = 0.7
            lift = 0.0
        else:
            ev = _exposure_ev(before_full, box)
            lift = _bird_lift(before_full, box)

        xmp_path = bench / f"{stem}.xmp"
        full = bench / f"{stem}._full.jpg"
        frac = Box(left, top, right, bottom)  # normalized crop corners
        _render_frame(src_nef, xmp_path, full, frac, ev, lift, template)

        # lifer edit: same crop, long edge 3000, q92
        lifer = _resize_long_edge(Image.open(full), LONG_EDGE)
        lifer_path = bench / f"{stem}.lifer.jpg"
        lifer.convert("RGB").save(lifer_path, "JPEG", quality=JPEG_Q)
        full.unlink(missing_ok=True)

        ref_jpg = _find_reference_export(reference_dir, stem)
        if ref_jpg is None:
            print(f"skip {name}: no reference export for {stem} in {reference_dir}")
            continue

        # reference export may be full-res; match our framing/long-edge
        ref = _resize_long_edge(Image.open(ref_jpg), LONG_EDGE)
        ref_path = bench / f"{stem}.ref.jpg"
        ref.convert("RGB").save(ref_path, "JPEG", quality=JPEG_Q)

        cam = _resize_long_edge(_fractional_crop(before_full, frac), LONG_EDGE)
        cam_path = bench / f"{stem}.cam.jpg"
        cam.convert("RGB").save(cam_path, "JPEG", quality=90)

        # bird box: detection box mapped into the crop, else a central box
        det = _bird_det_box(cache_dir, name)
        if det is not None:
            bx0, bx1, by0, by1 = det
            bbox = ((bx0 - left) / (right - left), (bx1 - left) / (right - left),
                    (by0 - top) / (bottom - top), (by1 - top) / (bottom - top))
        else:
            bbox = (0.30, 0.68, 0.30, 0.62)
        bbox = tuple(min(max(v, 0.01), 0.99) for v in bbox)

        b = _lab_stats(lifer_path, bbox)
        o = _lab_stats(ref_path, bbox)
        ok = abs(b["ratio"] / o["ratio"] - 1) <= 0.25 and \
             abs(b["bird_C"] / o["bird_C"] - 1) <= 0.25
        rows.append({
            "frame": stem,
            "cam_img": f"{stem}.cam.jpg",
            "lifer_img": f"{stem}.lifer.jpg",
            "ref_img": f"{stem}.ref.jpg",
            "b": b, "o": o, "pass": "pass" if ok else "FAIL",
            "class": "pass" if ok else "fail",
        })
        print(f"{stem}: ratio b/o = {b['ratio']:.2f}/{o['ratio']:.2f} "
              f"chroma b/o = {b['bird_C']:.1f}/{o['bird_C']:.1f} "
              f"-> {'pass' if ok else 'FAIL'}")

    if rows:
        _write_benchmark_html(bench, rows)
    print(f"benchmark compare -> {bench}/compare.html")
    return 0

