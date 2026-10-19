"""darktable sidecar XMP codec and generator.

darktable stores each module's parameter struct in the sidecar XMP, encoded as
a hex string when it is small (<=100 bytes) or as ``gz`` + a 2-digit
compression factor + base64(zlib) when it is large.  The module param structs
are the raw C structs; we only need to read/write a handful for the develop
pipeline (exposure, crop, tone equalizer, bilat, sharpen) and to copy the rest
verbatim from edited sidecars.

``write_sidecar`` emits a complete, valid darktable 5.x XMP that ``darktable-cli``
feeds to its pixelpipe.
"""

from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

# darktable's canonical iop order (from a real sidecar). Darktable inserts each
# module into the pixelpipe at the position implied by this list, regardless of
# the order in ``history``.
IOP_ORDER = (
    "rawprepare,0,invert,0,temperature,0,highlights,0,cacorrect,0,hotpixels,0,"
    "rawdenoise,0,demosaic,0,denoiseprofile,0,bilateral,0,rotatepixels,0,"
    "scalepixels,0,lens,0,cacorrectrgb,0,hazeremoval,0,ashift,0,overlay,0,"
    "enlargecanvas,0,flip,0,clipping,0,liquify,0,spots,0,retouch,0,exposure,0,"
    "mask_manager,0,tonemap,0,toneequal,0,crop,0,graduatednd,0,profile_gamma,0,"
    "equalizer,0,colorin,0,channelmixerrgb,0,diffuse,0,censorize,0,primaries,0,"
    "negadoctor,0,blurs,0,nlmeans,0,colorchecker,0,defringe,0,atrous,0,lowpass,0,"
    "highpass,0,sharpen,0,colortransfer,0,colormapping,0,channelmixer,0,basicadj,0,"
    "colorbalance,0,colorequal,0,colorbalancergb,0,rgbcurve,0,rgblevels,0,basecurve,0,"
    "filmic,0,sigmoid,0,filmicrgb,0,lut3d,0,colisa,0,tonecurve,0,levels,0,shadhi,0,"
    "zonesystem,0,globaltonemap,0,relight,0,bilat,0,colorcorrection,0,colorcontrast,0,"
    "velvia,0,vibrance,0,colorzones,0,bloom,0,colorize,0,lowlight,0,monochrome,0,"
    "grain,0,soften,0,splittoning,0,vignette,0,colorreconstruct,0,colorout,0,clahe,0,"
    "finalscale,0,overexposed,0,rawoverexposed,0,dither,0,borders,0,watermark,0,gamma,0"
)

# A "no mask, neutral blend" blendop_params blob (the module default). Identical
# for the scene-referred default exposure and filmicrgb that darktable emits.
NO_MASK_BLENDOP = (
    "gz08eJxjYGBgYAFiCQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dlAx68oBEMbFxwX+"
    "AwGIBgCbGCeh"
)
BLENDOP_VERSION = 13

# Module defaults we read back from the reference sidecars (filled by dtstyle).
_OP_DEFAULTS: dict[str, dict] = {}


def decode_params(s: str) -> bytes:
    """Decode a darktable ``params``/``blendop_params`` attribute to bytes."""
    s = s.strip()
    if s.startswith("gz"):
        return zlib.decompress(base64.b64decode(s[4:]))
    if len(s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in s):
        return bytes.fromhex(s)
    return base64.b64decode(s)


def encode_params(b: bytes) -> str:
    """Encode raw struct bytes the way darktable does (gz+base64 if large)."""
    if len(b) <= 100:
        return b.hex()
    comp = zlib.compress(b)
    factor = min(len(b) // max(len(comp), 1) + 1, 99)
    return f"gz{factor:02d}{base64.b64encode(comp).decode()}"


def pack_floats(*vals: float) -> bytes:
    return struct.pack(f"<{len(vals)}f", *vals)


def unpack_floats(b: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(b) // 4}f", b)


# --- per-module param constructors ------------------------------------------

def exposure_params(ev: float, black: float = -0.000244, percentile: float = 50.0,
                    target: float = -4.0) -> bytes:
    """exposure v6: mode(0=manual), black, ev, deflicker percentile, target, bias."""
    return struct.pack("<6f", 0.0, black, ev, percentile, target, 1.0)


def crop_params(cx: float, cy: float, cw: float, ch: float) -> bytes:
    """crop v1: center x/y, width/height (normalized) + ratio 0/0."""
    return struct.pack("<6f", cx, cy, cw, ch, 0.0, 0.0)


def bilat_params(detail: float, midtone: float, sigma_r: float = 1.0,
                 sigma_s: float = 0.5, mode: float = 1.0) -> bytes:
    """bilat v3: mode, sigma_r (range), sigma_s (spatial), detail, midtone."""
    return struct.pack("<5f", mode, sigma_r, sigma_s, detail, midtone)


def sharpen_params(radius: float, amount: float, threshold: float) -> bytes:
    """sharpen v1: radius, amount, threshold."""
    return struct.pack("<3f", radius, amount, threshold)


def toneequal_params(lift: float, smoothing: float = 1.4142,
                     blending: float = 5.0, feathering: float = 1.0) -> bytes:
    """toneequal v1.  Lift is distributed across the shadow/midtone bands.

    order: noise, ultra_deep_blacks, deep_blacks, blacks, shadows, midtones,
    highlights, whites, speculars, blending, smoothing, feathering, quantization,
    contrast_boost, exposure_boost (15 floats) + details, method, iterations.
    """
    shadow = (lift, lift, lift, lift, lift * 0.7, lift * 0.4)
    t = (0.0, 0.0, 0.0, blending, smoothing, feathering, 0.0, 0.0, lift * 0.3)
    return struct.pack("<15f", *(shadow + t)) + struct.pack("<3i", 1, 2, 1)


# --- module container + XMP writer -----------------------------------------

@dataclass
class Module:
    operation: str
    modversion: int
    params: bytes
    enabled: bool = True
    multi_name: str = ""
    multi_priority: int = 0
    blendop_version: int = BLENDOP_VERSION
    blendop_params: bytes = field(default_factory=bytes)


def _li(num: int, m: Module) -> str:
    e = '1' if m.enabled else '0'
    mn = '1' if m.multi_name else '0'
    return f"""<rdf:li
     darktable:num="{num}"
     darktable:operation="{m.operation}"
     darktable:enabled="{e}"
     darktable:modversion="{m.modversion}"
     darktable:params="{encode_params(m.params)}"
     darktable:multi_name="{m.multi_name}"
     darktable:multi_name_hand_edited="0"
     darktable:multi_priority="{m.multi_priority}"
     darktable:blendop_version="{m.blendop_version}"
     darktable:blendop_params="{encode_params(m.blendop_params)}"/>"""


def write_sidecar(path: Path, modules: list[Module], masks: list[dict],
                  derived_from: str, history_end: int | None = None) -> None:
    """Write a darktable XMP sidecar for ``modules`` (+ optional masks)."""
    if history_end is None:
        history_end = len(modules)
    li = "\n".join(_li(i, m) for i, m in enumerate(modules))
    mask_li = "\n".join(
        f"""<rdf:li
     darktable:mask_num="{mk.get('mask_num', '')}"
     darktable:mask_id="{mk.get('mask_id', '')}"
     darktable:mask_type="{mk.get('mask_type', '')}"
     darktable:mask_name="{mk.get('mask_name', '')}"
     darktable:mask_version="{mk.get('mask_version', '6')}"
     darktable:mask_src="{mk.get('mask_src', '')}" />"""
        for mk in masks
    )
    masks_html = (
        f"<darktable:masks_history>\n    <rdf:Seq>\n    {mask_li}\n    </rdf:Seq>\n   "
        "</darktable:masks_history>"
        if masks else "<darktable:masks_history>\n    <rdf:Seq/>\n   </darktable:masks_history>"
    )
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:exif="http://ns.adobe.com/exif/1.0/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
    xmlns:darktable="http://darktable.sf.net/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
   xmpMM:DerivedFrom="{derived_from}"
   darktable:import_timestamp="0"
   darktable:xmp_version="5"
   darktable:raw_params="0"
   darktable:auto_presets_applied="1"
   darktable:history_end="{history_end}"
   darktable:iop_order_version="0"
   darktable:iop_order_list="{IOP_ORDER}"
   darktable:history_auto_hash="lifer"
   darktable:history_current_hash="lifer">
   {masks_html}
   <darktable:history>
    <rdf:Seq>
    {li}
    </rdf:Seq>
   </darktable:history>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
    path.write_text(text)
