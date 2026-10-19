"""Extract a darktable develop style from edited sidecars and build a reusable template.

darktable stores each module's parameter struct in the sidecar XMP.  We decode
the modules that define the look (bilat local contrast, colorbalancergb
vibrance/saturation/chroma, colorzones, sharpen, and the scene-referred defaults
filmicrgb/channelmixerrgb) and report their typical values.  The same decoding
drives both the ``lifer dt-style`` summary and the template history the
``develop`` command patches per photo.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .dxmp import (Module, decode_params, unpack_floats)

# Modules that are per-photo and must never land in the shared template.
_PER_IMAGE = {"crop", "exposure", "toneequal"}
# Auto-applied baseline modules (scene-referred default), copied verbatim.
_BASELINE = ["rawprepare", "temperature", "demosaic", "highlights",
             "colorin", "colorout", "gamma", "flip"]
# Style modules that characterise the look.
_STYLE = ["bilat", "colorbalancergb", "colorzones", "sharpen"]

# Field slots we report per module, as (name, index-in-decoded-float-vector).
_FIELDS = {
    "bilat": [("detail", 3), ("midtone", 4)],
    "exposure": [("mode", 0), ("black", 1), ("ev", 2), ("percentile", 3), ("target", 4)],
    "sharpen": [("radius", 0), ("amount", 1), ("threshold", 2)],
    "filmicrgb": [("grey_point_source", 0), ("black_point_source", 1),
                  ("white_point_source", 2), ("contrast", 14), ("saturation", 15)],
    "colorbalancergb": [("chroma_global", 17), ("saturation_global", 19),
                        ("vibrance", 29), ("grey_fulcrum", 30)],
    "channelmixerrgb": [("r_row", 0), ("g_row", 5), ("b_row", 10)],
}


def _ns(dt: object) -> str:
    return "http://darktable.sf.net/"


def load_sidecar(path: Path) -> tuple[list[Module], list[dict], dict]:
    """Parse a darktable sidecar into modules, masks and description attrs."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
    dt = el = None
    for el in root.iter(f"{rdf}Description"):
        if el.get(f"{{{_ns(dt)}}}history_end") is not None:
            dt = el
            break
    history = []
    h = dt.find(f"{{{_ns(dt)}}}history")
    for container in ("Bag", "Seq", "Alt"):
        seq = h.find(f"{rdf}{container}")
        if seq is not None:
            for li in seq.findall(f"{rdf}li"):
                attrs = {k.split("}")[-1]: v for k, v in li.attrib.items()}
                history.append(Module(
                    operation=attrs.get("operation", ""),
                    modversion=int(attrs.get("modversion", "1")),
                    enabled=attrs.get("enabled", "1") == "1",
                    params=decode_params(attrs.get("params", "")),
                    multi_name=attrs.get("multi_name", ""),
                    multi_priority=int(attrs.get("multi_priority", "0")),
                    blendop_version=int(attrs.get("blendop_version", "13")),
                    blendop_params=decode_params(attrs.get("blendop_params", "")),
                ))
            break
    masks = []
    mh = dt.find(f"{{{_ns(dt)}}}masks_history")
    if mh is not None:
        for container in ("Bag", "Seq", "Alt"):
            seq = mh.find(f"{rdf}{container}")
            if seq is not None:
                for li in seq.findall(f"{rdf}li"):
                    masks.append({k.split("}")[-1]: v for k, v in li.attrib.items()})
                break
    meta = {k.split("}")[-1]: v for k, v in dt.attrib.items()}
    return history, masks, meta


def _module(history: list[Module], op: str, wanted: str = "") -> Module | None:
    """First module with ``op`` and (optionally) a specific multi_name."""
    for m in history:
        if m.operation == op and (not wanted or m.multi_name == wanted):
            return m
    return None


def build_template(base: Path) -> list[Module]:
    """The base develop history: baseline + scene defaults + style.

    ``base`` is the sidecar of one edited photo that best represents the
    style; per-image modules (crop/exposure/toneequal) are deliberately
    excluded and injected by ``develop``.
    """
    modules, _masks, _meta = load_sidecar(base)
    picked: list[Module] = []
    for op in _BASELINE:
        m = _module(modules, op)
        if m is not None:
            picked.append(m)
    for op in ("channelmixerrgb", "filmicrgb"):
        m = _module(modules, op, "scene-referred default")
        if m is None:
            m = _module(modules, op)
        if m is not None:
            picked.append(m)
    for op in _STYLE:
        m = _module(modules, op)
        if m is not None:
            picked.append(m)
    # Style modules keep darktable's own blendop (already on each Module).
    return picked


def _typical(histories: list[list[Module]]) -> dict:
    """Per-module typical (median) values across the edited sidecars."""
    out: dict = {}
    for op in _STYLE + ["filmicrgb", "channelmixerrgb"]:
        allvals: list[tuple] = []
        for h in histories:
            for m in h:
                if m.operation == op:
                    allvals.append(unpack_floats(m.params))
                    break
        if not allvals:
            continue
        n = min(len(v) for v in allvals)
        med = [statistics.median(v[i] for v in allvals) for i in range(n)]
        fields = _FIELDS.get(op)
        if op == "colorzones":
            # an unrolled per-channel curve preset; enumerating 130 slots is
            # noise, so just record that a consistent preset is applied
            out[op] = {"modversion": next(m.modversion for h in histories
                                         for m in h if m.operation == op),
                       "count": len(allvals),
                       "note": "per-channel curve preset (consistent across frames)"}
            continue
        if fields is None:
            fields = [(f"f{i}", i) for i in range(n)]
        out[op] = {
            "modversion": next(m.modversion for h in histories
                               for m in h if m.operation == op),
            "count": len(allvals),
            "fields": [name for name, _ in fields],
            "median": [med[idx] for _, idx in fields],
        }
    return out


def extract_style(sidecar_paths: list[Path]) -> dict:
    """Summarise an editing style from edited darktable sidecars."""
    histories = [load_sidecar(p)[0] for p in sidecar_paths]
    style = _typical(histories)
    style["_comments"] = {
        "bilat": "local contrast (detail/midtone)",
        "colorbalancergb": "color balance rgb (global chroma/sat boost)",
        "sharpen": "output sharpening",
        "filmicrgb": "scene-referred filmic tone curve",
        "channelmixerrgb": "identity RGB matrix (D65)",
    }
    return style


def _fmt_val(v: float) -> str:
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


def style_summary(style: dict) -> str:
    """Human-readable summary printed by ``lifer dt-style``."""
    lines = ["develop style (decoded from edited darktable sidecars)"]
    for op, info in style.items():
        if op.startswith("_"):
            continue
        lines.append(f"  {op} (v{info['modversion']}, n={info['count']})")
        for name, val in zip(info.get("fields", []), info.get("median", [])):
            lines.append(f"      {name:22} = {_fmt_val(val)}")
        if "note" in info:
            lines.append(f"      {info['note']}")
    return "\n".join(lines)


def save_style(style: dict, home: Path) -> Path:
    """Write style.json under LIFER_HOME/style/."""
    d = home / "style"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "style.json"
    p.write_text(json.dumps(style, indent=2) + "\n")
    return p


def _edited_sidecars(folder: Path) -> list[Path]:
    """Sidecars with an edited history (not just auto-applied defaults)."""
    found: list[Path] = []
    for x in sorted(folder.glob("*.xmp")):
        # edited histories are larger than the plain auto-import sidecar
        if x.stat().st_size < 4000:
            continue
        try:
            modules, _masks, _m = load_sidecar(x)
        except Exception:
            continue
        if any(m.operation in ("bilat", "colorbalancergb", "colorzones")
               for m in modules):
            found.append(x)
    return found


def run_dt_style(sidecars: str | None = None, source: str | None = None) -> None:
    """CLI entry for `lifer dt-style`: summarise a darktable develop style."""
    from .home import lifer_home
    if sidecars:
        paths = [Path(p).expanduser() for p in sidecars.split(",") if p.strip()]
    elif source:
        paths = _edited_sidecars(Path(source).expanduser())
    else:
        raise SystemExit("error: give --sidecars or --source "
                         "(edited .xmp sidecars to analyse)")
    if not paths:
        raise SystemExit("error: no edited sidecars found")
    print(f"analysing {len(paths)} edited sidecar(s)...")
    style = extract_style(paths)
    out = save_style(style, lifer_home())
    print(style_summary(style))
    print(f"\nwrote {out}")
