"""LIFER_HOME/config.toml: the owner's per-machine lifer settings.

Watermark placement ([watermark]) and the handoff style sidecar ([develop])
live here so the repo stays generic and the owner tweaks the values without
touching the code. Values not present fall back to the agreed defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .home import lifer_home

# The agreed defaults; the owner overrides any of them in config.toml.
DEFAULTS: dict[str, dict[str, object]] = {
    "watermark": {
        "enabled": True,
        "file": "watermarkfull.png",   # the handwritten signature
        "monogram": "watermark.png",   # the "ET" mark for tight/small crops
        "width_frac": 0.10,            # mark width as a fraction of image width
        "min_width": 120,              # px, so the mark stays legible
        "opacity": 0.7,
        "corner": "bottom-right",
        "inset_frac": 0.02,            # inset as a fraction of the short edge
    },
    "develop": {
        "style_sidecar": "",           # path to the owner's edited .xmp
    },
    "gallery": {
        "roots": ["~/BIRDING", "~/Pictures"],   # scan darktable sidecars here
    },
}


def config_path() -> Path:
    return lifer_home() / "config.toml"


def load_config() -> dict[str, dict[str, object]]:
    """The built-in defaults, overlaid with LIFER_HOME/config.toml if present."""
    cfg: dict[str, dict[str, object]] = {key: dict(values) for key, values in DEFAULTS.items()}
    path = config_path()
    if path.is_file():
        with path.open("rb") as f:
            try:
                data = tomllib.load(f)
            except tomllib.TOMLDecodeError:
                data = {}
        for section, values in data.items():
            if isinstance(values, dict) and section in cfg:
                cfg[section].update(values)
    return cfg
