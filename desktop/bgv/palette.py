"""Tier colour palettes shared by the interface, the building layer and the road layer.

    "std"  the desktop viewer's standard tier colours (blue / teal / orange / red, same as the web viewer's "Standard tiers")
    "cvd"  colour-blind safe: the web viewer's default, a cividis-derived ramp (yellow -> grey -> dark blue), distinguishable with
           deuteranopia, protanopia and tritanopia; lightness falls monotonically with depth, so it also survives greyscale

`App.palette` holds the key (persisted in the settings); the layers and `ui.theme.set_palette` follow it.
"""
from __future__ import annotations

import numpy as np

PALETTES = {
    "std": ("#4dabf7", "#00c9c9", "#ff8c00", "#e74c3c"),
    "cvd": ("#f0da46", "#a09875", "#575d6d", "#00224e"),
}
LABELS = {"std": "Standard tiers", "cvd": "Colour-blind safe (cividis)"}
DEFAULT = "std"


def hexes(key: str):
    return PALETTES.get(key, PALETTES[DEFAULT])


def srgb(key: str):
    return [tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)) for h in hexes(key)]


def linear(key: str):
    out = []
    for c in srgb(key):
        c = np.asarray(c, np.float64)
        out.append(tuple(float(v) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)))
    return out
