"""Shared helpers for the data-layer tests (no conftest on purpose, other test modules may bring their own)."""
from __future__ import annotations

import functools
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

DESKTOP = Path(__file__).resolve().parents[1]
if str(DESKTOP) not in sys.path:
    sys.path.insert(0, str(DESKTOP))

from bgv.geo import GridFrame, merc_to_lnglat  # noqa: E402

REAL_ROOT = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")


def real_root() -> Path:
    from bgv import config
    try:
        root = config.data_root()
    except FileNotFoundError:
        pytest.skip("flow data root not found")
    if not (root / "layers" / "buildings.geojson").is_file():
        pytest.skip("buildings.geojson missing")
    return root


@functools.lru_cache(maxsize=None)
def real_frame() -> GridFrame:
    root = real_root()
    return GridFrame.from_meta(json.loads((root / "layers" / "flow" / "25yr" / "meta.json").read_text()))


@functools.lru_cache(maxsize=None)
def real_buildings():
    from bgv.data.buildings import load_buildings
    return load_buildings(real_root(), real_frame())


@functools.lru_cache(maxsize=None)
def real_roads():
    from bgv.data.roads import load_roads
    return load_roads(real_root(), real_frame())


@functools.lru_cache(maxsize=4)
def real_hour(storm: str, hour: int) -> np.ndarray:
    f = real_frame()
    p = real_root() / "layers" / "flow" / storm / f"h{hour:02d}.rgba.gz"
    raw = gzip.open(p).read()
    assert len(raw) == f.width * f.height * 4
    return np.frombuffer(raw, dtype=np.uint8).reshape(f.height, f.width, 4)


# ----------------------------------------------------------------------------------------------- tiny synthetic grids
def tiny_frame(w: int = 20, h: int = 20, cell_m: float = 10.0, k: float = 0.8) -> GridFrame:
    xmin, ymax = -9633395.2, 4446630.4
    return GridFrame(w, h, cell_m, cell_m * k, xmin, ymax - h * cell_m, xmin + w * cell_m, ymax)


def px_to_lnglat(fr: GridFrame, col: float, row: float):
    """Grid pixel coordinates (col, row; pixel (c,r) is [c,c+1)x[r,r+1), row 0 north) -> (lng, lat)."""
    x = fr.xmin + col * fr.cell_m
    y = fr.ymax - row * fr.cell_m
    return merc_to_lnglat(x, y)


def ring(fr: GridFrame, pts, close=True):
    r = [list(px_to_lnglat(fr, c, rw)) for c, rw in pts]
    if close and r[0] != r[-1]:
        r.append(list(r[0]))
    return r


def polygon(fr: GridFrame, *rings_px, **props):
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring(fr, r) for r in rings_px]}}


def rect(c0, r0, c1, r1):
    return [(c0, r0), (c1, r0), (c1, r1), (c0, r1)]


def line(fr: GridFrame, pts_px, highway="residential"):
    return {"type": "Feature", "properties": {"highway": highway},
            "geometry": {"type": "MultiLineString", "coordinates": [[list(px_to_lnglat(fr, c, r)) for c, r in pts_px]]}}


def blank_grid(fr: GridFrame, valid=True) -> np.ndarray:
    g = np.zeros((fr.height, fr.width, 4), dtype=np.uint8)
    g[..., 1] = 128
    g[..., 2] = 128
    if valid:
        g[..., 3] = 255
    return g


def depth_to_code(d: float) -> int:
    return int(round(255 * math.sqrt(min(max(d / 8.0, 0.0), 1.0))))
