"""Building exposure filter and export: filter the buildings of the current domain by depth tier, land use, sale value and storm,
and write the filtered list to CSV or GeoJSON (points at the footprint centroid).

Depths are the per-building footprint maxima (cm) the caller passes in (native solver tiles via TileService.building_depths, or
the flow grid).  Tiers follow the viewer: dry < 10 cm, shallow 10-30, moderate 30-60, deep 60-120, severe >= 120 cm.
Land use and sale price come from the parcel join (city); the county building file carries neither, so those filters only
narrow the list where the data exists ('unknown' is its own land-use value and a value filter excludes unknown prices only when
the user asks for it).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIER_EDGES_CM = (10, 30, 60, 120)
TIER_NAMES = ("dry", "shallow", "moderate", "deep", "severe")
UNKNOWN = "unknown"


def tier_of_cm(cm: np.ndarray) -> np.ndarray:
    """int8[n]: -1 no data, 0 dry, 1 shallow .. 4 severe."""
    cm = np.asarray(cm)
    t = np.searchsorted(np.asarray(TIER_EDGES_CM), cm, side="right").astype(np.int8)
    return np.where(cm < 0, -1, t).astype(np.int8)


@dataclass
class ExposureFilter:
    tiers: tuple = (False, True, True, True, True)      # dry, shallow, moderate, deep, severe
    land_uses: tuple = ()                               # empty = all; else the values to keep (UNKNOWN for '')
    min_value: float = 0.0                              # sale price in $ (0 = no lower bound)
    max_value: float = 0.0                              # 0 = no upper bound
    require_value: bool = False                         # drop buildings with no known price

    def key(self):
        return (tuple(bool(x) for x in self.tiers), tuple(self.land_uses), float(self.min_value), float(self.max_value),
                bool(self.require_value))


def land_use_values(land_use: np.ndarray) -> list[tuple[str, int]]:
    lu = np.where(np.asarray(land_use, dtype=str) == "", UNKNOWN, np.asarray(land_use, dtype=str))
    vals, cnt = np.unique(lu, return_counts=True)
    order = np.argsort(-cnt)
    return [(str(vals[i]), int(cnt[i])) for i in order]


def apply(depth_cm: np.ndarray, land_use: np.ndarray, sale_price: np.ndarray, f: ExposureFilter) -> np.ndarray:
    """int64 ids of the buildings that pass, deepest first."""
    depth_cm = np.asarray(depth_cm)
    t = tier_of_cm(depth_cm)
    keep = np.zeros(depth_cm.shape, bool)
    for k, on in enumerate(f.tiers):
        if on:
            keep |= t == k
    if f.land_uses:
        lu = np.where(np.asarray(land_use, dtype=str) == "", UNKNOWN, np.asarray(land_use, dtype=str))
        keep &= np.isin(lu, np.asarray(list(f.land_uses), dtype=str))
    price = np.asarray(sale_price, np.float64)
    known = np.isfinite(price) & (price > 0)
    if f.require_value:
        keep &= known
    if f.min_value > 0:
        keep &= known & (price >= f.min_value)
    if f.max_value > 0:
        keep &= known & (price <= f.max_value)                   # a value range only keeps buildings with a known value
    ids = np.flatnonzero(keep)
    return ids[np.argsort(-depth_cm[ids], kind="stable")]


def summarize(ids, depth_cm, sale_price) -> dict:
    ids = np.asarray(ids, np.int64)
    d = np.asarray(depth_cm)[ids]
    p = np.asarray(sale_price, np.float64)[ids]
    known = np.isfinite(p) & (p > 0)
    return {"count": int(ids.size), "by_tier": {TIER_NAMES[k]: int((tier_of_cm(d) == k).sum()) for k in range(5)},
            "value_known": int(known.sum()), "value_sum": float(p[known].sum()), "max_cm": int(d.max()) if d.size else 0}


FIELDS = ("building", "address", "land_use", "sale_price_usd", "depth_cm", "depth_ft", "tier", "lng", "lat")


def rows(ids, b, depth_cm, frame) -> list[dict]:
    """One dict per building (frame: GridFrame for world -> lng/lat; b: the Buildings arrays)."""
    out = []
    for i in np.asarray(ids, np.int64):
        lng, lat = frame.world_to_lnglat(float(b.cx[i]), float(b.cy[i]))
        cm = int(depth_cm[i])
        pr = float(b.sale_price[i])
        out.append({"building": int(i), "address": str(b.address[i]), "land_use": str(b.land_use[i]) or UNKNOWN,
                    "sale_price_usd": (round(pr) if np.isfinite(pr) and pr > 0 else ""), "depth_cm": cm,
                    "depth_ft": round(cm / 30.48, 2), "tier": TIER_NAMES[int(tier_of_cm(np.array([cm]))[0])] if cm >= 0 else "no data",
                    "lng": round(float(lng), 7), "lat": round(float(lat), 7)})
    return out


def write_csv(path, rws, meta: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        for k, v in meta.items():                                   # provenance header lines (# key: value)
            fh.write(f"# {k}: {v}\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rws)
    return p


def write_geojson(path, rws, meta: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lng"], r["lat"]]},
              "properties": {k: v for k, v in r.items() if k not in ("lng", "lat")}} for r in rws]
    p.write_text(json.dumps({"type": "FeatureCollection", "metadata": meta, "features": feats}), encoding="utf-8")
    return p


def read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))
