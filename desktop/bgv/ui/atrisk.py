"""At-risk building list: filtering / searching / CSV export (pure CPU, no imgui).

    f = AtRiskFilter(tiers=(True, True, True, True), search="main st")
    ids = filter_ids(stats, buildings, f)        # building ids, deepest first (stats.order is already deepest first)
    export_csv(path, ids, buildings, stats, storm="25yr", hour=12, source="tiers", frame=None)
    AddressIndex(buildings)                       # lower-cased address array, built once per buildings object

`stats` is a bgv.data.stats.BuildingStats (depth_m, tier, speed_ms, order ...), `buildings` a bgv.data.buildings.Buildings.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .units import M_TO_FT, TIER_NAMES, depth_short, tier_range_text


@dataclass
class AtRiskFilter:
    tiers: tuple = (True, True, True, True)      # shallow, moderate, deep, severe
    search: str = ""
    min_depth_m: float = 0.0

    def key(self) -> tuple:
        return (tuple(bool(t) for t in self.tiers), self.search.strip().lower(), round(float(self.min_depth_m), 3))


class AddressIndex:
    """Lower-cased searchable text per building (address + land use), built lazily once."""

    def __init__(self, buildings):
        self.n = int(buildings.n)
        addr = np.asarray(buildings.address)
        land = np.asarray(buildings.land_use)
        self.addr = np.char.lower(addr.astype(str)) if addr.size else np.zeros(0, "<U1")
        self.land = np.char.lower(land.astype(str)) if land.size else np.zeros(0, "<U1")

    def match(self, ids: np.ndarray, query: str) -> np.ndarray:
        """bool mask over `ids`: every whitespace separated word of `query` occurs in the address or the land use."""
        mask = np.ones(len(ids), bool)
        for word in query.lower().split():
            hit = (np.char.find(self.addr[ids], word) >= 0) | (np.char.find(self.land[ids], word) >= 0)
            mask &= hit
        return mask


def filter_ids(stats, buildings, flt: AtRiskFilter, index: AddressIndex | None = None, limit: int | None = None) -> np.ndarray:
    """Flooded building ids, deepest first, restricted by tier / min depth / search text."""
    if stats is None:
        return np.zeros(0, np.int32)
    ids = np.asarray(stats.order, np.int64)
    if ids.size == 0:
        return ids.astype(np.int32)
    tier = np.asarray(stats.tier)[ids]
    keep = np.zeros(ids.size, bool)
    for k in range(4):
        if k < len(flt.tiers) and flt.tiers[k]:
            keep |= tier == k + 1
    if flt.min_depth_m > 0:
        keep &= np.asarray(stats.depth_m)[ids] >= flt.min_depth_m
    ids = ids[keep]
    q = flt.search.strip()
    if q and ids.size and buildings is not None:
        if index is None:
            index = AddressIndex(buildings)
        ids = ids[index.match(ids, q)]
    if limit is not None:
        ids = ids[:limit]
    return ids.astype(np.int32)


@dataclass
class Row:
    rank: int
    id: int
    address: str
    depth_m: float
    tier: int
    land_use: str
    price: float
    height_m: float
    speed_ms: float
    area_m2: float


def row_for(i: int, rank: int, stats, b) -> Row:
    price = float(b.sale_price[i]) if len(b.sale_price) > i else float("nan")
    addr = str(b.address[i]).strip() or f"Building #{int(i)}"
    return Row(rank, int(i), addr, float(stats.depth_m[i]), int(stats.tier[i]), str(b.land_use[i]).strip(), price,
               float(b.height_m[i]), float(stats.speed_ms[i]), float(b.area_m2[i]))


CSV_COLUMNS = ["rank", "building_id", "address", "depth_m", "depth_ft", "depth_note", "tier", "tier_range",
               "speed_ms", "land_use", "sale_price_usd", "height_m", "footprint_m2", "lng", "lat", "storm", "hour",
               "depth_source"]


def csv_rows(ids, buildings, stats, *, storm: str, hour: int, source: str, frame=None):
    """Yield the CSV data rows (lists) for `ids` (in the given order)."""
    approx = source == "tiers"
    for rank, i in enumerate(np.asarray(ids, np.int64), 1):
        i = int(i)
        d = float(stats.depth_m[i])
        t = int(stats.tier[i])
        price = float(buildings.sale_price[i])
        lng = lat = ""
        if frame is not None and math.isfinite(float(buildings.cx[i])):
            lo, la = frame.world_to_lnglat(float(buildings.cx[i]), float(buildings.cy[i]))
            lng, lat = f"{lo:.6f}", f"{la:.6f}"
        note = depth_short(d, tier=t, approx=approx, ge=">=")
        yield [rank, i, str(buildings.address[i]).strip(), f"{d:.3f}", f"{d * M_TO_FT:.2f}", note, TIER_NAMES[t - 1] if 1 <= t <= 4 else "",
               tier_range_text(t), f"{float(stats.speed_ms[i]):.2f}", str(buildings.land_use[i]).strip(),
               "" if not math.isfinite(price) or price <= 0 else f"{price:.0f}", f"{float(buildings.height_m[i]):.1f}",
               f"{float(buildings.area_m2[i]):.0f}", lng, lat, storm, hour,
               "depth tiers (approximate)" if approx else "solver depth rasters"]


def export_csv(path, ids, buildings, stats, *, storm: str, hour: int, source: str, frame=None) -> int:
    """Write the CSV (UTF-8 with BOM so Excel opens it correctly).  Returns the number of data rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for row in csv_rows(ids, buildings, stats, storm=storm, hour=hour, source=source, frame=frame):
            w.writerow(row)
            n += 1
    os.replace(tmp, path)
    return n


def default_export_dir() -> Path:
    """Documents\\BG_Flood_Exports (created on demand by export_csv)."""
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    docs = home / "Documents"
    return (docs if docs.is_dir() else home) / "BG_Flood_Exports"


def default_export_name(storm: str, hour: int) -> str:
    import time
    return f"at_risk_{storm}_h{int(hour):02d}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
