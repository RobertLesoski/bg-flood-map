"""Per-address assessment: the index values of every storm, the depth-tile values (county) and how they compare.

    fp = Footprints.load(buildings_geojson)               # county: OSM way id -> polygon (cached scene file with osm_id)
    a = assess(idx, i, tileset, fp)                       # Assessment for address record i
    a.rows[k]  -> StormAssess (index StormRow, tile_cm, comparison, shown depth/tier/source)
    a.method   -> 'footprint' | 'point' | None            how the tile numbers were taken
    a.notes    -> honest remarks (what the numbers are, why they might differ)

The HEADLINE value of a storm is the tile value when tiles exist for that storm (domain with tiles), else the index value; both are
always kept so the UI / report can show the two numbers and their difference.  Sources are labelled
'solver depth (tiles, 6.25 m; verification pending)' and 'index (grid)'.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import tiles as tl
from .addrindex import AddressIndex, StormRow, tier_of_cm, TIER_NAMES


class Footprints:
    """OSM way id -> polygon coordinates (list of rings, lng/lat) from a GeoJSON file whose features carry `osm_id`."""

    def __init__(self, polys: dict, source: str):
        self.polys = polys
        self.source = source

    @classmethod
    def load(cls, path) -> "Footprints":
        p = Path(path)
        doc = json.loads(p.read_text(encoding="utf-8"))
        polys = {}
        for ft in doc["features"]:
            oid = (ft.get("properties") or {}).get("osm_id")
            g = ft.get("geometry") or {}
            if oid is None or g.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            polys[int(oid)] = g["coordinates"]
        return cls(polys, str(p))

    def get(self, way_id: int):
        return self.polys.get(int(way_id))


@dataclass
class StormAssess:
    key: str
    label: str
    index: StormRow
    tile_cm: int | None = None          # None: no tiles for this storm / outside the tile domain
    tile_cells: int = 0
    comparison: tl.Comparison | None = None
    shown_cm: int = 0                   # the headline depth (tiles when present)
    shown_tier: int = 0
    shown_source: str = tl.INDEX_LABEL
    source_kind: str = ""               # storm source of the index grid: 'rasters' | 'tiers'

    @property
    def short(self) -> str:
        return short_label(self.label)

    @property
    def flooded(self) -> bool:
        return self.shown_cm >= tl.FLOOD_MIN_CM


@dataclass
class Assessment:
    rec: object
    rows: list = field(default_factory=list)
    method: str | None = None
    tile_label: str | None = None
    notes: list = field(default_factory=list)

    def row(self, key: str):
        return next((r for r in self.rows if r.key == key), None)


def assess(idx: AddressIndex, i: int, tileset: "tl.TileSet | None" = None, footprints: "Footprints | None" = None) -> Assessment:
    rec = idx.record(i)
    rows = [StormAssess(sr.key, sr.label, sr, source_kind=sr.source) for sr in idx.storm_rows(i)]
    out = Assessment(rec, rows)
    for r in rows:
        r.shown_cm, r.shown_tier = r.index.depth_cm, r.index.tier
        r.shown_source = tl.INDEX_LABEL
    poly = None
    if tileset is not None:
        if rec.kind == 0 and rec.bid >= 0 and footprints is not None:
            poly = footprints.get(rec.bid)
        out.method = "footprint" if poly is not None else "point"
        out.tile_label = tileset.label
        for r in rows:
            if not tileset.has(r.key):
                continue
            if poly is not None:
                cm, cells = tileset.footprint_max(r.key, poly)
            else:
                v = int(tileset.sample_points(r.key, [rec.lng], [rec.lat])[0])
                cm, cells = v, (1 if v >= 0 else 0)
            if cm < 0:
                continue                          # outside the tiles / no data: stays on the index value
            r.tile_cm, r.tile_cells = cm, cells
            r.comparison = tl.compare(r.index.depth_cm, cm)
            r.shown_cm, r.shown_tier, r.shown_source = cm, tier_of_cm(cm), tileset.label
        if out.method == "footprint":
            out.notes.append("Tiles: maximum over every %g m cell the building footprint touches." % tileset.res_m)
            if rec.nb > 1:
                out.notes.append(f"This address record merges {rec.nb} footprints; the tiles use the main one, the index all of them.")
        else:
            why = "a lot without a building footprint" if rec.kind == 1 else "no footprint available for this building"
            out.notes.append(f"Tiles: the cell under the address point ({why}).")
    out.notes.append("Index: maximum over the flow-grid cells the footprint touches (%s)." % _grid_text(idx, rec.lat))
    return out


def _grid_text(idx: AddressIndex, lat: float = 37.0) -> str:
    g = (idx.storms[0].get("grid") if idx.storms else None) or {}
    cm = g.get("cell_m")
    if not cm:
        return "flow grid"
    return f"{cm:g} m Mercator cells, about {cm * math.cos(math.radians(lat)):.0f} m on the ground"


def short_label(label: str) -> str:
    """'100-year storm (county)' -> '100-year storm'; 'Historical event 6 (county)' -> 'Event 6'."""
    s = label.replace(" (county)", "").replace(" (city)", "")
    return s.replace("Historical event", "Event")


def tier_name(t: int) -> str:
    return TIER_NAMES[t] if 0 <= t < len(TIER_NAMES) else "?"
