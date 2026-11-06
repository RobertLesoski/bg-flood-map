"""Utilities layer data (water + power from OpenStreetMap, flood exposure from the county COG tiles).

File: <data root>/layers/utilities.geojson.gz, built by scripts/build_utilities.py (format documented there and in docs/UTILITIES.md).
Pure numpy / json, no GPU.  `UtilitySet.for_app(app)` loads it once per data root (cached); `view(app)` projects it onto the
terrain frame of the current domain (city = features inside the city box, county = all).
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SOURCE_LABEL = "solver depth (tiles, 6.25 m; verification pending)"
FLOOD_CM = 10
R_MERC = 6378137.0

# group key -> (label, types, colour rgb 0..1 (Okabe-Ito, colour-blind safe), shape, search words)
GROUPS = {
    "water_tower": ("Water tower", ("water_tower",), (0.0, 0.447, 0.698), "triangle", ("water tower", "tower")),
    "water_tank": ("Water tank / reservoir", ("water_tank",), (0.337, 0.706, 0.914), "circle", ("water tank", "tank", "reservoir", "storage")),
    "water_treatment": ("Water works / treatment / pumping", ("water_works", "wastewater_plant", "pumping_station"),
                        (0.0, 0.62, 0.451), "square", ("water works", "water treatment", "treatment", "wastewater", "sewage", "pumping station", "pumping")),
    "dam": ("Dam / weir", ("dam", "weir"), (0.8, 0.475, 0.655), "diamond", ("dam", "weir", "lock")),
    "hydro": ("Hydroelectric plant", ("hydro_plant",), (0.94, 0.894, 0.259), "hexagon", ("hydro", "hydroelectric", "power plant", "power station")),
    "power_plant": ("Power plant / generator", ("power_plant", "generator"), (0.902, 0.624, 0.0), "pentagon", ("power plant", "power station", "generator", "solar", "plant")),
    "substation": ("Substation", ("substation",), (0.835, 0.369, 0.0), "cross", ("substation", "electric")),
    "transformer": ("Transformer", ("transformer",), (0.6, 0.6, 0.6), "dot", ("transformer",)),
}
LINE_GROUPS = {
    "power_lines": ("Power lines", ("power_line", "minor_line", "power_cable")),
    "pipelines": ("Pipelines", ("pipeline",)),
}
TYPE_LABEL = {"water_tower": "Water tower", "water_tank": "Water tank / reservoir", "water_works": "Water works",
              "wastewater_plant": "Wastewater plant", "pumping_station": "Pumping station", "dam": "Dam", "weir": "Weir",
              "hydro_plant": "Hydroelectric plant", "power_plant": "Power plant", "generator": "Generator (single unit)", "substation": "Substation",
              "transformer": "Transformer", "power_line": "Power line", "minor_line": "Minor power line",
              "power_cable": "Power cable", "pipeline": "Pipeline"}
GROUP_OF_TYPE = {t: g for g, v in GROUPS.items() for t in v[1]}
LINE_GROUP_OF_TYPE = {t: g for g, v in LINE_GROUPS.items() for t in v[1]}


@dataclass
class Feature:
    i: int
    type: str
    name: str
    operator: str
    osm_id: str
    source: str
    voltage: str
    lng: float
    lat: float
    in_city: bool
    d: list
    first: str | None
    f10: list | None = None
    f30: list | None = None
    coords: list | None = None          # LineString [(lng, lat)]

    @property
    def is_line(self) -> bool:
        return self.coords is not None

    def title(self) -> str:
        return self.name or TYPE_LABEL.get(self.type, self.type)


class UtilitySet:
    def __init__(self, doc: dict):
        meta = doc.get("metadata") or {}
        self.storms: list = list(meta.get("storms") or [])
        self.counts: dict = dict(meta.get("counts") or {})
        self.note: str = meta.get("note", "")
        self.features: list[Feature] = []
        for f in doc.get("features", []):
            p = f.get("properties") or {}
            g = f.get("geometry") or {}
            if p.get("type") not in TYPE_LABEL:
                continue
            line = g.get("type") == "LineString"
            coords = [(float(c[0]), float(c[1])) for c in g["coordinates"]] if line else None
            lng = float(p["lng"]) if "lng" in p else float(np.mean([c[0] for c in coords]))
            lat = float(p["lat"]) if "lat" in p else float(np.mean([c[1] for c in coords]))
            self.features.append(Feature(len(self.features), p["type"], p.get("name", ""), p.get("operator", ""), p.get("osm_id", ""),
                                         p.get("source", ""), p.get("voltage", ""), lng, lat, bool(p.get("in_city")),
                                         list(p.get("d") or []), p.get("first"), p.get("f10"), p.get("f30"), coords))

    @classmethod
    def load(cls, path) -> "UtilitySet | None":
        path = Path(path)
        if not path.is_file():
            return None
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rt", encoding="utf-8") as f:
            return cls(json.load(f))

    def storm_index(self, storm: str) -> int:
        try:
            return self.storms.index(storm)
        except ValueError:
            return -1

    def depth_cm(self, f: Feature, storm: str) -> int | None:
        k = self.storm_index(storm)
        return int(f.d[k]) if 0 <= k < len(f.d) else None

    def first_flooded(self, f: Feature) -> str | None:
        return f.first

    def search(self, text: str, limit: int = 4, near=None) -> list:
        """Name / type search: every word of `text` must be found in name + type label + group words.  Needs a letter in the
        query (a bare number is an address).  -> Feature list, name matches first, then the closest to `near` (lng, lat)."""
        words = [w for w in text.lower().replace(",", " ").split() if w]
        if not words or len(text.strip()) < 3 or not any(any(ch.isalpha() for ch in w) for w in words):
            return []
        scored = []
        for f in self.features:
            grp = GROUP_OF_TYPE.get(f.type)
            extra = " ".join(GROUPS[grp][4]) if grp else ""
            hay = f"{f.name} {TYPE_LABEL.get(f.type, '')} {f.operator} {extra}".lower()
            if not all(w in hay for w in words):
                continue
            named = all(w in f.name.lower() for w in words) if f.name else False
            dist = 0.0 if near is None else math.hypot((f.lng - near[0]) * 0.8, f.lat - near[1])
            scored.append((0 if named else 1, 1 if f.is_line else 0, dist, f.i))
        scored.sort()
        return [self.features[s[3]] for s in scored[:limit]]


_CACHE: dict = {}


def for_root(root) -> "UtilitySet | None":
    p = str(Path(root) / "layers" / "utilities.geojson.gz")
    if p not in _CACHE:
        try:
            _CACHE[p] = UtilitySet.load(p)
        except Exception:
            _CACHE[p] = None
    return _CACHE[p]


@dataclass
class DomainView:
    """The features of a domain projected into its world frame (rebuilt on a domain change only)."""
    key: str
    ids: np.ndarray                                  # point feature indices
    xy: np.ndarray                                   # [n, 2] world metres
    z: np.ndarray                                    # ground height (un-exaggerated)
    group: list                                      # group key per point
    line_ids: np.ndarray
    line_off: np.ndarray                             # vertex offsets into line_xyz (len n_lines + 1)
    line_xyz: np.ndarray                             # [m, 3] world x, y, ground z
    line_group: list
    line_bounds: np.ndarray                          # [n_lines, 4] xmin ymin xmax ymax


def _world(frame, lng, lat):
    lng, lat = np.asarray(lng, float), np.asarray(lat, float)
    x = R_MERC * np.radians(lng)
    y = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return frame.merc_to_world(x, y)


def build_view(uset: UtilitySet, domain_key: str, frame, sampler=None) -> DomainView:
    feats = [f for f in uset.features if domain_key != "city" or f.in_city]
    pts = [f for f in feats if not f.is_line]
    lines = [f for f in feats if f.is_line]
    X, Y = _world(frame, [f.lng for f in pts], [f.lat for f in pts]) if pts else (np.zeros(0), np.zeros(0))
    z = np.asarray(sampler.sample(X, Y), float) if (sampler is not None and len(pts)) else np.zeros(len(pts))
    off, xs, ys = [0], [], []
    for f in lines:
        a = np.asarray(f.coords, float)
        x, y = _world(frame, a[:, 0], a[:, 1])
        xs.append(x); ys.append(y)
        off.append(off[-1] + len(x))
    if lines:
        LX, LY = np.concatenate(xs), np.concatenate(ys)
        LZ = np.asarray(sampler.sample(LX, LY), float) if sampler is not None else np.zeros(len(LX))
        bounds = np.array([[x.min(), y.min(), x.max(), y.max()] for x, y in zip(xs, ys)])
    else:
        LX = LY = LZ = np.zeros(0)
        bounds = np.zeros((0, 4))
    return DomainView(domain_key, np.array([f.i for f in pts], np.int64), np.stack([X, Y], axis=1) if pts else np.zeros((0, 2)), z,
                      [GROUP_OF_TYPE[f.type] for f in pts], np.array([f.i for f in lines], np.int64), np.array(off, np.int64),
                      np.stack([LX, LY, LZ], axis=1), [LINE_GROUP_OF_TYPE.get(f.type, "power_lines") for f in lines], bounds)
