"""Critical facilities (hospitals, schools, fire, police, shelters, clinics, town halls ...) with flood depths.

    fs = FacilitySet.load(path)                  # layers/infrastructure.geojson | layers/county/infrastructure.geojson
    fs.attach_tiles(tileset)                     # county: point samples of the depth tiles (second source)
    fs.depth_cm("100yr") -> int32[n]             # EFFECTIVE depth: tiles when present for that storm, else the index value
    fs.source("100yr") -> label                  # 'solver depth (tiles, 6.25 m; verification pending)' | 'index (grid)'
    fs.info(i, storm) -> dict                    # everything the click card shows (both sources, difference, first storm ...)

The file's `d_<storm>` (cm) is the flow-grid maximum over all hours at the facility (a node = its cell, a way = max over its
outline); `flooded` means >= 10 cm.  Tiles: the cell under the facility point.  First flooding storm: the smallest DESIGN storm
(2yr .. 100yr) whose effective depth is >= 10 cm; if no design storm floods it, the first event storm that does.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import tiles as tl

FLOOD_MIN_CM = 10
NEAR_M = 30.0            # radius of the extra "deepest water within ... of the point" figure shown next to the tile depth

# type -> (label, colour (r, g, b) 0..1, shape).  Colours are Okabe-Ito-like hues chosen to differ from the blue / teal / orange /
# red depth tiers; the SHAPE is the second cue so colour is never the only one (colour-blind safe).
TYPE_STYLE = {
    "hospital": ("Hospital", (0.95, 0.35, 0.66), "cross"),
    "clinic": ("Clinic", (0.99, 0.62, 0.80), "cross_small"),
    "fire_station": ("Fire station", (1.00, 0.30, 0.20), "triangle"),
    "police": ("Police", (0.55, 0.62, 1.00), "square"),
    "school": ("School", (0.97, 0.90, 0.26), "circle"),
    "shelter": ("Shelter", (0.20, 0.90, 0.62), "diamond"),
    "assembly_point": ("Assembly point", (0.20, 0.90, 0.62), "diamond"),
    "community_centre": ("Community centre", (0.55, 0.95, 0.80), "circle_small"),
    "townhall": ("Town hall", (0.90, 0.90, 0.92), "square_small"),
    "ems": ("EMS", (1.00, 0.55, 0.30), "triangle_small"),
}
DEFAULT_STYLE = ("Other facility", (0.80, 0.80, 0.80), "circle")


def style(t: str):
    return TYPE_STYLE.get(t, DEFAULT_STYLE)


def facilities_path(root, domain_key: str) -> Path:
    root = Path(root)
    return root / "layers" / ("county" if domain_key == "county" else "") / "infrastructure.geojson"


def storm_order_key(s: str):
    m = re.fullmatch(r"(\d+)yr", s)
    if m:
        return (0, int(m.group(1)), s)
    m = re.fullmatch(r"ev(\d+)", s)
    if m:
        return (1, int(m.group(1)), s)
    return (2, 0, s)


class FacilityError(Exception):
    """Facilities cannot be shown; str(e) is the message for the user."""


@dataclass
class Facility:
    name: str
    type: str
    lng: float
    lat: float
    source: str
    osm_id: str


class FacilitySet:
    def __init__(self, feats: list[Facility], storms: list[str], d_index: dict[str, np.ndarray], meta: dict):
        self.items = feats
        self.n = len(feats)
        self.storms = sorted(storms, key=storm_order_key)
        self.d_index = d_index
        self.meta = meta
        self.lng = np.array([f.lng for f in feats], np.float64)
        self.lat = np.array([f.lat for f in feats], np.float64)
        self.types = np.array([f.type for f in feats])
        self.tiles: tl.TileSet | None = None
        self.d_tiles: dict[str, np.ndarray] = {}
        self.d_near: dict[str, np.ndarray] = {}
        self._eff: dict[str, np.ndarray] = {}
        self.tier_sources: set = set()          # storms whose flow grid is a depth-TIER preview (severe = ">= 120 cm", not a depth)

    def is_approx(self, storm: str) -> bool:
        """The shown depth of this storm comes from a depth-tier preview grid (no tile replaced it)."""
        return storm in self.tier_sources and storm not in self.d_tiles

    @classmethod
    def load(cls, path) -> "FacilitySet":
        p = Path(path)
        if not p.is_file():
            raise FacilityError(f"Facilities file not found: {p} (build it with scripts/build_infrastructure.py)")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            feats, rows = [], []
            storms = list((doc.get("metadata") or {}).get("storms") or [])
            for f in doc["features"]:
                g = f.get("geometry") or {}
                if g.get("type") != "Point":
                    continue
                pr = f.get("properties") or {}
                feats.append(Facility(str(pr.get("name") or ""), str(pr.get("type") or ""), float(g["coordinates"][0]),
                                      float(g["coordinates"][1]), str(pr.get("source") or ""), str(pr.get("osm_id") or "")))
                rows.append(pr)
        except Exception as e:                                                     # noqa: BLE001
            raise FacilityError(f"Facilities file {p} cannot be read: {type(e).__name__}: {e}") from None
        if not storms and rows:
            storms = sorted({k[2:] for r in rows for k in r if k.startswith("d_")}, key=storm_order_key)
        d = {s: np.array([int(r.get("d_" + s, 0) or 0) for r in rows], np.int32) for s in storms}
        return cls(feats, storms, d, doc.get("metadata") or {})

    # ------------------------------------------------------------------------------------------ sources
    def attach_tiles(self, ts: "tl.TileSet | None", storms=None):
        """Sample the tiles at every facility point (98 points x 12 storms: about a second)."""
        self.tiles = ts
        self.d_tiles = {}
        self.d_near: dict[str, np.ndarray] = {}
        self._eff = {}
        if ts is None or self.n == 0:
            return
        for s in storms or self.storms:
            if ts.has(s):
                self.d_tiles[s] = ts.sample_points(s, self.lng, self.lat)
                self.d_near[s] = ts.neighbourhood_max(s, self.lng, self.lat, NEAR_M)

    def has_tiles(self, storm: str) -> bool:
        return storm in self.d_tiles

    def depth_cm(self, storm: str) -> np.ndarray:
        """Effective depth (cm) of every facility: tiles where available (-1 = outside the tile domain falls back to the index)."""
        c = self._eff.get(storm)
        if c is None:
            base = self.d_index.get(storm, np.zeros(self.n, np.int32))
            t = self.d_tiles.get(storm)
            c = base if t is None else np.where(t >= 0, t, base)
            self._eff[storm] = c
        return c

    def source(self, storm: str) -> str:
        if self.has_tiles(storm) and self.tiles is not None:
            return self.tiles.label
        return tl.INDEX_LABEL

    def flooded(self, storm: str) -> np.ndarray:
        return self.depth_cm(storm) >= FLOOD_MIN_CM

    def first_flooded(self, i: int) -> str | None:
        for s in self.storms:
            if self.depth_cm(s)[i] >= FLOOD_MIN_CM:
                return s
        return None

    def max_cm(self, i: int) -> tuple[int, str | None]:
        best, bs = 0, None
        for s in self.storms:
            v = int(self.depth_cm(s)[i])
            if v > best:
                best, bs = v, s
        return best, bs

    def info(self, i: int, storm: str) -> dict:
        f = self.items[i]
        label, rgb, shape = style(f.type)
        per = []
        for s in self.storms:
            idx = int(self.d_index.get(s, np.zeros(self.n, np.int32))[i])
            t = self.d_tiles.get(s)
            tv = None if t is None or t[i] < 0 else int(t[i])
            eff = tv if tv is not None else idx
            nr = self.d_near.get(s)
            per.append({"storm": s, "index_cm": idx, "tile_cm": tv, "effective_cm": eff,
                        "near_cm": None if nr is None or nr[i] < 0 else int(nr[i]),
                        "comparison": tl.compare(idx, tv) if tv is not None else None,
                        "source": self.tiles.label if tv is not None else tl.INDEX_LABEL})
        mx, ms = self.max_cm(i)
        cur = next((p for p in per if p["storm"] == storm), None)
        return {"name": f.name or "(unnamed)", "type": f.type, "type_label": label, "lng": f.lng, "lat": f.lat,
                "source": f.source, "osm_id": f.osm_id, "first_flooded": self.first_flooded(i), "max_cm": mx, "max_storm": ms,
                "current": cur, "per_storm": per}

    def nearest(self, lng: float, lat: float, max_m: float) -> int:
        if self.n == 0:
            return -1
        k = np.cos(np.radians(lat))
        d = np.hypot((self.lng - lng) * 111320.0 * k, (self.lat - lat) * 110540.0)
        i = int(np.argmin(d))
        return i if d[i] <= max_m else -1

    def counts(self, storm: str) -> tuple[int, int]:
        return int(self.flooded(storm).sum()), self.n
