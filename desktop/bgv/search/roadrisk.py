"""Roads at risk: the flooded-road lists of layers/roads_flooded/<domain>/<storm>.json with sorting / filtering, and the way
geometry for "zoom to the road".

    t = RoadTable.load(root, "county", "100yr")      # raises RoadsError (user-facing message) when the file is missing / bad
    ids = t.view(sort="flooded_m", ascending=False, min_flooded_pct=10, evac_only=False, text="russellville")   # row indices
    t.name(i)  t.class_name(i)  t.evac_text(i)
    geo = RoadGeometry.load(root, domain_key, cache_scene_roads=None)   # way id -> [[lng, lat], ...] (lazy, worker thread)
    geo.bbox(way_id) / geo.coords(way_id)

Row columns (build_infrastructure.py): id (OSM way), name, hw (index into `classes`), evac (0 none, 1 tagged in OSM, 2 major class),
len_m, wet_pct (% of the way's length with >= 2 cm), flooded_pct (% >= 10 cm), max_cm.  ALL values are the maximum over the
storm's hours (flow-grid maximum), NOT the state at the hour shown in the viewer.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVAC_TEXT = {0: "", 1: "evacuation route (OSM tag)", 2: "major road (treated as evacuation corridor)"}
SORT_KEYS = ("flooded_pct", "max_cm", "len_m", "flooded_m", "evac", "name")


class RoadsError(Exception):
    """Road list unavailable; str(e) is shown to the user."""


def roads_path(root, domain_key: str, storm: str) -> Path:
    return Path(root) / "layers" / "roads_flooded" / domain_key / f"{storm}.json"


class RoadTable:
    def __init__(self, doc: dict, storm: str):
        self.storm = storm
        self.doc = doc
        self.classes = list(doc.get("classes") or [])
        rows = doc.get("rows") or []
        self.n = len(rows)
        self.id = np.array([r[0] for r in rows], np.int64)
        self._name = [str(r[1] or "") for r in rows]
        self.hw = np.array([r[2] for r in rows], np.int16)
        self.evac = np.array([r[3] for r in rows], np.int8)
        self.len_m = np.array([r[4] for r in rows], np.float64)
        self.wet_pct = np.array([r[5] for r in rows], np.float64)
        self.flooded_pct = np.array([r[6] for r in rows], np.float64)
        self.max_cm = np.array([r[7] for r in rows], np.int32)
        self.flooded_m = self.len_m * self.flooded_pct / 100.0
        self._ltext = np.array([(n + " " + self.class_name(i)).lower() for i, n in enumerate(self._name)]) if self.n else np.zeros(0, "U1")
        self.total_km = float(doc.get("total_km", 0.0))
        self.flooded_km = float(doc.get("flooded_km", 0.0))
        self.n_roads = int(doc.get("n_roads", 0))
        self.source = doc.get("source", "")
        self.label = doc.get("label", storm)
        self.rules = doc.get("rules", "")
        self.built = doc.get("built", "")

    @classmethod
    def load(cls, root, domain_key: str, storm: str) -> "RoadTable":
        p = roads_path(root, domain_key, storm)
        if not p.is_file():
            raise RoadsError(f"No flooded-road list for the {storm} storm: {p} is missing (scripts/build_infrastructure.py writes it).")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if doc.get("format") != "bgroads":
                raise ValueError(f"format {doc.get('format')!r}")
            return cls(doc, storm)
        except Exception as e:                                                     # noqa: BLE001
            raise RoadsError(f"The flooded-road list {p} cannot be read: {type(e).__name__}: {e}") from None

    def name(self, i: int) -> str:
        n = self._name[i]
        return n if n else f"(unnamed {self.class_name(i)})"

    def class_name(self, i: int) -> str:
        h = int(self.hw[i])
        return self.classes[h] if 0 <= h < len(self.classes) else "other"

    def evac_text(self, i: int) -> str:
        return EVAC_TEXT.get(int(self.evac[i]), "")

    def view(self, sort: str = "flooded_pct", ascending: bool = False, min_flooded_pct: float = 0.0, evac_only: bool = False,
             text: str = "", min_len_m: float = 0.0) -> np.ndarray:
        """Row indices after filtering, ordered by `sort` (stable; ties broken by max depth then way id)."""
        keep = np.ones(self.n, bool)
        if min_flooded_pct > 0:
            keep &= self.flooded_pct >= min_flooded_pct
        if min_len_m > 0:
            keep &= self.len_m >= min_len_m
        if evac_only:
            keep &= self.evac > 0
        words = text.lower().split()
        if words and self.n:
            for w in words:
                keep &= np.char.find(self._ltext, w) >= 0
        idx = np.flatnonzero(keep)
        if sort not in SORT_KEYS:
            sort = "flooded_pct"
        if sort == "name":
            key = np.array([self._name[i].lower() or "￿" for i in idx])
        else:
            key = getattr(self, sort).astype(np.float64)[idx]
            if not ascending:
                key = -key
        # ties: deeper first, then way id (stable, reproducible)
        if sort == "name":
            order = np.lexsort((self.id[idx], key))
            if not ascending:
                order = order[::-1]
        else:
            order = np.lexsort((self.id[idx], -self.max_cm[idx], key))
        return idx[order]


@dataclass
class _Way:
    coords: list


class RoadGeometry:
    """Way id -> polyline (lng, lat).  Source: <root>/layers/county/roads.geojson.gz (OSM ways with `osm_id`, covers the city too)
    or, failing that, a GeoJSON file given as `fallback` (the cached OSM scene of the county)."""

    def __init__(self, ways: dict, source: str):
        self.ways = ways
        self.source = source

    @classmethod
    def load(cls, root, fallback=None) -> "RoadGeometry":
        cands = [Path(root) / "layers" / "county" / "roads.geojson.gz"]
        if fallback:
            cands.append(Path(fallback))
        last = None
        for p in cands:
            if not p.is_file():
                continue
            try:
                opener = gzip.open if p.suffix == ".gz" else open
                with opener(p, "rt", encoding="utf-8") as f:
                    doc = json.load(f)
                ways = {}
                for ft in doc["features"]:
                    oid = (ft.get("properties") or {}).get("osm_id")
                    g = ft.get("geometry") or {}
                    if oid is None:
                        continue
                    c = g.get("coordinates")
                    if g.get("type") == "LineString":
                        ways[int(oid)] = c
                    elif g.get("type") == "MultiLineString":
                        ways[int(oid)] = [pt for part in c for pt in part]
                return cls(ways, str(p))
            except Exception as e:                                                 # noqa: BLE001
                last = e
        raise RoadsError("Road geometry not available (zoom to road is disabled): "
                         + (f"{type(last).__name__}: {last}" if last else f"{cands[0]} is missing"))

    def coords(self, way_id: int):
        return self.ways.get(int(way_id))

    def bbox(self, way_id: int):
        c = self.coords(way_id)
        if not c:
            return None
        a = np.asarray(c, np.float64)
        return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())
