"""Depth tiles (cloud-optimised GeoTIFFs of the solver depth) as a second, finer depth source for the county.

    ts = TileSet.discover(root, "county", settings)       # None when the folder / storms do not exist
    ts.label                                              # 'solver depth (tiles, 6.25 m; verification pending)'
    ts.sample_points("100yr", lng, lat) -> int32[n]       # depth cm at the pixel under each point, -1 = outside / no data
    ts.footprint_max("100yr", rings_lnglat) -> (cm, n_pixels)   # max over every pixel the footprint touches (all-touched)
    compare(index_cm, tile_cm) -> Comparison              # the two numbers, the difference and whether they disagree

Layout (written by the COG pipeline): <tiles dir>/<domain>/<storm>/max_depth.tif (+ depth_<sec>s.tif per hour, cog_meta.json).
`max_depth.tif` is EPSG:3857, uint16 centimetres, nodata 65535 (outside the model domain), cell 6.25 m (Mercator metres, about
5 m on the ground at 37 N).  The tile folder is configurable so finer native tiles can be dropped in later: the cell size, bounds
and nodata are read from each raster, nothing is hard-coded.  Folder order: settings["tiles_dir"], env BGV_TILES_DIR, else
<data root>/layers/cog.  Only the city has no tiles yet (it stays on the index / flow grid).

Sampling rule = the apps' rule: a building owns every cell its footprint touches (all-touched), its depth is the MAXIMUM over
those cells; flooded = depth >= 10 cm; tiers 10/30/60/120 cm.  Cells with nodata are ignored.  rasterio is used for the reads
only; the footprint rasterisation is shapely boxes (vectorised), cross-checked against rasterio.mask in tests / verification.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

R_MERC = 6378137.0
NODATA = 65535
FLOOD_MIN_CM = 10
DISAGREE_CM = 10
TIER_EDGES_CM = (10, 30, 60, 120)
INDEX_LABEL = "index (grid)"
MAX_NAME = "max_depth.tif"


def merc(lng, lat):
    lng = np.asarray(lng, np.float64)
    lat = np.asarray(lat, np.float64)
    return R_MERC * np.radians(lng), R_MERC * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))


def tier_of_cm(cm) -> int:
    return int(np.searchsorted(TIER_EDGES_CM, cm, side="right"))


def tiles_root(root, settings: dict | None = None) -> Path:
    p = (settings or {}).get("tiles_dir") or os.environ.get("BGV_TILES_DIR")
    return Path(p) if p else Path(root) / "layers" / "cog"


@dataclass
class Comparison:
    index_cm: int | None
    tile_cm: int | None
    tier_index: int
    tier_tile: int

    @property
    def delta_cm(self) -> int | None:
        return None if self.index_cm is None or self.tile_cm is None else int(self.tile_cm) - int(self.index_cm)

    @property
    def tier_differs(self) -> bool:
        return self.index_cm is not None and self.tile_cm is not None and self.tier_index != self.tier_tile

    @property
    def disagree(self) -> bool:
        """A material disagreement: the flooded state differs (10 cm threshold) or the depths differ by 10 cm or more.  A few cm
        that happen to straddle a tier edge are NOT flagged (finer cells give a few cm of difference everywhere)."""
        if self.index_cm is None or self.tile_cm is None:
            return False
        if (self.index_cm >= FLOOD_MIN_CM) != (self.tile_cm >= FLOOD_MIN_CM):
            return True
        return abs(int(self.tile_cm) - int(self.index_cm)) >= DISAGREE_CM


def compare(index_cm, tile_cm) -> Comparison:
    ti = None if index_cm is None else int(index_cm)
    tt = None if tile_cm is None or tile_cm < 0 else int(tile_cm)
    return Comparison(ti, tt, 0 if ti is None else tier_of_cm(ti), 0 if tt is None else tier_of_cm(tt))


class TileSet:
    def __init__(self, folder: Path, domain_key: str, storms: dict, verified: bool = False):
        self.folder = Path(folder)
        self.domain_key = domain_key
        self.paths: dict[str, Path] = storms
        self.verified = bool(verified)
        self._lock = threading.RLock()
        self._ds: dict[str, object] = {}
        self._info = None

    # ------------------------------------------------------------------------------------------ discovery
    @classmethod
    def discover(cls, root, domain_key: str, settings: dict | None = None, storms=None):
        """The tile set of a domain, or None.  Only the county has tiles; `storms` (optional) limits the keys looked for."""
        if domain_key != "county":
            return None
        base = tiles_root(root, settings) / domain_key
        if not base.is_dir():
            return None
        found = {}
        try:
            for d in sorted(base.iterdir()):
                p = d / MAX_NAME
                if d.is_dir() and p.is_file() and (storms is None or d.name in storms):
                    found[d.name] = p
        except OSError:
            return None
        if not found:
            return None
        ts = cls(base, domain_key, found, bool((settings or {}).get("tiles_verified", False)))
        try:
            ts.info()
        except Exception:                                                          # noqa: BLE001
            return None
        return ts

    @property
    def storms(self) -> list[str]:
        return list(self.paths)

    def has(self, storm: str) -> bool:
        return storm in self.paths

    def _open(self, storm: str):
        import rasterio
        with self._lock:
            ds = self._ds.get(storm)
            if ds is None:
                ds = rasterio.open(self.paths[storm])
                self._ds[storm] = ds
            return ds

    def close(self):
        with self._lock:
            for ds in self._ds.values():
                try:
                    ds.close()
                except Exception:                                                  # noqa: BLE001
                    pass
            self._ds.clear()

    def info(self) -> dict:
        """cell size (Mercator m), bounds, size of the first storm's raster (all storms share the grid)."""
        if self._info is None:
            ds = self._open(next(iter(self.paths)))
            t = ds.transform
            if ds.crs is None or ds.crs.to_epsg() != 3857:
                raise ValueError(f"depth tiles must be EPSG:3857, got {ds.crs}")
            self._info = {"res_x": float(t.a), "res_y": float(-t.e), "left": float(t.c), "top": float(t.f), "width": ds.width,
                          "height": ds.height, "nodata": ds.nodata if ds.nodata is not None else NODATA,
                          "dtype": str(ds.dtypes[0])}
        return self._info

    @property
    def res_m(self) -> float:
        return self.info()["res_x"]

    @property
    def label(self) -> str:
        r = self.res_m
        return f"solver depth (tiles, {r:g} m{'' if self.verified else '; verification pending'})"

    @property
    def note(self) -> str:
        return (f"Depth read from the solver depth tiles ({self.res_m:g} m cells, maximum over all hours of the storm); building = "
                "maximum over every cell its footprint touches, other points = the cell under the point.")

    # ------------------------------------------------------------------------------------------ sampling
    def _px(self, lng, lat):
        i = self.info()
        x, y = merc(lng, lat)
        return (x - i["left"]) / i["res_x"], (i["top"] - y) / i["res_y"]

    def sample_points(self, storm: str, lng, lat) -> np.ndarray:
        """Depth cm of the cell under every point (int32; -1 outside the raster or on no-data)."""
        from rasterio.windows import Window
        lng = np.atleast_1d(np.asarray(lng, np.float64))
        lat = np.atleast_1d(np.asarray(lat, np.float64))
        out = np.full(lng.shape, -1, np.int32)
        if storm not in self.paths:
            return out
        i = self.info()
        gx, gy = self._px(lng, lat)
        col = np.floor(gx).astype(np.int64)
        row = np.floor(gy).astype(np.int64)
        ok = (col >= 0) & (col < i["width"]) & (row >= 0) & (row < i["height"])
        if not ok.any():
            return out
        with self._lock:
            ds = self._open(storm)
            # group the points by 512 x 512 block so each block is decoded once
            idx = np.flatnonzero(ok)
            key = (row[idx] // 512) * 100000 + (col[idx] // 512)
            for k in np.unique(key):
                sel = idx[key == k]
                c0, c1 = int(col[sel].min()), int(col[sel].max())
                r0, r1 = int(row[sel].min()), int(row[sel].max())
                blk = ds.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
                v = blk[row[sel] - r0, col[sel] - c0].astype(np.int64)
                out[sel] = np.where(v == i["nodata"], -1, v)
        return out

    def neighbourhood_max(self, storm: str, lng, lat, radius_m: float = 30.0) -> np.ndarray:
        """Max depth cm over the cells within `radius_m` (ground metres, cell centres inside the circle) of each point (-1 none)."""
        from rasterio.windows import Window
        lng = np.atleast_1d(np.asarray(lng, np.float64))
        lat = np.atleast_1d(np.asarray(lat, np.float64))
        out = np.full(lng.shape, -1, np.int32)
        if storm not in self.paths:
            return out
        i = self.info()
        gx, gy = self._px(lng, lat)
        k = np.cos(np.radians(lat))
        rad = radius_m / np.maximum(k, 0.1) / i["res_x"]          # radius in cells (Mercator metres stretch by 1/cos lat)
        with self._lock:
            ds = self._open(storm)
            for j in range(lng.size):
                r = float(rad[j])
                c0, r0 = int(math.floor(gx[j] - r)), int(math.floor(gy[j] - r))
                c1, r1 = int(math.floor(gx[j] + r)), int(math.floor(gy[j] + r))
                c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, i["width"] - 1), min(r1, i["height"] - 1)
                if c1 < c0 or r1 < r0:
                    continue
                blk = ds.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)).astype(np.int64)
                cc, rr = np.meshgrid(np.arange(c0, c1 + 1) + 0.5, np.arange(r0, r1 + 1) + 0.5)
                inside = (cc - gx[j]) ** 2 + (rr - gy[j]) ** 2 <= r * r
                v = blk[inside & (blk != i["nodata"])]
                if v.size:
                    out[j] = int(v.max())
        return out

    def footprint_max(self, storm: str, rings, ) -> tuple[int, int]:
        """(max depth cm over the cells the polygon touches, number of valid cells); (-1, 0) when it touches none.
        `rings` = GeoJSON polygon coordinates [[ [lng, lat], ... ] (outer), hole, ...] or a list of such polygons."""
        import shapely
        from rasterio.windows import Window
        if storm not in self.paths:
            return -1, 0
        polys = rings if (rings and isinstance(rings[0][0][0], (list, tuple))) else [rings]
        i = self.info()
        geoms = []
        for pr in polys:
            ring_px = []
            for r in pr:
                a = np.asarray(r, np.float64)
                gx, gy = self._px(a[:, 0], a[:, 1])
                ring_px.append(np.stack([gx, gy], axis=1))
            try:
                g = shapely.Polygon(ring_px[0], ring_px[1:])
                if not g.is_valid:
                    g = g.buffer(0)
                geoms.append(g)
            except Exception:                                                      # noqa: BLE001
                continue
        if not geoms:
            return -1, 0
        g = shapely.union_all(geoms) if len(geoms) > 1 else geoms[0]
        if g.is_empty:
            return -1, 0
        x0, y0, x1, y1 = g.bounds
        c0 = max(int(math.floor(x0)), 0)
        c1 = min(int(math.floor(x1)), i["width"] - 1)
        r0 = max(int(math.floor(y0)), 0)
        r1 = min(int(math.floor(y1)), i["height"] - 1)
        if c1 < c0 or r1 < r0:
            return -1, 0
        with self._lock:
            blk = self._open(storm).read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)).astype(np.int64)
        rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        boxes = shapely.box(cc.ravel(), rr.ravel(), cc.ravel() + 1.0, rr.ravel() + 1.0)
        shapely.prepare(g)
        hit = shapely.intersects(g, boxes).reshape(blk.shape)
        v = blk[hit]
        v = v[v != i["nodata"]]
        if v.size == 0:
            return -1, 0
        return int(v.max()), int(v.size)


def facility_depths(ts: TileSet, lng, lat, storms=None) -> dict[str, np.ndarray]:
    """{storm: int32[n] depth cm under each point (-1 none)} for every storm of the tile set."""
    return {s: ts.sample_points(s, lng, lat) for s in (storms or ts.storms)}
