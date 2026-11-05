"""Elevation cross-section along a drawn line with the water surface of the chosen storm / hour.

    ground = make_ground(app)                          # the loaded DEM at full resolution when the file is there, else the terrain grid
    p = build_profile(world_xy, frame, depth_source, "100yr", 12.0, ground)
    p.dist_m, p.ground_m, p.depth_m, p.surface_m       arrays along the line (surface = ground + depth where wet, NaN where dry)
    p.max_depth_m, p.wet_length_m, p.length_m, p.vertex_dist_m   summary numbers

Distances along the line are true ellipsoidal ground distances (geodesy.cumulative_m of the dense sample points).  Sample spacing
is half the depth raster's cell (about 3 m on the 6.25 m tiles), at most `max_points` samples (then L / (n - 1) is larger).
Ground comes from the DEM the terrain was built from (5 m for the county, read through rastersrc with the block cache) so a levee
or a channel narrower than the 20 m terrain grid still shows; the fallback is the terrain grid the 3D view uses, `ground_label`
says which one was used.  Depth comes from the depth adapter (tiles or flow grid, `source_label`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geodesy
from .rastersrc import BlockRaster


class DemGround:
    """Ground elevation (metres) from the DEM file of the domain's terrain source, bilinear, NaN where it has no data."""

    kind = "dem"

    def __init__(self, path, units: str = "metres", label: str = ""):
        self.path = str(path)
        self.scale = 0.3048 if units == "feet" else 1.0
        self.raster = BlockRaster(path, scale=self.scale, nodata=-9999.0)
        self.label = label or f"ground: DEM {self.resolution_m():g} m"

    def resolution_m(self) -> float:
        i = self.raster.info()
        return round(float(i["resx"]), 3)

    def sample_lnglat(self, lng, lat) -> np.ndarray:
        crs = self.raster.crs
        if crs is not None and crs.to_epsg() == 3857:
            x, y = geodesy.lnglat_to_merc(lng, lat)
        else:
            from rasterio.warp import transform
            xs, ys = transform("EPSG:4326", crs, np.asarray(lng, np.float64).tolist(), np.asarray(lat, np.float64).tolist())
            x, y = np.asarray(xs), np.asarray(ys)
        return self.raster.sample(x, y)


class GridGround:
    """Ground from the terrain grid of the 3D view (flow-grid resolution); NaN where the DEM had no coverage."""

    kind = "grid"

    def __init__(self, terrain, frame):
        self.t = terrain
        self.frame = frame
        cg = frame.cell_ground_m
        self.label = f"ground: terrain grid {cg:.0f} m"

    def sample_lnglat(self, lng, lat) -> np.ndarray:
        X, Y = geodesy.lnglat_to_world(self.frame, lng, lat)
        z = np.asarray(self.t.sampler.sample(X, Y), np.float64)
        valid = getattr(self.t, "valid", None)
        if valid is not None:
            c, r = self.frame.world_to_pixel(X, Y)
            ci = np.clip(np.floor(c).astype(np.int64), 0, valid.shape[1] - 1)
            ri = np.clip(np.floor(r).astype(np.int64), 0, valid.shape[0] - 1)
            z = np.where(valid[ri, ci], z, np.nan)
        return z


def make_ground(app):
    """The best ground source for the running app (full-resolution DEM, else the terrain grid)."""
    try:
        src = app.domain.terrain
        p = getattr(src, "path", None)
        if p is not None and p.is_file():
            return DemGround(p, getattr(src, "units", "metres"), f"ground: DEM {p.name}")
    except Exception:
        pass
    return GridGround(app.terrain, app.terrain.frame)


@dataclass
class Profile:
    storm: str
    hour: float
    dist_m: np.ndarray
    x: np.ndarray
    y: np.ndarray
    lng: np.ndarray
    lat: np.ndarray
    ground_m: np.ndarray
    depth_m: np.ndarray
    vertex_dist_m: np.ndarray
    source_label: str = ""
    ground_label: str = ""
    step_m: float = 0.0

    @property
    def surface_m(self) -> np.ndarray:
        s = self.ground_m + self.depth_m
        return np.where(self.depth_m > 0, s, np.nan)

    @property
    def length_m(self) -> float:
        return float(self.dist_m[-1]) if self.dist_m.size else 0.0

    @property
    def max_depth_m(self) -> float:
        return float(self.depth_m.max()) if self.depth_m.size else 0.0

    @property
    def wet_length_m(self) -> float:
        """Length of the line under water (sample spacing x wet samples)."""
        return float((self.depth_m > 0).sum()) * self.step_m

    @property
    def n(self) -> int:
        return int(self.dist_m.size)

    def at(self, d: float) -> dict:
        """Readout at distance d (m) along the line (linear between samples)."""
        d = min(max(float(d), 0.0), self.length_m)
        i = int(np.clip(np.searchsorted(self.dist_m, d) - 1, 0, max(self.n - 2, 0)))
        j = min(i + 1, self.n - 1)
        span = self.dist_m[j] - self.dist_m[i]
        f = 0.0 if span <= 0 else (d - self.dist_m[i]) / span

        def lerp(a):
            a0, a1 = a[i], a[j]
            if np.isnan(a0) or np.isnan(a1):
                return float(a0 if f < 0.5 else a1)
            return float(a0 + (a1 - a0) * f)
        g, dp = lerp(self.ground_m), lerp(self.depth_m)
        return {"dist_m": d, "ground_m": g, "depth_m": dp, "surface_m": (g + dp) if dp > 0 and not math.isnan(g) else float("nan"),
                "x": lerp(self.x), "y": lerp(self.y)}


def resample_polyline(xy: np.ndarray, n: int):
    """n points evenly spaced (in world metres) along the polyline xy (k, 2).  Returns (points (n, 2), distance of each vertex
    along the polyline in world metres)."""
    xy = np.asarray(xy, np.float64)
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    s = np.linspace(0.0, total, n)
    # snap the sample nearest to every interior vertex onto it so corners are not cut
    x = np.interp(s, cum, xy[:, 0])
    y = np.interp(s, cum, xy[:, 1])
    return np.column_stack([x, y]), cum


def build_profile(world_xy, frame, source, storm: str, hour: float, ground, max_points: int = 2400,
                  max_step_m: float | None = None) -> Profile:
    xy = np.asarray(world_xy, np.float64).reshape(-1, 2)
    if xy.shape[0] < 2:
        raise ValueError("a profile needs at least two points")
    lng_v, lat_v = geodesy.world_to_lnglat(frame, xy[:, 0], xy[:, 1])
    length = geodesy.path_length_m(lng_v, lat_v)
    res = max(float(source.resolution_m(storm)) or frame.cell_ground_m, 0.5)
    target_step = res * 0.5
    n = int(min(max_points, max(2, math.ceil(length / target_step) + 1)))
    if max_step_m:
        n = max(n, int(math.ceil(length / max_step_m)) + 1)
    pts, cum_w = resample_polyline(xy, n)
    lng, lat = geodesy.world_to_lnglat(frame, pts[:, 0], pts[:, 1])
    dist = geodesy.cumulative_m(lng, lat)
    # distance of each user vertex along the dense path (their nearest dense sample)
    tot_w = cum_w[-1] if cum_w[-1] > 0 else 1.0
    vidx = np.clip(np.rint(cum_w / tot_w * (n - 1)).astype(int), 0, n - 1)
    gz = np.asarray(ground.sample_lnglat(lng, lat), np.float64)
    dp = np.asarray(source.depth_many(lng, lat, storm, float(hour)), np.float64)
    step = float(dist[-1] / (n - 1)) if n > 1 else 0.0
    return Profile(storm, float(hour), dist, pts[:, 0], pts[:, 1], lng, lat, gz, dp, dist[vidx],
                   source.label(storm), getattr(ground, "label", ""), step)
