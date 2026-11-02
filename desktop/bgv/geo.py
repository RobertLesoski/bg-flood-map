"""Shared coordinate conventions for the BG Flood desktop viewer. EVERY module uses these, nobody re-derives them.

World frame (right-handed, metres, float32-safe):  +X east, +Y north, +Z up, origin = centre of the flow grid.
  X = (x_merc - x_c) * k,  Y = (y_merc - y_c) * k,  Z = elevation above sea level in metres (NOT exaggerated here)
  k = cell_ground_m / cell_m  (about 0.8: Web Mercator metres are stretched by 1/cos(lat))
Grid UV (what textures use): u = (col + 0.5) / W, v = (row + 0.5) / H, row 0 = NORTH.
  u = X / (W * cell_ground_m) + 0.5        v = 0.5 - Y / (H * cell_ground_m)
Texture texel centres are therefore exactly the flow-grid pixel centres, and a bilinear sampler matches fluid-data.js.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

R_MERC = 6378137.0


def lnglat_to_merc(lng, lat):
    x = R_MERC * math.radians(lng)
    y = R_MERC * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def merc_to_lnglat(x, y):
    lng = math.degrees(x / R_MERC)
    lat = math.degrees(2 * math.atan(math.exp(y / R_MERC)) - math.pi / 2)
    return lng, lat


@dataclass(frozen=True)
class GridFrame:
    width: int
    height: int
    cell_m: float            # mercator metres per cell
    cell_ground_m: float     # ground metres per cell
    xmin: float              # mercator bounds
    ymin: float
    xmax: float
    ymax: float

    @staticmethod
    def from_meta(meta: dict) -> "GridFrame":
        b = meta["bounds_merc"]
        return GridFrame(int(meta["width"]), int(meta["height"]), float(meta["cell_m"]), float(meta["cell_ground_m"]),
                         float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    @property
    def k(self) -> float:
        return self.cell_ground_m / self.cell_m

    @property
    def xc(self) -> float:
        return 0.5 * (self.xmin + self.xmax)

    @property
    def yc(self) -> float:
        return 0.5 * (self.ymin + self.ymax)

    @property
    def size_m(self):
        """Ground extent (X, Y) of the grid in metres."""
        return self.width * self.cell_ground_m, self.height * self.cell_ground_m

    def merc_to_world(self, x, y):
        return (x - self.xc) * self.k, (y - self.yc) * self.k

    def world_to_merc(self, X, Y):
        return X / self.k + self.xc, Y / self.k + self.yc

    def lnglat_to_world(self, lng, lat):
        x, y = lnglat_to_merc(lng, lat)
        return self.merc_to_world(x, y)

    def world_to_lnglat(self, X, Y):
        return merc_to_lnglat(*self.world_to_merc(X, Y))

    def world_to_uv(self, X, Y):
        sx, sy = self.size_m
        return X / sx + 0.5, 0.5 - Y / sy

    def world_to_pixel(self, X, Y):
        """Fractional pixel coordinates (col, row); pixel centres are at integer + 0.5."""
        u, v = self.world_to_uv(X, Y)
        return u * self.width, v * self.height
