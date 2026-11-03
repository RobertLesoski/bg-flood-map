"""Terrain heights on the flow-grid frame - pure CPU, no wgpu.

`load_terrain(root, frame)` warps `dem_bg.tif` (EPSG:32616, elevations in FEET, nodata -9999, 5.088 m cells) onto the
flow grid (EPSG:3857, row 0 north, same pixel grid as the flow textures), bilinear, and returns float32[H, W] METRES
(feet x 0.3048), NaN where the DEM has no coverage.  Result is cached in `<desktop>/cache` keyed by the DEM's
size + mtime and the grid geometry.  Nothing is ever written into the data root.

Also here (CPU helpers for the renderer): `fill_nan`, `terrain_stats`, `build_normal_map`, `normal_mip_chain`.

Domains: every function takes an optional `source` (`TerrainSource(path, units)`).  The default is the city DEM above and
behaves exactly as before.  The county source is any GeoTIFF (UTM 16N, any resolution, metres - `dem_warren_5m.tif` is
10404 x 12043 cells of 5 m, 125 M cells, 500 MB as float32): it is warped lazily by GDAL straight from the file, with AREA
AVERAGING onto the flow grid (about 20 m on the ground), so the whole raster is never held in RAM (peak a few hundred MB)
and the 5 / 10 m relief is averaged instead of aliased.  The terrain textures use the flow-grid pixel grid, so the county
terrain is shown at the flow resolution; see `desktop/DESIGN.md` ("Domains").
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config
from ..geo import GridFrame

FEET_TO_M = 0.3048
DEM_NODATA = -9999.0
DEM_NAME = "dem_bg.tif"
CACHE_VERSION = 1
LIKELY_FEET_MEDIAN_M = 330.0     # Warren County ground is 110 - 330 m: a "metres" DEM whose median is above that is in feet


@dataclass(frozen=True)
class TerrainSource:
    """Where a domain's terrain comes from."""
    path: Path
    units: str = "feet"                 # "feet" (city dem_bg.tif) | "metres"
    lazy: bool = False                  # True: GDAL warps straight from the file with area averaging (big county DEMs)
    label: str = ""

    def __post_init__(self):
        if self.units not in ("feet", "metres"):
            raise ValueError(f"terrain units must be 'feet' or 'metres', got {self.units!r}")


def city_source(root) -> TerrainSource:
    return TerrainSource(Path(root) / DEM_NAME, "feet", False, "city DEM (feet)")


def dem_to_metres(raw: np.ndarray, nodata: float | None = DEM_NODATA) -> np.ndarray:
    """Raw DEM values (feet) -> float32 metres; NaN for nodata (-9999 / file nodata) and non-finite values."""
    raw = np.asarray(raw)
    bad = ~np.isfinite(raw) | (raw == DEM_NODATA)
    if nodata is not None:
        bad |= raw == nodata
    z = (raw.astype(np.float64) * FEET_TO_M).astype(np.float32)
    z[bad] = np.nan
    return z


def _cache_key(dem: Path, frame: GridFrame, source: "TerrainSource | None" = None) -> str:
    st = dem.stat()
    s = f"{CACHE_VERSION}|{st.st_size}|{int(st.st_mtime)}|{frame.width}x{frame.height}|{frame.cell_m!r}|" \
        f"{frame.xmin!r},{frame.ymin!r},{frame.xmax!r},{frame.ymax!r}"
    if source is not None and (source.units != "feet" or source.lazy):          # the city key stays what it always was
        s += f"|{source.units}|{'lazy' if source.lazy else 'full'}|{dem.name}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _source(root, source: "TerrainSource | None") -> "TerrainSource":
    return source if source is not None else city_source(root)


def terrain_cache_path(root, frame: GridFrame, source: "TerrainSource | None" = None) -> Path:
    src = _source(root, source)
    return config.cache_dir() / f"terrain_{_cache_key(src.path, frame, src)}.npy"


def warp_dem(dem_path, frame: GridFrame) -> np.ndarray:
    """Warp the feet DEM to the frame grid -> float32 metres, NaN outside coverage."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    with rasterio.open(dem_path) as ds:
        raw = ds.read(1)
        src_transform, src_crs, nodata = ds.transform, ds.crs, ds.nodata
    z = dem_to_metres(raw, nodata)
    src = np.where(np.isnan(z), np.float32(DEM_NODATA), z).astype(np.float32)   # metres, sentinel nodata
    dst = np.full((frame.height, frame.width), DEM_NODATA, np.float32)
    dst_transform = Affine(frame.cell_m, 0.0, frame.xmin, 0.0, -frame.cell_m, frame.ymax)
    reproject(src, dst, src_transform=src_transform, src_crs=src_crs, src_nodata=DEM_NODATA,
              dst_transform=dst_transform, dst_crs=CRS.from_epsg(3857), dst_nodata=DEM_NODATA,
              resampling=Resampling.bilinear, num_threads=4)
    dst[dst == DEM_NODATA] = np.nan
    return dst


def warp_dem_lazy(dem_path, frame: GridFrame, units: str = "metres", progress=None) -> tuple[np.ndarray, str]:
    """Warp a (possibly huge) DEM onto the flow grid with GDAL, reading windows of the file as needed and averaging the
    source cells under every output cell (area average: a 5 m raster on a 20 m grid).  Returns (float32[H, W] metres with
    NaN outside coverage, note) - `note` is '' or a warning (a 'metres' file that is evidently in feet is converted)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    if progress is not None:
        progress("terrain: warping the DEM onto the flow grid", 0.05)
    with rasterio.open(dem_path) as ds:
        if ds.crs is None:
            raise ValueError(f"{dem_path}: the DEM has no coordinate reference system")
        nodata = ds.nodata
        src_res = abs(ds.transform.a)
        dst = np.full((frame.height, frame.width), DEM_NODATA, np.float32)
        dst_transform = Affine(frame.cell_m, 0.0, frame.xmin, 0.0, -frame.cell_m, frame.ymax)
        # average only when the output cell is clearly coarser than the source cell (a same-resolution DEM keeps the
        # bilinear look of the city path)
        ratio = frame.cell_ground_m / max(src_res if ds.crs.is_projected else frame.cell_ground_m, 1e-6)
        method = Resampling.average if ratio >= 1.5 else Resampling.bilinear
        reproject(rasterio.band(ds, 1), dst, src_nodata=nodata if nodata is not None else DEM_NODATA,
                  dst_transform=dst_transform, dst_crs=CRS.from_epsg(3857), dst_nodata=DEM_NODATA,
                  resampling=method, num_threads=4, warp_mem_limit=256)
    bad = ~np.isfinite(dst) | (dst <= -9000.0) | (dst > 9000.0)
    dst[bad] = np.nan
    note = ""
    if units == "feet":
        dst *= np.float32(FEET_TO_M)
    else:
        fin = dst[np.isfinite(dst)]
        if fin.size and float(np.median(fin)) > LIKELY_FEET_MEDIAN_M:
            dst *= np.float32(FEET_TO_M)
            note = (f"{Path(dem_path).name}: elevations look like FEET (median {float(np.median(fin)):.0f}, Warren County is "
                    f"110 - 330 m): converted to metres")
    if progress is not None:
        progress("terrain: DEM warped", 0.8)
    return dst, note


def load_terrain(root, frame: GridFrame, use_cache: bool = True, source: "TerrainSource | None" = None,
                 progress=None) -> np.ndarray:
    """float32[H, W] ground elevation in metres on the flow grid (row 0 = north), NaN = no DEM coverage."""
    src = _source(root, source)
    dem = src.path
    if not dem.is_file():
        raise FileNotFoundError(f"terrain not found: {dem}")
    cache = terrain_cache_path(root, frame, src)
    if use_cache and cache.is_file():
        try:
            a = np.load(cache)
            if a.shape == (frame.height, frame.width) and a.dtype == np.float32:
                return a
        except Exception:
            pass
    if src.lazy:
        z, note = warp_dem_lazy(dem, frame, src.units, progress)
        if note:
            _write_note(cache, note)
    else:
        z = warp_dem(dem, frame)
    if use_cache:
        _atomic_save(cache, z)
    return z


def _write_note(cache: Path, note: str):
    try:
        cache.with_suffix(".note.txt").write_text(note, encoding="utf-8")
    except OSError:
        pass


def read_note(cache: Path) -> str:
    try:
        return cache.with_suffix(".note.txt").read_text(encoding="utf-8")
    except OSError:
        return ""


def _atomic_save(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass


def terrain_stats(h: np.ndarray) -> dict:
    """min / max / mean (finite cells only), NaN count and fraction."""
    fin = np.isfinite(h)
    n = int(fin.sum())
    if n == 0:
        return dict(min=float("nan"), max=float("nan"), mean=float("nan"), nan_count=int(h.size), nan_fraction=1.0,
                    shape=tuple(h.shape))
    v = h[fin]
    return dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean(dtype=np.float64)),
                nan_count=int(h.size - n), nan_fraction=float(1 - n / h.size), shape=tuple(h.shape))


def fill_nan(h: np.ndarray):
    """Replace NaN by the nearest finite value.  Returns (filled float32, valid bool mask).  All-NaN -> zeros."""
    valid = np.isfinite(h)
    if valid.all():
        return h.astype(np.float32, copy=False), valid
    if not valid.any():
        return np.zeros_like(h, dtype=np.float32), valid
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = h[idx[0], idx[1]].astype(np.float32)
    return filled, valid


def build_normal_map(filled: np.ndarray, valid: np.ndarray, cell_ground_m: float) -> np.ndarray:
    """float16 RGBA 'slope map' (texture rgba16float): R = dz/dEast, G = dz/dNorth (unit-less slope, central
    differences over the grid, so +R = ground rises toward +col, +G = rises toward smaller rows), B = curvature
    (laplacian, 1/m: >0 in hollows, <0 on crests), A = 1 on DEM coverage / 0 where the height was NaN-filled.
    Slopes (not unit normals) are stored because they average correctly in the mip chain and keep precision on
    the very flat ground of this area; the shader rebuilds n = normalize(-exag*g, 1) for any exaggeration."""
    z = filled.astype(np.float32)
    H, W = z.shape
    zp = np.pad(z, 1, mode="edge")
    gx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2.0 * cell_ground_m)
    gy = -(zp[2:, 1:-1] - zp[:-2, 1:-1]) / (2.0 * cell_ground_m)          # rows grow southwards
    lap = (zp[1:-1, 2:] + zp[1:-1, :-2] + zp[2:, 1:-1] + zp[:-2, 1:-1] - 4.0 * z) / (cell_ground_m ** 2)
    out = np.empty((H, W, 4), np.float16)
    out[..., 0] = gx
    out[..., 1] = gy
    out[..., 2] = lap
    out[..., 3] = valid
    return out


def normal_mip_chain(nm: np.ndarray, max_levels: int = 16) -> list[np.ndarray]:
    """Mip chain of the slope map: 2x2 box average of every channel (float32 math, stored float16)."""
    levels = [nm]
    cur = nm
    while len(levels) < max_levels and (cur.shape[0] > 1 or cur.shape[1] > 1):
        h, w = cur.shape[:2]
        h2, w2 = max(h // 2, 1), max(w // 2, 1)
        a = cur[: h2 * 2 if h > 1 else 1, : w2 * 2 if w > 1 else 1].astype(np.float32)
        if h > 1 and w > 1:
            a = a.reshape(h2, 2, w2, 2, 4).mean(axis=(1, 3))
        elif h > 1:
            a = a.reshape(h2, 2, w2, 4).mean(axis=1)
        else:
            a = a.reshape(h2, w2, 2, 4).mean(axis=2)
        cur = a.astype(np.float16)
        levels.append(cur)
    return levels


def _mip_shapes(h: int, w: int, max_levels: int = 16):
    shapes = [(h, w)]
    while len(shapes) < max_levels and (h > 1 or w > 1):
        h, w = max(h // 2, 1), max(w // 2, 1)
        shapes.append((h, w))
    return shapes


def load_terrain_gpu_data(root, frame: GridFrame, source: "TerrainSource | None" = None, progress=None):
    """Everything the renderer uploads, from cache when possible.

    Returns (height_filled float32[H,W], normal_mips list[uint8[h,w,4]] (level 0 first), stats dict, valid bool[H,W]).
    Cached files (desktop/cache): filled height, flat concatenated normal mip chain, validity mask is derived from
    the normal map alpha.  `source`: a TerrainSource (default: the city DEM); stats["note"] carries a warning text
    when the DEM needed a unit correction."""
    src = _source(root, source)
    if not src.path.is_file():
        raise FileNotFoundError(f"terrain not found: {src.path}")
    key = _cache_key(src.path, frame, src)
    p_h = config.cache_dir() / f"terrain_filled_{key}.npy"
    p_n = config.cache_dir() / f"terrain_slopes_{key}.npy"
    p_s = config.cache_dir() / f"terrain_stats_{key}.json"
    H, W = frame.height, frame.width
    shapes = _mip_shapes(H, W)
    total = sum(h * w * 4 for h, w in shapes)   # float16 elements
    try:
        filled = np.load(p_h)
        flat = np.load(p_n)
        import json as _json
        stats = _json.loads(p_s.read_text())
        if filled.shape == (H, W) and flat.size == total:
            mips, off = [], 0
            for h, w in shapes:
                n = h * w * 4
                mips.append(flat[off:off + n].reshape(h, w, 4))
                off += n
            return filled, mips, stats, mips[0][..., 3] > 0.5
    except Exception:
        pass
    z = load_terrain(root, frame, source=src, progress=progress)
    stats = terrain_stats(z)
    note = read_note(terrain_cache_path(root, frame, src))
    if note:
        stats["note"] = note
    if progress is not None:
        progress("terrain: slope map", 0.85)
    filled, valid = fill_nan(z)
    nm = build_normal_map(filled, valid, frame.cell_ground_m)
    mips = normal_mip_chain(nm)
    flat = np.concatenate([m.reshape(-1) for m in mips])
    _atomic_save(p_h, filled)
    _atomic_save(p_n, flat)
    try:
        import json as _json
        p_s.write_text(_json.dumps(stats))
    except Exception:
        pass
    return filled, mips, stats, valid
