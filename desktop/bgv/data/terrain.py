"""Terrain heights on the flow-grid frame - pure CPU, no wgpu.

`load_terrain(root, frame)` warps `dem_bg.tif` (EPSG:32616, elevations in FEET, nodata -9999, 5.088 m cells) onto the
flow grid (EPSG:3857, row 0 north, same pixel grid as the flow textures), bilinear, and returns float32[H, W] METRES
(feet x 0.3048), NaN where the DEM has no coverage.  Result is cached in `<desktop>/cache` keyed by the DEM's
size + mtime and the grid geometry.  Nothing is ever written into the data root.

Also here (CPU helpers for the renderer): `fill_nan`, `terrain_stats`, `build_normal_map`, `normal_mip_chain`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .. import config
from ..geo import GridFrame

FEET_TO_M = 0.3048
DEM_NODATA = -9999.0
DEM_NAME = "dem_bg.tif"
CACHE_VERSION = 1


def dem_to_metres(raw: np.ndarray, nodata: float | None = DEM_NODATA) -> np.ndarray:
    """Raw DEM values (feet) -> float32 metres; NaN for nodata (-9999 / file nodata) and non-finite values."""
    raw = np.asarray(raw)
    bad = ~np.isfinite(raw) | (raw == DEM_NODATA)
    if nodata is not None:
        bad |= raw == nodata
    z = (raw.astype(np.float64) * FEET_TO_M).astype(np.float32)
    z[bad] = np.nan
    return z


def _cache_key(dem: Path, frame: GridFrame) -> str:
    st = dem.stat()
    s = f"{CACHE_VERSION}|{st.st_size}|{int(st.st_mtime)}|{frame.width}x{frame.height}|{frame.cell_m!r}|" \
        f"{frame.xmin!r},{frame.ymin!r},{frame.xmax!r},{frame.ymax!r}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def terrain_cache_path(root, frame: GridFrame) -> Path:
    return config.cache_dir() / f"terrain_{_cache_key(Path(root) / DEM_NAME, frame)}.npy"


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


def load_terrain(root, frame: GridFrame, use_cache: bool = True) -> np.ndarray:
    """float32[H, W] ground elevation in metres on the flow grid (row 0 = north), NaN = no DEM coverage."""
    dem = Path(root) / DEM_NAME
    if not dem.is_file():
        raise FileNotFoundError(f"terrain not found: {dem}")
    cache = terrain_cache_path(root, frame)
    if use_cache and cache.is_file():
        try:
            a = np.load(cache)
            if a.shape == (frame.height, frame.width) and a.dtype == np.float32:
                return a
        except Exception:
            pass
    z = warp_dem(dem, frame)
    if use_cache:
        _atomic_save(cache, z)
    return z


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


def load_terrain_gpu_data(root, frame: GridFrame):
    """Everything the renderer uploads, from cache when possible.

    Returns (height_filled float32[H,W], normal_mips list[uint8[h,w,4]] (level 0 first), stats dict, valid bool[H,W]).
    Cached files (desktop/cache): filled height, flat concatenated normal mip chain, validity mask is derived from
    the normal map alpha."""
    key = _cache_key(Path(root) / DEM_NAME, frame)
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
    z = load_terrain(root, frame)
    stats = terrain_stats(z)
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
