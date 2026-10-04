"""
Compute rainfall flood simulation from dem_bg.tif.
Outputs layers/flood_sim.geojson  (polygons with min_rain_in property)
       layers/min_rain.tif        (float32 raster, sampled by export_all_parcels.py)

Method:
  - Elevation percentiles define when an area floods (primary driver)
  - Karst depression depth provides a ~1-level early-flood boost for sinkhole basins
  - Results are realistic for BG's mixed karst / riverine / upland topography
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import gaussian_filter, binary_opening, generate_binary_structure
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer
from pathlib import Path

DEM_PATH     = Path(r"C:\Users\Minec\Documents\BG_3D_Map\dem_bg.tif")
MIN_RAIN_TIF = Path(r"C:\Users\Minec\Documents\BG_3D_Map\layers\min_rain.tif")
OUT_JSON     = Path(r"C:\Users\Minec\Documents\BG_3D_Map\layers\flood_sim.geojson")
MAX_RAIN     = 3.0
WGS84        = "EPSG:4326"
UTM16N       = "EPSG:32616"

# Rainfall levels and corresponding elevation percentile thresholds
# Interpretation: at rain=X", areas in the bottom P% by elevation flood
RAIN_PCT = [
    (0.25, 3),    # active floodplain only
    (0.50, 7),    # low-lying river corridors
    (0.75, 12),   # major sinkholes + stream bottoms
    (1.00, 18),   # moderate karst basins
    (1.25, 25),
    (1.50, 33),
    (2.00, 43),
    (2.50, 55),
    (3.00, 67),
]

# ── 1. Load DEM ───────────────────────────────────────────────────────────────
print("Loading DEM...", flush=True)
with rasterio.open(DEM_PATH) as src:
    elev = src.read(1).astype(np.float32)
    nodata_val = src.nodata
    transform  = src.transform
    raster_crs = src.crs.to_string()

if nodata_val is not None:
    elev[np.isclose(elev, nodata_val)] = np.nan
elev[elev < -500] = np.nan

valid = np.isfinite(elev)
print(f"  Shape {elev.shape}  Elevation {np.nanmin(elev):.0f}–{np.nanmax(elev):.0f} ft  "
      f"Valid cells: {valid.sum():,}", flush=True)

# ── 2. Compute depression depth for karst boost ───────────────────────────────
fill_val = float(np.nanmean(elev))
smooth_coarse = gaussian_filter(np.where(valid, elev, fill_val), sigma=20)
depression    = np.maximum(smooth_coarse - elev, 0)  # how deep each cell is below its surroundings
# Top 20% by depression depth = significant sinkhole basins
depr_thresh = np.percentile(depression[valid], 80)
in_sinkhole  = valid & (depression > depr_thresh)
print(f"  Sinkhole cells (top 20% depression): {in_sinkhole.sum():,}", flush=True)

# ── 3. Build min_rain raster from elevation percentiles ───────────────────────
min_rain_raster = np.full(elev.shape, MAX_RAIN + 1.0, dtype=np.float32)

rain_levels = [r for r, _ in RAIN_PCT]
pct_levels  = [p for _, p in RAIN_PCT]
elev_thresholds = [np.nanpercentile(elev[valid], p) for p in pct_levels]

# Iterate in REVERSE so low-rain levels are written last and win (overwrite high-rain assignments)
for rain, thresh in reversed(list(zip(rain_levels, elev_thresholds))):
    min_rain_raster[valid & (elev <= thresh)] = rain

# Karst boost: sinkhole cells flood one rain level earlier
for i in range(1, len(rain_levels)):
    # Cells that would flood at rain_levels[i] but are in a sinkhole → rain_levels[i-1]
    mask = in_sinkhole & (min_rain_raster == rain_levels[i])
    min_rain_raster[mask] = rain_levels[i - 1]

# Set nodata areas
min_rain_raster[~valid] = -9999

print("Elevation thresholds:", flush=True)
for rain, pct, thresh in zip(rain_levels, pct_levels, elev_thresholds):
    n_cells = ((min_rain_raster == rain) & valid).sum()
    print(f"  {rain:.2f}\" ≤ elev {thresh:.0f} ft (P{pct})  {n_cells:,} cells", flush=True)

# ── 4. Save min_rain raster ───────────────────────────────────────────────────
with rasterio.open(DEM_PATH) as src:
    profile = src.profile.copy()
profile.update(dtype="float32", count=1, nodata=-9999)
with rasterio.open(MIN_RAIN_TIF, "w", **profile) as dst:
    dst.write(min_rain_raster, 1)
print(f"\n  min_rain.tif saved → {MIN_RAIN_TIF}", flush=True)

# ── 5. Vectorize cumulative flood polygons per rain level ─────────────────────
print("\nVectorizing flood zones...", flush=True)
struct   = generate_binary_structure(2, 1)
features = []

# Downsample 4× to reduce vectorization vertex count
DSAMP = 4
mr_ds = min_rain_raster[::DSAMP, ::DSAMP]
# Rebuild a downsampled transform
from rasterio.transform import Affine
ds_transform = Affine(transform.a * DSAMP, transform.b, transform.c,
                      transform.d, transform.e * DSAMP, transform.f)
valid_ds = np.isfinite(mr_ds) & (mr_ds > -9000)

for rain in rain_levels:
    mask = ((mr_ds <= rain) & valid_ds).astype(np.uint8)
    n    = mask.sum()
    if n < 50:
        continue

    mask_c = binary_opening(mask, structure=struct, iterations=1).astype(np.uint8)
    if mask_c.sum() < 50:
        mask_c = mask

    polys = [shape(g) for g, v in rio_shapes(mask_c, mask=mask_c, transform=ds_transform) if v == 1]
    if not polys:
        continue

    merged = unary_union(polys)
    # Simplify ~150 m (≈ 0.0014°)
    merged = merged.simplify(0.0014, preserve_topology=True)
    if merged.is_empty:
        continue

    # Area via geopandas reproject
    gdf_tmp  = gpd.GeoDataFrame({"geometry": [merged]}, crs=WGS84).to_crs(UTM16N)
    area_km2 = float(gdf_tmp.geometry.area.values[0]) / 1e6

    features.append({
        "type": "Feature",
        "geometry": merged.__geo_interface__,
        "properties": {
            "min_rain_in": rain,
            "area_km2":    round(area_km2, 2),
            "label":       f"{rain:.2f}\" rainfall"
        }
    })
    print(f"  {rain:.2f}\" → {area_km2:.1f} km²  ({n:,} downsampled cells)", flush=True)

# ── 6. Save GeoJSON ───────────────────────────────────────────────────────────
gdf_flood = gpd.GeoDataFrame.from_features(features, crs=WGS84)
gdf_flood.to_file(str(OUT_JSON), driver="GeoJSON")
sz = OUT_JSON.stat().st_size / 1e6
print(f"\n✓ flood_sim.geojson: {len(gdf_flood)} bands  {sz:.1f} MB → {OUT_JSON}", flush=True)
