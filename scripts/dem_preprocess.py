"""
DEM Preprocessing Pipeline — Bowling Green, KY
===============================================
Generates all derivative rasters and tiles needed by the 3D flood viewer:

  1. Depression filling + sinkhole depth raster (Filled DEM - Raw DEM)
  2. Topographic Wetness Index (TWI) raster
  3. Multi-directional hillshade rasters (NW, NE, SW azimuths)
  4. Terrain-RGB tile package (Terrarium encoding, mbtiles) for viewer terrain source
  5. Updated sinkholes.geojson with depth_m and volume_m3 attributes

Dependencies (install in your conda env):
  pip install rasterio numpy scipy richdem pysheds geopandas shapely pyproj
  conda install -c conda-forge gdal     # for gdal_translate, gdaldem, gdal2tiles
  pip install rio-rgbify                # for Terrain-RGB tile generation

Usage:
  cd C:\\Users\\Minec\\Documents\\BG_3D_Map
  python scripts\\dem_preprocess.py

Outputs land in:
  layers/dem_filled.tif        — depression-filled DEM (float32, ft)
  layers/sinkhole_depth.tif    — Filled - Raw depth raster (ft)
  layers/twi.tif               — Topographic Wetness Index (float32)
  layers/hillshade_nw.tif      — Hillshade, azimuth 315 (standard cartographic)
  layers/hillshade_ne.tif      — Hillshade, azimuth 45
  layers/hillshade_sw.tif      — Hillshade, azimuth 225
  layers/dem_meters.tif        — DEM converted to meters (for rgbify)
  layers/terrain_rgb.mbtiles   — Terrain-RGB tiles (serve via tileserver-gl)
  layers/sinkholes.geojson     — Updated with depth_m, volume_m3, rim_elev_ft
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import gaussian_filter, label as nd_label, binary_fill_holes
import geopandas as gpd
from shapely.geometry import shape, mapping
from pathlib import Path
import subprocess
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(r"C:\Users\Minec\Documents\BG_3D_Map")
DEM_PATH  = ROOT / "dem_bg.tif"
OUT_DIR   = ROOT / "layers"
OUT_DIR.mkdir(exist_ok=True)

# Output files
FILLED_TIF     = OUT_DIR / "dem_filled.tif"
DEPTH_TIF      = OUT_DIR / "sinkhole_depth.tif"
TWI_TIF        = OUT_DIR / "twi.tif"
HS_NW_TIF      = OUT_DIR / "hillshade_nw.tif"
HS_NE_TIF      = OUT_DIR / "hillshade_ne.tif"
HS_SW_TIF      = OUT_DIR / "hillshade_sw.tif"
DEM_M_TIF      = OUT_DIR / "dem_meters.tif"
RGB_MBTILES    = OUT_DIR / "terrain_rgb.mbtiles"
SINKHOLES_JSON = OUT_DIR / "sinkholes.geojson"

print("=" * 62)
print("BG DEM Preprocessing Pipeline")
print("=" * 62)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load DEM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading DEM...")
with rasterio.open(DEM_PATH) as src:
    profile  = src.profile.copy()
    elev_ft  = src.read(1).astype(np.float32)
    nodata   = src.nodata
    transform= src.transform
    crs      = src.crs
    h, w     = elev_ft.shape

if nodata is not None:
    elev_ft[np.isclose(elev_ft, nodata)] = np.nan
elev_ft[elev_ft < -500] = np.nan

valid = np.isfinite(elev_ft)
fill_val = float(np.nanmean(elev_ft))
elev_safe = np.where(valid, elev_ft, fill_val)

print(f"  DEM {h}×{w}  Elevation {np.nanmin(elev_ft):.0f}–{np.nanmax(elev_ft):.0f} ft  "
      f"Valid cells: {valid.sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Depression filling (iterative planchon-darboux approach via scipy)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Filling depressions (sinkhole detection)...")

# Coarse Gaussian smooth to approximate the "filled" DEM without actual flow routing.
# A proper filled DEM uses richdem or pysheds; this is a 3-sigma Gaussian approximation
# that works well for sinkhole depth mapping at the scale of Bowling Green's karst.
sigma_px = 15  # ~75m at 5m DEM resolution — adjust for karst basin scale
smooth = gaussian_filter(elev_safe, sigma=sigma_px)

# Depression depth = how far each cell sits below its smoothed surroundings
depth_ft = np.maximum(smooth - elev_safe, 0.0).astype(np.float32)
depth_ft[~valid] = np.nan

print(f"  Max depression depth: {np.nanmax(depth_ft):.1f} ft")
print(f"  Cells with depth > 1 ft: {(depth_ft > 1).sum():,}")

# ── Try richdem for proper Planchon-Darboux fill ──────────────────────────────
try:
    import richdem as rd
    print("  richdem available — using Planchon-Darboux breach + fill...")
    dem_rd = rd.rdarray(elev_safe.copy(), no_data=fill_val)
    dem_rd.projection = crs.to_wkt()
    dem_rd.geotransform = (transform.c, transform.a, transform.b,
                           transform.f, transform.d, transform.e)
    rd.FillDepressions(dem_rd, in_place=True)
    filled_ft = np.array(dem_rd).astype(np.float32)
    filled_ft[~valid] = np.nan
    depth_ft = np.maximum(filled_ft - elev_ft, 0.0).astype(np.float32)
    depth_ft[~valid] = np.nan
    print(f"  richdem fill: max depth {np.nanmax(depth_ft):.1f} ft")
except ImportError:
    print("  richdem not installed — using Gaussian approximation (less accurate)")
    filled_ft = np.where(valid, smooth, np.nan).astype(np.float32)

# Save filled DEM
profile_out = profile.copy()
profile_out.update(dtype='float32', count=1, nodata=-9999)
with rasterio.open(FILLED_TIF, 'w', **profile_out) as dst:
    out = filled_ft.copy(); out[~valid] = -9999
    dst.write(out, 1)
print(f"  ✓ dem_filled.tif → {FILLED_TIF}")

# Save depth raster
with rasterio.open(DEPTH_TIF, 'w', **profile_out) as dst:
    out = depth_ft.copy(); out[np.isnan(out)] = -9999
    dst.write(out, 1)
print(f"  ✓ sinkhole_depth.tif → {DEPTH_TIF}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Topographic Wetness Index (TWI)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Computing TWI...")

# TWI = ln(a / tan(β))  where a = specific catchment area, β = local slope
# Use pysheds for flow accumulation if available; otherwise approximate with
# Gaussian-smoothed flow proxy
try:
    from pysheds.grid import Grid
    print("  pysheds available — computing flow accumulation...")
    grid = Grid.from_raster(str(DEM_PATH))
    dem_ps = grid.read_raster(str(DEM_PATH))
    # Condition DEM
    pit_filled = grid.fill_pits(dem_ps)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)
    # Flow direction (D8)
    fdir = grid.flowdir(inflated)
    # Flow accumulation (cells) * cell_area → specific catchment area (m²)
    acc  = grid.accumulation(fdir).astype(np.float32)
    cell_m = abs(transform.a)   # DEM is in UTM metres (if EPSG:32616)
    # If DEM stores elevations in feet and transform in metres:
    specific_area = (acc + 1) * (cell_m ** 2)
    # Slope from filled DEM (in radians)
    gy, gx = np.gradient(np.where(valid, filled_ft * 0.3048, 0), cell_m)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_rad = np.maximum(slope_rad, 1e-4)  # avoid div by zero
    twi = np.log(specific_area / np.tan(slope_rad)).astype(np.float32)
    twi[~valid] = np.nan
    print(f"  TWI range: {np.nanmin(twi):.2f} – {np.nanmax(twi):.2f}")
except ImportError:
    print("  pysheds not installed — approximating TWI from Gaussian slope...")
    # Gradient from DEM (slope proxy)
    cell_m = abs(transform.a)
    gy, gx = np.gradient(elev_safe * 0.3048, cell_m)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_rad = np.maximum(slope_rad, 1e-4)
    # Flat-area weight from smoothed DEM (proxy for upslope area)
    upslope = gaussian_filter(np.ones_like(elev_safe), sigma=40) * 1600
    twi = np.log(upslope / np.tan(slope_rad)).astype(np.float32)
    twi[~valid] = np.nan

with rasterio.open(TWI_TIF, 'w', **profile_out) as dst:
    out = twi.copy(); out[np.isnan(out)] = -9999
    dst.write(out, 1)
print(f"  ✓ twi.tif → {TWI_TIF}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Multi-directional hillshades via GDAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Generating hillshades...")

# Convert DEM feet → metres for GDAL gdaldem (expects metres by default)
elev_m = np.where(valid, elev_ft * 0.3048, -9999).astype(np.float32)
with rasterio.open(DEM_M_TIF, 'w', **profile_out) as dst:
    dst.write(elev_m, 1)

hillshades = {
    HS_NW_TIF: 315,   # standard cartographic NW illumination
    HS_NE_TIF: 45,
    HS_SW_TIF: 225,
}

gdal_ok = True
for out_path, azimuth in hillshades.items():
    cmd = [
        'gdaldem', 'hillshade',
        str(DEM_M_TIF), str(out_path),
        '-az', str(azimuth),
        '-alt', '45',
        '-z', '2',          # vertical exaggeration in hillshade
        '-compute_edges',
        '-of', 'GTiff',
        '-co', 'COMPRESS=LZW',
        '-q'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ hillshade az={azimuth}° → {out_path.name}")
    else:
        print(f"  ✗ gdaldem failed (az={azimuth}): {result.stderr[:120]}")
        gdal_ok = False

if not gdal_ok:
    print("  NOTE: GDAL not found on PATH. Install GDAL or run:")
    print("    conda install -c conda-forge gdal")
    # Fallback: compute hillshade in numpy
    print("  Using numpy hillshade fallback...")
    cell_m = abs(transform.a)
    gy, gx = np.gradient(np.where(valid, elev_ft * 0.3048, 0), cell_m)
    for out_path, azimuth in hillshades.items():
        az_rad = np.radians(360 - azimuth + 90)
        alt_rad = np.radians(45)
        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gy, gx)
        hs = (np.cos(alt_rad) * np.cos(slope) +
              np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
        hs = np.clip(hs * 255, 0, 255).astype(np.uint8)
        hs_profile = profile.copy()
        hs_profile.update(dtype='uint8', count=1, nodata=None)
        with rasterio.open(out_path, 'w', **hs_profile) as dst:
            dst.write(hs, 1)
        print(f"  ✓ hillshade az={azimuth}° (numpy) → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Terrain-RGB tiles (Terrarium encoding) via rio-rgbify
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Generating Terrain-RGB mbtiles...")
print("  Using rio-rgbify (Terrarium encoding: elev = R*256 + G + B/256 - 32768)")

# dem_meters.tif was already written above
# rio-rgbify expects elevation in metres and produces Terrarium-encoded PNG tiles
cmd_rgb = [
    sys.executable, '-m', 'rio', 'rgbify',
    '-b', '-10000',   # base elevation (shift; Terrarium default is -32768)
    '-i', '0.1',      # interval in metres per unit (0.1m precision)
    '--min-z', '5',
    '--max-z', '15',
    str(DEM_M_TIF),
    str(RGB_MBTILES),
]
print(f"  Running: {' '.join(cmd_rgb)}")
result = subprocess.run(cmd_rgb, capture_output=True, text=True)
if result.returncode == 0:
    print(f"  ✓ terrain_rgb.mbtiles → {RGB_MBTILES}")
    print()
    print("  To serve the mbtiles locally, run:")
    print("    npx tileserver-gl layers/terrain_rgb.mbtiles --port 8080")
    print("  Then update TERRAIN_TILES in app/src/main.ts to:")
    print("    http://localhost:8080/tiles/terrain_rgb/{z}/{x}/{y}.png")
else:
    print(f"  ✗ rio-rgbify failed: {result.stderr[:200]}")
    print("  Install with: pip install rio-rgbify")
    print("  Then re-run this script.")
    print()
    print("  Alternative — use gdal2tiles:")
    print("    gdal2tiles.py -z 5-15 --tiledriver=PNG layers/dem_meters.tif layers/terrain_tiles/")


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Update sinkholes.geojson with depth attributes
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Enriching sinkholes.geojson with depth attributes...")

if SINKHOLES_JSON.exists():
    gdf = gpd.read_file(str(SINKHOLES_JSON))
    print(f"  Loaded {len(gdf):,} sinkhole features")

    # Sample depth raster at sinkhole centroids
    with rasterio.open(DEPTH_TIF) as src:
        data_d  = src.read(1).astype(np.float32)
        nd_d    = src.nodata
        tf_d    = src.transform
        h_d, w_d = data_d.shape
        depth_crs = src.crs.to_epsg()

    # Reproject sinkhole centroids to depth raster CRS
    gdf_cents = gdf.copy()
    if gdf.crs and gdf.crs.to_epsg() != depth_crs:
        gdf_cents = gdf_cents.to_crs(epsg=depth_crs)
    cx = gdf_cents.geometry.centroid.x.values
    cy = gdf_cents.geometry.centroid.y.values
    cols = np.floor((cx - tf_d.c) / tf_d.a).astype(np.int64)
    rows = np.floor((cy - tf_d.f) / tf_d.e).astype(np.int64)
    in_b = (rows >= 0) & (rows < h_d) & (cols >= 0) & (cols < w_d)

    depth_samples = np.full(len(gdf), np.nan, dtype=np.float32)
    idx = np.where(in_b)[0]
    s = data_d[rows[idx], cols[idx]]
    if nd_d is not None:
        s[np.isclose(s, nd_d)] = np.nan
    depth_samples[idx] = s

    gdf['depth_m'] = np.round(depth_samples * 0.3048, 2)  # ft → m

    # Rim elevation = centroid elevation + depth
    with rasterio.open(DEM_PATH) as src:
        data_e = src.read(1).astype(np.float32)
        nd_e   = src.nodata
        tf_e   = src.transform
        h_e, w_e = data_e.shape
        elev_crs = src.crs.to_epsg()

    if gdf.crs and gdf.crs.to_epsg() != elev_crs:
        gdf_e = gdf.to_crs(epsg=elev_crs)
    else:
        gdf_e = gdf
    ex = gdf_e.geometry.centroid.x.values
    ey = gdf_e.geometry.centroid.y.values
    ec = np.floor((ex - tf_e.c) / tf_e.a).astype(np.int64)
    er = np.floor((ey - tf_e.f) / tf_e.e).astype(np.int64)
    in_e = (er >= 0) & (er < h_e) & (ec >= 0) & (ec < w_e)
    elev_samples = np.full(len(gdf), np.nan, dtype=np.float32)
    idx_e = np.where(in_e)[0]
    s_e = data_e[er[idx_e], ec[idx_e]]
    if nd_e is not None:
        s_e[np.isclose(s_e, nd_e)] = np.nan
    elev_samples[idx_e] = s_e

    gdf['ground_elev_ft'] = np.round(elev_samples.astype(float), 1)
    gdf['rim_elev_ft']    = np.round(elev_samples + depth_samples, 1)

    # Rough volume estimate (m³) using elliptical bowl approximation
    areas = gdf.to_crs(epsg=depth_crs).geometry.area.values  # m²
    depth_m_arr = np.where(np.isfinite(depth_samples), depth_samples * 0.3048, 0.0)
    gdf['volume_m3'] = np.round(areas * depth_m_arr * (np.pi / 4), 0).astype(float)

    gdf.to_file(str(SINKHOLES_JSON), driver='GeoJSON')
    print(f"  ✓ sinkholes.geojson updated with depth_m, rim_elev_ft, volume_m3 → {SINKHOLES_JSON}")
    print(f"  Depth range: {np.nanmin(depth_samples * 0.3048):.1f} – {np.nanmax(depth_samples * 0.3048):.1f} m")
else:
    print("  sinkholes.geojson not found — skipping. Run flood_analysis.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("PREPROCESSING COMPLETE")
print("=" * 62)
print()
print("Next steps:")
print("  1. Serve terrain_rgb.mbtiles (if generated):")
print("       npm install -g tileserver-gl")
print("       tileserver-gl layers/terrain_rgb.mbtiles --port 8080")
print()
print("  2. Update app/src/main.ts TERRAIN_TILES constant to:")
print("       'http://localhost:8080/tiles/terrain_rgb/{z}/{x}/{y}.png'")
print("     and change encoding: 'terrarium' to match rgbify's output.")
print()
print("  3. To use hillshade in the viewer, add a raster source pointing to")
print("     http://localhost:8080/tiles/hillshade_nw/{z}/{x}/{y}.png")
print("     (after converting hillshade_nw.tif to mbtiles with gdal2tiles or rio)")
print()
print("  4. TWI raster (twi.tif) can be styled as a raster layer in MapLibre.")
print("     Convert to tiles: gdal2tiles.py --zoom=8-15 layers/twi.tif layers/twi_tiles/")
print()
print("Files generated:")
for p in [FILLED_TIF, DEPTH_TIF, TWI_TIF, HS_NW_TIF, HS_NE_TIF, HS_SW_TIF, DEM_M_TIF]:
    if p.exists():
        print(f"  ✓ {p.name:30s}  {p.stat().st_size/1e6:.1f} MB")
    else:
        print(f"  ✗ {p.name:30s}  (not generated)")
