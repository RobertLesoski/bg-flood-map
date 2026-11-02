"""
Clip and prepare a 5 m Warren County DEM for the county-wide flood solver on Vista.

Run this ON VISTA before submitting submit_county.sh:
    python3 dem_county_clip.py

Input options (use whichever is available on Vista — checked in order):
  1. $SCRATCH/bgflood/dem/ky_5m_full.tif   — full KyFromAbove Phase 2 statewide mosaic
  2. $SCRATCH/bgflood/dem/ky_5m_nw.tif     — NW quadrant tile covering Warren County

Output:
  $SCRATCH/bgflood/dem/dem_warren_5m.tif   — 5 m UTM 16N DEM clipped to Warren County
  $SCRATCH/bgflood/dem/mannings_n_county.tif — roughness raster for solver

Warren County, KY bounding box (WGS84):  -86.695, 36.673, -86.160, 37.173
In UTM Zone 16N (approximate):           490000, 4060000, 535000, 4114000

The solver expects:
  - EPSG:32616 (UTM Zone 16N)
  - Elevations in METRES (float32)
  - nodata = -9999
"""
import os, sys, subprocess
import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask as rio_mask
    import fiona
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print('[warn] rasterio/fiona not available — falling back to GDAL CLI')

BASE    = os.path.join(os.environ.get('SCRATCH', '.'), 'bgflood')
DEM_DIR = os.path.join(BASE, 'dem')
OUT_TIF = os.path.join(DEM_DIR, 'dem_warren_5m.tif')
MAN_TIF = os.path.join(DEM_DIR, 'mannings_n_county.tif')
os.makedirs(DEM_DIR, exist_ok=True)

# Warren County in WGS84 (west, south, east, north)
COUNTY_BBOX_WGS84 = (-86.695, 36.673, -86.160, 37.173)
# Same in UTM 16N (metres) — used for GDAL projwin
COUNTY_BBOX_UTM = (490000, 4060000, 535000, 4114000)

UTM16N_EPSG = 'EPSG:32616'

SOURCE_CANDIDATES = [
    os.path.join(DEM_DIR, 'ky_5m_full.tif'),
    os.path.join(DEM_DIR, 'ky_5m_nw.tif'),
]

def find_source():
    for p in SOURCE_CANDIDATES:
        if os.path.exists(p):
            print(f'  Source DEM: {p}  ({os.path.getsize(p)/1e9:.2f} GB)')
            return p
    return None

def clip_with_gdal(src, dst):
    """Clip and reproject to UTM 16N using GDAL CLI (always available on Vista)."""
    xmin, ymin, xmax, ymax = COUNTY_BBOX_WGS84
    cmd = [
        'gdalwarp',
        '-te', str(xmin), str(ymin), str(xmax), str(ymax),
        '-te_srs', 'EPSG:4326',
        '-t_srs', UTM16N_EPSG,
        '-tr', '5', '5',
        '-r', 'bilinear',
        '-ot', 'Float32',
        '-dstnodata', '-9999',
        '-co', 'COMPRESS=LZW',
        '-co', 'BIGTIFF=YES',
        '-co', 'TILED=YES',
        src, dst,
    ]
    print('  Running gdalwarp...')
    print('  ', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('  FAIL:', r.stderr[:400])
        sys.exit(1)
    size_gb = os.path.getsize(dst) / 1e9
    print(f'  ✓ {dst}  ({size_gb:.2f} GB)')

def build_mannings(dem_path, out_path):
    """
    Build a uniform Manning's n roughness raster matching the county DEM extent.
    Uses 0.035 (moderate-roughness mixed urban/rural) as a single flat value.
    Replace with land-cover-derived values if NLCD is available on Vista.
    """
    print('  Building Manning\'s n raster (uniform n=0.035)...')
    if not os.path.exists(dem_path):
        print('  SKIP — DEM not found')
        return
    gdal_calc = ['gdal_calc.py', '-A', dem_path, f'--outfile={out_path}',
                 '--calc=A*0+0.035', '--type=Float32', '--NoDataValue=-9999',
                 '--quiet', '--overwrite']
    r = subprocess.run(gdal_calc, capture_output=True, text=True)
    if r.returncode == 0:
        print(f'  ✓ {out_path}')
    else:
        # Fallback: write manually with rasterio
        if HAS_GEO:
            with rasterio.open(dem_path) as src:
                profile = src.profile.copy()
                shape   = src.read(1).shape
            profile.update(dtype='float32', count=1, nodata=-9999)
            arr = np.full(shape, 0.035, dtype=np.float32)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(arr, 1)
            print(f'  ✓ {out_path}  (rasterio fallback)')
        else:
            print('  WARN: could not create mannings_n_county.tif — set n=0.035 manually in solver config')

# ── Main ──────────────────────────────────────────────────────────────────────
print('=' * 62)
print('Warren County DEM Clip — Vista prep script')
print('=' * 62)

if os.path.exists(OUT_TIF):
    size_gb = os.path.getsize(OUT_TIF) / 1e9
    print(f'\ndem_warren_5m.tif already exists ({size_gb:.2f} GB).')
    if size_gb >= 0.2:
        print('Looks complete. Delete it and re-run if you want to regenerate.')
        build_mannings(OUT_TIF, MAN_TIF)
        sys.exit(0)
    else:
        print('File is too small — may be corrupt. Regenerating...')
        os.remove(OUT_TIF)

print('\n[1] Locating source DEM...')
src = find_source()
if src is None:
    print('\n  ERROR: No source DEM found. Upload one of:')
    for p in SOURCE_CANDIDATES:
        print(f'    {p}')
    print()
    print('  Download options:')
    print('    KyFromAbove Phase 2 (5 m): https://kyfromabove.ky.gov/')
    print('    USGS 3DEP (1/9 arc-sec ≈ 3 m): https://apps.nationalmap.gov/downloader/')
    print()
    print('  Transfer to Vista:')
    print('    scp dem_ky_5m.tif rlesoski@stampede3.tacc.utexas.edu:$SCRATCH/bgflood/dem/ky_5m_full.tif')
    sys.exit(1)

print('\n[2] Clipping to Warren County and reprojecting to UTM 16N / 5 m...')
clip_with_gdal(src, OUT_TIF)

print('\n[3] Manning\'s n roughness raster...')
build_mannings(OUT_TIF, MAN_TIF)

print()
print('=' * 62)
print('COUNTY DEM READY')
print('=' * 62)
print(f'  dem_warren_5m.tif  → {OUT_TIF}')
print(f'  mannings_n_county  → {MAN_TIF}')
print()
print('Next: run the county preflight, then submit:')
print('  python3 vista_county_preflight.py')
print('  sbatch submit_county.sh')
