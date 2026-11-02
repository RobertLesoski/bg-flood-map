#!/bin/bash
# Builds dem/dem_warren_5m.tif and dem/mannings_n_county.tif. Run from $SCRATCH/bgflood.
set -e
cd "$SCRATCH/bgflood/dem"
OUT=dem_warren_5m.tif; MAN=mannings_n_county.tif
# Warren County bbox with a margin so the whole county, including its edge, is inside the model domain
W=-86.72; S=36.65; E=-86.14; N=37.19
need() { command -v "$1" >/dev/null || { echo "FATAL: $1 not found (conda install -c conda-forge gdal)"; exit 1; }; }
need gdalwarp; need gdal_calc.py; need gdalinfo
if [[ -f $OUT && -f $MAN ]] && [[ $(stat -c %s $OUT) -gt 200000000 ]]; then echo "  county grids already built - reusing"; exit 0; fi

echo "  downloading USGS 3DEP 1/3 arc-second tiles (about 10 m, public, no login)..."
for T in n37w087 n38w087; do
  F=USGS_13_${T}.tif
  [[ -s $F ]] || curl -fL --retry 4 -o $F "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/${T}/${F}"
  gdalinfo $F >/dev/null || { echo "FATAL: $F is corrupt - delete it and re-run"; exit 1; }
done

SRCS="USGS_13_n37w087.tif USGS_13_n38w087.tif"
# The 2 m KyFromAbove city DEM sits on top where it exists (later inputs win in gdalwarp).
CITY=dem_bg_2m.tif
if [[ -f $CITY ]]; then SRCS="$SRCS $CITY"; fi

echo "  warping to 5 m UTM 16N..."
gdalwarp -overwrite -t_srs EPSG:32616 -te_srs EPSG:4326 -te $W $S $E $N -tr 5 5 -r bilinear -ot Float32 \
  -srcnodata -9999 -dstnodata -9999 -multi -wo NUM_THREADS=ALL \
  -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES $SRCS $OUT

# Sanity: where the city DEM and 3DEP overlap they must agree to within a few metres (catches feet-vs-metres and datum slips)
python3 - <<'PYEOF'
import numpy as np, rasterio, os
from rasterio.warp import transform
# compare mid-county elevation (blended) against pure 3DEP at ~200 points inside the city DEM footprint
city = 'dem_bg_2m.tif'
if os.path.exists(city):
    with rasterio.open(city) as cd, rasterio.open('USGS_13_n37w087.tif') as u:
        l, b, r, t = cd.bounds
        xs = np.linspace(l + 2000, r - 2000, 15); ys = np.linspace(b + 2000, t - 2000, 15)
        px, py = np.meshgrid(xs, ys); px, py = px.ravel(), py.ravel()
        cv = np.array([v[0] for v in cd.sample(zip(px, py))], dtype='float64')
        lon, lat = transform(cd.crs, u.crs, px.tolist(), py.tolist())
        uv = np.array([v[0] for v in u.sample(zip(lon, lat))], dtype='float64')
        ok = (cv > -1000) & (uv > -1000)
        d = np.median(cv[ok] - uv[ok]); s = np.std(cv[ok] - uv[ok])
        print(f"  city DEM minus 3DEP over {ok.sum()} points: median {d:+.2f} m, std {s:.2f} m")
        if abs(d) > 3.0:
            raise SystemExit("FATAL: city DEM and 3DEP disagree by more than 3 m - units or datum mismatch; not building the county grid")
PYEOF

# Roughness: 0.035 everywhere (mixed rural), the city's detailed Manning's n on top where it exists.
echo "  roughness..."
gdal_calc.py -A $OUT --outfile=base_n.tif --calc="(A>-9000)*0.035" --type=Float32 --NoDataValue=-9999 --overwrite --quiet \
  --co COMPRESS=LZW --co TILED=YES --co BIGTIFF=YES
if [[ -f mannings_n.tif ]]; then
  gdalwarp -overwrite -t_srs EPSG:32616 -te_srs EPSG:4326 -te $W $S $E $N -tr 5 5 -r near -ot Float32 \
    -srcnodata -9999 -dstnodata -9999 -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES base_n.tif mannings_n.tif $MAN
else
  mv base_n.tif $MAN
fi
rm -f base_n.tif
gdalinfo $OUT | grep -E "Size is|Pixel Size"
echo "  county grids ready"
