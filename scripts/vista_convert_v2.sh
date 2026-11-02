#!/bin/bash
# PASTE ON VISTA (login node). Converts the solver-v2 depth rasters to tier-polygon GeoJSON ON VISTA, one batch task per storm,
# so only the small GeoJSON files need to be downloaded. Costs a few node-hours (one node per storm, CPU only).
set -u
BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
[[ -f "$SCRATCH/miniforge3/bin/activate" ]] || { echo "FAIL: $SCRATCH/miniforge3/bin/activate missing"; exit 1; }
source "$SCRATCH/miniforge3/bin/activate"
for s in 5yr 10yr 25yr 50yr 100yr; do
  n=$(ls "$BASE/runs/output_v2/$s"/depth_*.tif 2>/dev/null | wc -l)
  [[ "$n" -ge 24 ]] || { echo "FAIL: $s has only $n depth files"; exit 1; }
done
python -c "import rasterio, numpy" 2>/dev/null || { echo "FAIL: rasterio/numpy missing in the miniforge env"; exit 1; }
python -c "import shapely" 2>/dev/null || { echo "shapely missing - installing it (login node, one small package)"; python -m pip install --quiet shapely || { echo "FAIL: could not install shapely"; exit 1; }; }
python -c "import shapely; print('shapely', shapely.__version__)"
mkdir -p "$BASE/logs" "$BASE/geojson_v2"
cat > "$BASE/tif_to_geojson.py" <<'PYEOF'
"""
Convert Vista depth rasters (depth_<seconds>s.tif) to tier-polygon GeoJSON in UTM 16N, the format
reproject_hpc.py expects:  <prefix>_depth_<seconds>s.geojson,  properties {tier, color, file, run}.

Tiers (m):  shallow 0.1-0.3 | moderate 0.3-0.6 | deep 0.6-1.2 | severe >= 1.2
Each polygon is one connected region of a single tier (holes kept), simplified to 2 m.

Usage:
  python tif_to_geojson.py <folder> <prefix>             all depth_*.tif directly inside <folder>
  python tif_to_geojson.py <file.tif> <prefix>           one raster
  options:  --out <dir>      write elsewhere (default: next to the TIF)
            --simplify <m>   tolerance in metres (default 2.0; 0 = off)
            --force          convert even if the raster fails the sanity check

Sanity check (skips the raster with an explanation): more than 40 % of valid cells wet, or more than
0.5 % of valid cells at >= 19.9 m. Both mean an unphysical run (e.g. the 1000x-rain runs pinned at the
20 m cap). Rasters with no valid cells are skipped as empty.

Needs: rasterio, shapely, numpy
"""
import argparse, json, os, re, sys, time
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping

TIERS = {1: ('shallow', '#4dabf7'), 2: ('moderate', '#00c9c9'), 3: ('deep', '#ff8c00'), 4: ('severe', '#e74c3c')}
BINS = [0.1, 0.3, 0.6, 1.2]


def rnd(c):
    if isinstance(c[0], (int, float)):
        return [round(c[0], 1), round(c[1], 1)]
    return [rnd(x) for x in c]


def check(depth, valid):
    v = depth[valid]
    wet = float((v > 0.1).sum()) / max(v.size, 1)
    capped = float((v >= 19.9).sum()) / max(v.size, 1)
    return wet, capped


def convert(tif, prefix, out_dir, tol, force, out_sec=None):
    # returns 'ok', 'empty' or 'refused'
    name = os.path.basename(tif)
    m = re.match(r'depth_(\d+)s\.tif$', name)
    if not m:
        print(f'  [skip] {name}: not a depth_<seconds>s.tif file')
        return 'empty'
    sec = m.group(1)
    out_sec = sec if out_sec is None else f'{out_sec:06d}'
    t0 = time.time()
    with rasterio.open(tif) as src:
        depth = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata if src.nodata is not None else -9999.0
        crs = src.crs
    valid = depth != nodata
    if not valid.any():
        print(f'  [skip] {name}: empty raster (no valid cells)')
        return 'empty'
    wet, capped = check(depth, valid)
    if not force and (wet > 0.40 or capped > 0.005):
        print(f'  [REFUSED] {name}: {100*wet:.0f}% of cells wet, {100*capped:.1f}% at the 20 m cap - '
              f'this looks like an unphysical run. Use --force to convert anyway.')
        return 'refused'
    depth[~valid] = 0.0
    depth[depth < 0] = 0.0
    tier = np.digitize(depth, BINS).astype(np.uint8)       # 0 = dry, 1..4 = tiers
    del depth
    out_name = f'{prefix}_depth_{out_sec}s.geojson'
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    stem = f'depth_{sec}s'
    run = prefix
    rp = int(re.match(r'(\d+)yr$', prefix).group(1)) if re.match(r'\d+yr$', prefix) else None
    counts = {k: 0 for k in TIERS}
    n = 0
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('{"type":"FeatureCollection","features":[')
        first = True
        for geom, val in shapes(tier, mask=tier > 0, transform=transform, connectivity=4):
            t = int(val)
            g = geom
            if tol > 0 and sum(len(r) for r in g['coordinates']) > 8:
                sg = shape(g).simplify(tol, preserve_topology=True)
                if sg.is_empty:
                    continue
                g = mapping(sg)
                if g['type'] != 'Polygon':
                    continue
            coords = rnd([[list(p) for p in ring] for ring in g['coordinates']])
            props = {'tier': TIERS[t][0], 'color': TIERS[t][1], 'file': stem, 'run': run}
            if rp is not None:
                props['rp'] = rp
            if not first:
                fh.write(',')
            fh.write(json.dumps({'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': coords},
                                 'properties': props}, separators=(',', ':')))
            first = False
            counts[t] += 1
            n += 1
        fh.write(']}')
    mb = os.path.getsize(out_path) / 1e6
    print(f'  {name} -> {out_name}  {mb:.1f} MB  {n:,} polygons '
          f'(shallow {counts[1]:,} / moderate {counts[2]:,} / deep {counts[3]:,} / severe {counts[4]:,})  '
          f'{time.time()-t0:.0f}s  [{crs}]')
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('prefix')
    ap.add_argument('--out')
    ap.add_argument('--simplify', type=float, default=2.0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--skip-hours', default='', help='comma-separated hours to skip (already done)')
    a = ap.parse_args()
    skip = {int(x) for x in a.skip_hours.split(',') if x.strip()}
    if os.path.isdir(a.path):
        out_dir = a.out or a.path
        # Group rasters by the whole hour they belong to (solver names files by actual time, e.g. 003603s).
        # Output is named by the nominal hour (003600s) so reproject_hpc.py finds it; t=0 is skipped.
        by_hour = {}
        for f in os.listdir(a.path):
            m = re.match(r'depth_(\d+)s\.tif$', f)
            if not m:
                continue
            sec = int(m.group(1))
            h = round(sec / 3600)
            if h >= 1 and h not in skip:
                by_hour.setdefault(h, []).append((abs(sec - h * 3600), os.path.join(a.path, f)))
        if not by_hour:
            print(f'Nothing to convert in {a.path} (no hourly rasters, or all hours already done).')
            return
        done = refused = 0
        for h in sorted(by_hour):
            for _, tif in sorted(by_hour[h]):
                r = convert(tif, a.prefix, out_dir, a.simplify, a.force, out_sec=h * 3600)
                if r == 'ok':
                    done += 1
                    break
                if r == 'refused':
                    refused += 1
                    break
            else:
                print(f'  [hour {h}] no usable raster')
        print(f'\n{done}/{len(by_hour)} hours converted ({refused} refused) -> {out_dir}')
        return
    out_dir = a.out or os.path.dirname(os.path.abspath(a.path))
    r = convert(a.path, a.prefix, out_dir, a.simplify, a.force)
    print(f'\n{r} -> {out_dir}')

if __name__ == '__main__':
    main()
PYEOF
cat > "$BASE/convert_v2_job.sh" <<'JOBEOF'
#!/bin/bash
#SBATCH --job-name=bgconv_v2
#SBATCH --output=__BASE__/logs/conv_%A_%a.out
#SBATCH --error=__BASE__/logs/conv_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-4
STORMS=(5yr 10yr 25yr 50yr 100yr)
S=${STORMS[$SLURM_ARRAY_TASK_ID]}
source $SCRATCH/miniforge3/bin/activate
cd __BASE__
echo "converting $S"
python tif_to_geojson.py runs/output_v2/$S v2_$S --out geojson_v2/$S
echo "done $S, exit $?"
JOBEOF
sed -i "s#__BASE__#$BASE#g" "$BASE/convert_v2_job.sh"
if OUT=$(sbatch "$BASE/convert_v2_job.sh" 2>&1); then echo "$OUT"; else echo "sbatch failed: $OUT"; exit 1; fi
echo "Watch:  squeue -u rlesoski -r     |   tail -f $BASE/logs/conv_<JOBID>_0.out   (task 0 = 5yr ... 4 = 100yr)"
echo "Output: $BASE/geojson_v2/<storm>/v2_<storm>_depth_<seconds>s.geojson"
