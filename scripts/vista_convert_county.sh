#!/bin/bash
# RUN ON VISTA (login node):  bash vista_convert_county.sh [--dry-run] [--only "25yr ev4"] [--simplify 5]
# Converts the county solver-v2 depth rasters (runs/county_v2/<storm>/depth_*.tif) to tier-polygon GeoJSON ON VISTA,
# one batch task per storm (CPU only), so only the GeoJSON has to be downloaded.  Needs scripts/tif_to_geojson.py
# next to this script's BASE folder ($SCRATCH/bgflood/tif_to_geojson.py; vista_convert_v2.sh wrote it, or scp it).
# Output: $SCRATCH/bgflood/county_geojson/<storm>/county_<storm>_depth_<seconds>s.geojson (UTM 16N, tiers 0.1/0.3/0.6/1.2 m)
set -u
BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
DRY=0; ONLY=""; SIMP=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="$2"; shift ;;
    --simplify) SIMP="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac; shift
done
[[ -f "$SCRATCH/miniforge3/bin/activate" ]] || { echo "FAIL: $SCRATCH/miniforge3/bin/activate missing"; exit 1; }
source "$SCRATCH/miniforge3/bin/activate"
[[ -f "$BASE/tif_to_geojson.py" ]] || { echo "FAIL: $BASE/tif_to_geojson.py missing - scp scripts/tif_to_geojson.py to $BASE/"; exit 1; }
python -c "import rasterio, numpy, shapely" 2>/dev/null || { echo "FAIL: rasterio/numpy/shapely missing in the miniforge env (python -m pip install shapely)"; exit 1; }

ALL="2yr 5yr 10yr 25yr 50yr 100yr ev1 ev2 ev3 ev4 ev5 ev6"
[[ -n "$ONLY" ]] && ALL="$ONLY"
OK=()
for s in $ALL; do
  n=$(ls "$BASE/runs/county_v2/$s"/depth_*.tif 2>/dev/null | wc -l)
  if [[ ! -f "$BASE/runs/county_v2/$s/run_summary.json" ]]; then echo "SKIP $s: no run_summary.json (run not finished)"
  elif [[ "$n" -lt 24 ]]; then echo "SKIP $s: only $n depth files (expected 24-25)"
  else echo "ok   $s: $n depth files"; OK+=("$s"); fi
done
[[ ${#OK[@]} -gt 0 ]] || { echo "nothing to convert"; exit 1; }
mkdir -p "$BASE/logs" "$BASE/county_geojson"
cat > "$BASE/convert_county_job.sh" <<'JOBEOF'
#!/bin/bash
#SBATCH --job-name=bgconv_county
#SBATCH --output=__BASE__/logs/convc_%A_%a.out
#SBATCH --error=__BASE__/logs/convc_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
STORMS=(__STORMS__)
S=${STORMS[$SLURM_ARRAY_TASK_ID]}
source $SCRATCH/miniforge3/bin/activate
cd __BASE__
echo "converting county $S"
python tif_to_geojson.py runs/county_v2/$S county_$S --out county_geojson/$S --simplify __SIMP__
echo "done $S, exit $?"
JOBEOF
sed -i "s#__BASE__#$BASE#g; s#__STORMS__#${OK[*]}#; s#__SIMP__#$SIMP#" "$BASE/convert_county_job.sh"
LAST=$(( ${#OK[@]} - 1 ))
echo "Will convert: ${OK[*]}   (simplify ${SIMP} m)"
if [[ $DRY -eq 1 ]]; then echo "DRY RUN: wrote $BASE/convert_county_job.sh, submitted nothing"; exit 0; fi
if OUT=$(sbatch --array=0-$LAST "$BASE/convert_county_job.sh" 2>&1); then echo "$OUT"; else echo "sbatch failed: $OUT"; exit 1; fi
echo "Watch:  squeue -u rlesoski -r    |   tail -f $BASE/logs/convc_<JOBID>_0.out"
echo "Output: $BASE/county_geojson/<storm>/county_<storm>_depth_<seconds>s.geojson   (check sizes: du -sh $BASE/county_geojson/*)"
