#!/bin/bash
# RUN ON VISTA (login node):  bash vista_make_cogs.sh [--domain county|city|both] [--only "25yr ev4"] [--res-m 6.25] [--dry-run] [--here] [--partition gh]
# Turns runs/county_v2/<storm>/{max_depth,depth_*s}.tif (city: runs/output_v2/<storm>) into Cloud-Optimized GeoTIFFs
# (EPSG:3857, uint16 centimetres, 512 tiles, deflate, overviews) under $SCRATCH/bgflood/county_cog/<storm>/ (city: city_cog/<storm>/).
# CPU only, resumable (finished files are skipped), one batch task per storm; --here runs in this shell (use an idev session).
# Needs scripts/make_cogs.py in $SCRATCH/bgflood (scp it) and rasterio + numpy in the miniforge env.
set -u
BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
DOM=county; ONLY=""; RES=6.25; DRY=0; HERE=0; PART=gh
while [[ $# -gt 0 ]]; do case "$1" in
  --domain) DOM="$2"; shift ;; --only) ONLY="$2"; shift ;; --res-m) RES="$2"; shift ;; --dry-run) DRY=1 ;; --here) HERE=1 ;;
  --partition) PART="$2"; shift ;; *) echo "unknown option $1"; exit 1 ;; esac; shift; done
[[ -f "$SCRATCH/miniforge3/bin/activate" ]] || { echo "FAIL: $SCRATCH/miniforge3/bin/activate missing"; exit 1; }
source "$SCRATCH/miniforge3/bin/activate"
[[ -f "$BASE/make_cogs.py" ]] || { echo "FAIL: scp scripts/make_cogs.py to $BASE/"; exit 1; }
python -c "import rasterio, numpy" 2>/dev/null || { echo "FAIL: rasterio/numpy missing in the env"; exit 1; }
CNT="2yr 5yr 10yr 25yr 50yr 100yr ev1 ev2 ev3 ev4 ev5 ev6"; CTY="5yr 10yr 25yr 50yr 100yr"
TASKS=()
add() {  # domain srcdir dstdir storms
  for s in $4; do
    if [[ -n "$ONLY" && " $ONLY " != *" $s "* ]]; then continue; fi
    n=$(ls "$2/$s"/depth_*.tif 2>/dev/null | wc -l)
    if [[ "$n" -lt 24 ]]; then echo "SKIP $1 $s: only $n depth files in $2/$s"; else echo "ok   $1 $s: $n depth files"; TASKS+=("$2/$s|$3/$s"); fi
  done
}
if [[ "$DOM" == county || "$DOM" == both ]]; then add county "$BASE/runs/county_v2" "$BASE/county_cog" "$CNT"; fi
if [[ "$DOM" == city || "$DOM" == both ]]; then add city "$BASE/runs/output_v2" "$BASE/city_cog" "$CTY"; fi
[[ ${#TASKS[@]} -gt 0 ]] || { echo "nothing to do"; exit 1; }
if [[ $HERE -eq 1 ]]; then
  for t in "${TASKS[@]}"; do python "$BASE/make_cogs.py" --src "${t%|*}" --dst "${t#*|}" --res-m "$RES"; done
  exit 0
fi
mkdir -p "$BASE/logs"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=bgcog"
  echo "#SBATCH --output=$BASE/logs/cog_%A_%a.out"
  echo "#SBATCH --partition=$PART"
  echo "#SBATCH --nodes=1"
  echo "#SBATCH --ntasks=1"
  echo "#SBATCH --cpus-per-task=8"
  echo "#SBATCH --time=06:00:00"
  echo "T=(${TASKS[*]})"
  echo 'P=${T[$SLURM_ARRAY_TASK_ID]}'
  echo "source $SCRATCH/miniforge3/bin/activate"
  echo "python $BASE/make_cogs.py --src \"\${P%|*}\" --dst \"\${P#*|}\" --res-m $RES"
} > "$BASE/cog_job.sh"
echo "Will convert ${#TASKS[@]} storm folders (res $RES m)."
if [[ $DRY -eq 1 ]]; then echo "DRY RUN: wrote $BASE/cog_job.sh, submitted nothing"; exit 0; fi
sbatch --array=0-$(( ${#TASKS[@]} - 1 )) "$BASE/cog_job.sh"
echo "Watch: squeue -u \$USER ; output $BASE/county_cog/<storm>/   (du -sh $BASE/county_cog/*)"
