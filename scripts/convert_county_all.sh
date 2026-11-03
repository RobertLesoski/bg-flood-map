#!/bin/bash
# =============================================================================
#  Convert the county solver rasters (runs/county_v2/<storm>/depth_*.tif) into viewer flow grids
#  ($SCRATCH/bgflood/county_flow/<storm>/{meta.json,hNN.rgba.gz,flowpaths.rgba.gz}), CPU only.
#
#  Needs in $SCRATCH/bgflood (copy once from the PC):
#       build_flow_from_rasters.py  flowgrid.py  convert_county_all.sh
#       dem/dem_warren_5m.tif  dem/mannings_n_county.tif        (written by run_county_all.sh)
#  and python3 with numpy + rasterio (the miniforge env of the solver runs; no scipy, no GPU).
#
#  Usage (login node, runs the storms one after the other in this shell - fine for one or two; use sbatch for all):
#       bash convert_county_all.sh                  all 12 storms (resumable: finished storms are skipped in seconds)
#       bash convert_county_all.sh 100yr ev3        just these
#       WORKERS=8 CELL_M=25 bash convert_county_all.sh 100yr     (hours converted in parallel; CELL_M in mercator metres)
#       FORCE=1 bash convert_county_all.sh 100yr    rebuild even if already built from the same rasters
#
#  As a SLURM job (one storm per array task; about 0.5 GB per worker, ~10 min per storm with 8 workers):
#       sbatch convert_county_all.sh                (the #SBATCH lines below are active; 12 tasks, run as the array)
#       sbatch --array=5 convert_county_all.sh      only task 5 = 100yr (order: 2yr 5yr 10yr 25yr 50yr 100yr ev1..ev6)
#  Check the partition with `sinfo`: "gg" = Vista's CPU-only Grace nodes. Use --partition=gh if gg is not available to you.
#
#SBATCH --job-name=countyflow
#SBATCH --output=/scratch/11931/rlesoski/bgflood/logs/countyflow_%A_%a.out
#SBATCH --error=/scratch/11931/rlesoski/bgflood/logs/countyflow_%A_%a.err
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#
#  Exit code: 0 when every requested storm is built (or was already), 1 when one failed or its rasters are incomplete.
# =============================================================================
set -u
BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
cd "$BASE" || { echo "FATAL: $BASE not found"; exit 1; }
mkdir -p logs county_flow
[[ -f "$SCRATCH/miniforge3/bin/activate" ]] && source "$SCRATCH/miniforge3/bin/activate"
command -v python3 >/dev/null || { echo "FATAL: python3 not found"; exit 1; }

ALL=(2yr 5yr 10yr 25yr 50yr 100yr ev1 ev2 ev3 ev4 ev5 ev6)
if [[ $# -gt 0 ]]; then
  STORMS=("$@")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  STORMS=("${ALL[$SLURM_ARRAY_TASK_ID]}")
else
  STORMS=("${ALL[@]}")
fi
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
SCRIPT="$BASE/build_flow_from_rasters.py"
[[ -f "$SCRIPT" && -f "$BASE/flowgrid.py" ]] || { echo "FATAL: copy build_flow_from_rasters.py and flowgrid.py to $BASE"; exit 1; }
DEM="$BASE/dem/dem_warren_5m.tif"; MANN="$BASE/dem/mannings_n_county.tif"
[[ -f "$DEM" ]] || { echo "FATAL: $DEM missing (run_county_all.sh builds it)"; exit 1; }
EXTRA=()
[[ -f "$MANN" ]] && EXTRA+=(--mann "$MANN") || echo "WARN: $MANN missing, using n = 0.035"
[[ -n "${CELL_M:-}" ]] && EXTRA+=(--cell-m "$CELL_M")
[[ -n "${FORCE:-}" ]] && EXTRA+=(--force)

rc_all=0
for ST in "${STORMS[@]}"; do
  RUN="$BASE/runs/county_v2/$ST"
  OUT="$BASE/county_flow/$ST"
  if [[ ! -d "$RUN" ]]; then echo "[$ST] SKIP: $RUN not found"; rc_all=1; continue; fi
  N=$(ls "$RUN"/depth_*s.tif 2>/dev/null | wc -l)
  if [[ $N -lt 24 || ! -f "$RUN/max_depth.tif" || ! -f "$RUN/run_summary.json" ]]; then
    echo "[$ST] SKIP: solver run not finished ($N depth rasters, need 24 hours + max_depth.tif + run_summary.json)"; rc_all=1; continue
  fi
  case "$ST" in
    ev*) LABEL="Historical event ${ST#ev} (county)" ;;
    *)   LABEL="${ST%yr}-year storm (county)" ;;
  esac
  echo "[$ST] $(date +%T) converting $N rasters -> $OUT  (workers=$WORKERS)"
  python3 "$SCRIPT" --run "$RUN" --out "$OUT" --storm "$ST" --label "$LABEL" \
      --dem "$DEM" --dem-units metres --workers "$WORKERS" "${EXTRA[@]}"
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "[$ST] FAILED (exit $rc); finished hours are kept for a resume: just run it again"; rc_all=1
  else echo "[$ST] $(date +%T) ok: $(du -sh "$OUT" | cut -f1)"; fi
done
exit $rc_all
