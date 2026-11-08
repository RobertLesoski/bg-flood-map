#!/bin/bash
#SBATCH --job-name=warrenv2
#SBATCH --output=/scratch/11931/rlesoski/bgflood/logs/county_%A_%a.out
#SBATCH --error=/scratch/11931/rlesoski/bgflood/logs/county_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com

# Whole of Warren County, 12 storms, one H100 per storm.
#   sbatch submit_county_all.sh                      all 12
#   sbatch --array=3 submit_county_all.sh            only task 3 (25yr)
#   sbatch --array=0-11%4 submit_county_all.sh       all 12, at most 4 at a time
# Task numbers: 0=2yr 1=5yr 2=10yr 3=25yr 4=50yr 5=100yr 6=ev1 7=ev2 8=ev3 9=ev4 10=ev5 11=ev6
#
# Before submitting, once:  bash run_county_all.sh --dry-run   (builds nothing, checks the rain files)
#                           bash build_county_grid.sh          (writes dem/dem_warren_5m.tif and mannings_n_county.tif)
# or simply run `bash run_county_all.sh`, which does both and submits for you.

STORM_LIST=(2yr 5yr 10yr 25yr 50yr 100yr ev1 ev2 ev3 ev4 ev5 ev6)
ST=${STORM_LIST[$SLURM_ARRAY_TASK_ID]}

export PYTHONUNBUFFERED=1
export CN_PERVIOUS=75
export CN_IMPERVIOUS=98
export OUTFLOW=1
export OUTFLOW_SLOPE=0.005
source $SCRATCH/miniforge3/bin/activate
cd $SCRATCH/bgflood || exit 1

for f in flood_solver_v2.py dem/dem_warren_5m.tif dem/mannings_n_county.tif runs/storm_${ST}.rain; do
  [[ -s $f ]] || { echo "FATAL: $f missing - run run_county_all.sh first (it builds it)"; exit 1; }
done

echo "[$(date)] county ${ST} (solver v2, 5 m) on $(hostname)"
python3 flood_solver_v2.py $ST \
    --dem dem/dem_warren_5m.tif \
    --mann dem/mannings_n_county.tif \
    --outdir runs/county_v2/$ST
rc=$?
echo "[$(date)] county ${ST} finished with exit code $rc"
exit $rc
