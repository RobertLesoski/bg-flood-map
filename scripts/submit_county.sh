#!/bin/bash
#SBATCH --job-name=warren_flood
#SBATCH --output=/scratch/11931/rlesoski/bgflood/logs/county_%A_%a.out
#SBATCH --error=/scratch/11931/rlesoski/bgflood/logs/county_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com

# County-wide Warren County flood simulation — same 6 historical events as BG batch,
# run on a 5 m county DEM instead of the 2 m city DEM.
# Output: $SCRATCH/bgflood/runs/county_output/<ev>/
#
# Submit after dem_county_clip.py has produced dem/dem_warren_5m.tif on Vista:
#   sbatch submit_county.sh
#
# Reproject locally after download:
#   python scripts/reproject_hpc.py county_ev1   (or "county_all")

EV_LIST=(ev1 ev2 ev3 ev4 ev5 ev6)
EV=${EV_LIST[$SLURM_ARRAY_TASK_ID]}

source $SCRATCH/miniforge3/bin/activate
cd $SCRATCH/bgflood

python3 vista_county_preflight.py || { echo "Preflight failed - aborting county $EV"; exit 1; }

echo "[$(date)] Starting county $EV on $(hostname)"

# Run solver pointing at county DEM and a county-namespaced output dir
python3 flood_solver.py "county_${EV}" \
    --dem dem/dem_warren_5m.tif \
    --outdir runs/county_output

echo "[$(date)] county $EV done"
