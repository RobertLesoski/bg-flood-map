#!/bin/bash
#SBATCH --job-name=bgflood_hist
#SBATCH --output=/scratch/11931/rlesoski/bgflood/logs/hist_%A_%a.out
#SBATCH --error=/scratch/11931/rlesoski/bgflood/logs/hist_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=robert.lesoski138@topper.wku.edu

# Six historical 24-h rainfall blocks, light -> extreme (see storms.json)
EV_LIST=(ev1 ev2 ev3 ev4 ev5 ev6)
EV=${EV_LIST[$SLURM_ARRAY_TASK_ID]}

source $SCRATCH/miniforge3/bin/activate
cd $SCRATCH/bgflood

python3 vista_preflight.py || { echo "Preflight failed - aborting $EV"; exit 1; }

echo "[$(date)] Starting $EV on $(hostname)"
python3 flood_solver.py $EV
echo "[$(date)] $EV done"
