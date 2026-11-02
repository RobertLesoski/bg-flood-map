#!/bin/bash
# Run in your regular Ubuntu terminal (NOT on Vista). One password + token prompt.
# Downloads the solver-v2 results (25 hourly rasters + max_depth.tif + run_summary.json per storm, ~10 GB each).
# Safe to re-run: it resumes where it stopped and skips files that are already complete.
set -u
HOST="rlesoski@vista.tacc.utexas.edu"
SRC="/scratch/11931/rlesoski/bgflood/runs/output_v2/"
DEST="/mnt/c/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/BG FLOOD MAP VISTA SUPERCOMPUTER/v2"

mkdir -p "$DEST" || { echo "Cannot create $DEST"; exit 1; }
echo "Free space on C: $(df -h /mnt/c | awk 'NR==2{print $4}')  (needs about 50 GB for all five storms)"
if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is not installed here. Run:  sudo apt install -y rsync   then re-run this script."
  exit 1
fi
echo "Downloading from $HOST ..."
rsync -av --partial --progress --prune-empty-dirs \
  --exclude='depth_0000??s.tif' \
  --include='*/' --include='depth_*.tif' --include='max_depth.tif' --include='run_summary.json' --exclude='*' \
  "$HOST:$SRC" "$DEST/"
rc=$?
echo
if [ $rc -eq 0 ]; then
  echo "Download finished. Folders now in: $DEST"
  ls -1 "$DEST"
  echo
  echo "Next, in PowerShell:   & \"\$env:USERPROFILE\\anaconda3\\python.exe\" \"C:\\Users\\Minec\\Documents\\BG_3D_Map\\scripts\\process_v2.py\""
else
  echo "rsync stopped with code $rc. Just run this script again; it resumes."
fi
exit $rc
