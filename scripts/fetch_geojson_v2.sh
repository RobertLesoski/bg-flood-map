#!/bin/bash
# Run in your regular Ubuntu terminal (NOT on Vista), after the Vista conversion jobs finish.
# Downloads the converted GeoJSON (UTM) plus run_summary.json for each storm. Much smaller than the rasters.
# Safe to re-run: it resumes and skips files that are already complete.
set -u
HOST="rlesoski@vista.tacc.utexas.edu"
SRC_GJ="/scratch/11931/rlesoski/bgflood/geojson_v2/"
SRC_RUN="/scratch/11931/rlesoski/bgflood/runs/output_v2/"
DEST="/mnt/c/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/BG FLOOD MAP VISTA SUPERCOMPUTER/v2"

mkdir -p "$DEST" || { echo "Cannot create $DEST"; exit 1; }
echo "Free space on C: $(df -h /mnt/c | awk 'NR==2{print $4}')"
if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is not installed here. Run:  sudo apt install -y rsync   then re-run this script."
  exit 1
fi
echo "Downloading GeoJSON from $HOST ..."
rsync -av --partial --progress --prune-empty-dirs --include='*/' --include='*.geojson' --exclude='*' "$HOST:$SRC_GJ" "$DEST/"
rc1=$?
echo "Downloading run summaries ..."
rsync -av --partial --prune-empty-dirs --include='*/' --include='run_summary.json' --exclude='*' "$HOST:$SRC_RUN" "$DEST/"
rc2=$?
echo
if [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]; then
  echo "Download finished. Folders now in: $DEST"
  ls -1 "$DEST"
  echo
  echo "Next, in PowerShell:   & \"\$env:USERPROFILE\\anaconda3\\python.exe\" \"C:\\Users\\Minec\\Documents\\BG_3D_Map\\scripts\\reproject_hpc.py\" v2_all"
else
  echo "rsync stopped (codes $rc1 / $rc2). Just run this script again; it resumes."
fi
exit $((rc1 + rc2))
