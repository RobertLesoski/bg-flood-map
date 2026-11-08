#!/bin/bash
# RUN ON VISTA login node (script only - nothing here has been run):
#   bash vista_climate_scenarios.sh --dry-run      check everything, write the job file, submit nothing
#   bash vista_climate_scenarios.sh                write the 4 scaled rain files, submit 4 county runs (one H100 each)
#   bash vista_climate_scenarios.sh --only "100yr_p10"   (names: 100yr_p10 100yr_p20 25yr_p10 25yr_p20)
#
# SCENARIO MULTIPLIERS, NOT A CLIMATE PROJECTION: the existing 100yr and 25yr design rain files are multiplied by 1.10 and 1.20
# (rain rate x factor at every time step, same hyetograph shape). Nothing else changes - same terrain, same curve numbers, same solver.
# It says "what if the same storm dropped 10 % / 20 % more rain", and nothing about future frequency, intensity or land use.
# Same setup as run_county_all.sh (which must have been run once: DEM + roughness + solver in $SCRATCH/bgflood).
# Output: $SCRATCH/bgflood/runs/county_climate/<name>/  (depth_*.tif, max_depth.tif, run_summary.json)
set -u
DRY=0; ONLY=""; THROTTLE=""
while [[ $# -gt 0 ]]; do case "$1" in
  --dry-run) DRY=1 ;; --only) shift; ONLY="${1:-}" ;; --throttle) shift; THROTTLE="${1:-}" ;;
  -h|--help) sed -n '2,15p' "$0"; exit 0 ;; *) echo "unknown option: $1"; exit 2 ;; esac; shift; done
BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
cd "$BASE" || { echo "FATAL: $BASE not found"; exit 1; }
mkdir -p runs/county_climate logs
fail() { echo; echo "FATAL: $*"; exit 1; }
ALL=(100yr_p10 100yr_p20 25yr_p10 25yr_p20)
if [[ -n "$ONLY" ]]; then read -r -a NAMES <<< "$ONLY"; for s in "${NAMES[@]}"; do [[ " ${ALL[*]} " == *" $s "* ]] || fail "unknown scenario '$s'"; done; else NAMES=("${ALL[@]}"); fi
[[ -f "$SCRATCH/miniforge3/bin/activate" ]] || fail "miniforge missing"
source "$SCRATCH/miniforge3/bin/activate"
[[ -f flood_solver_v2.py ]] || fail "flood_solver_v2.py missing (run run_county_all.sh once first)"
for f in dem/dem_warren_5m.tif dem/mannings_n_county.tif runs/storm_100yr.rain runs/storm_25yr.rain; do [[ -s $f ]] || fail "$f missing (run run_county_all.sh first)"; done
if command -v squeue >/dev/null 2>&1 && [[ -n "$(squeue -u "${USER:-$(whoami)}" -h -n warrenclim 2>/dev/null)" ]]; then fail "warrenclim jobs already queued (scancel -n warrenclim)"; fi

echo "== 1/3 Scaled rain files (rate x factor; rain files are m/s) =="
python3 - "${NAMES[@]}" <<'PYEOF' || fail "rain scaling failed"
import os, sys
B = os.path.join(os.environ['SCRATCH'], 'bgflood', 'runs')
def read(p):
    with open(p) as f:
        n = int(f.readline()); return [tuple(map(float, f.readline().split())) for _ in range(n)]
def mm(pts): return sum((pts[i+1][0]-pts[i][0]) * (pts[i+1][1]+pts[i][1]) / 2 for i in range(len(pts)-1)) * 1000
bad = 0
for name in sys.argv[1:]:
    base, tag = name.split('_'); fac = 1.0 + int(tag[1:]) / 100.0
    pts = read(os.path.join(B, f'storm_{base}.rain'))
    out = [(t, r * fac) for t, r in pts]
    with open(os.path.join(B, f'storm_{name}.rain'), 'w') as fh:
        fh.write(f'{len(out)}\n')
        for t, r in out: fh.write(f'{t:.1f} {r:.12e}\n')
    a, b = mm(pts), mm(read(os.path.join(B, f'storm_{name}.rain')))
    peak = max(r for _, r in out) * 3.6e6
    ok = abs(b / a - fac) < 0.003 and 0 < peak < 600
    print(f"  {'ok  ' if ok else 'FAIL'} {name:>9}: {a:7.2f} mm -> {b:7.2f} mm (x{b/a:.3f}), peak {peak:5.1f} mm/hr")
    bad += (not ok)
sys.exit(1 if bad else 0)
PYEOF

echo; echo "== 2/3 SLURM array (${#NAMES[@]} runs: ${NAMES[*]}) =="
LIST="${NAMES[*]}"; ARRAY="0-$(( ${#NAMES[@]} - 1 ))"; [[ -n "$THROTTLE" ]] && ARRAY="${ARRAY}%${THROTTLE}"
cat > submit_climate.template <<'SLEOF'
#!/bin/bash
#SBATCH --job-name=warrenclim
#SBATCH --output=__BASE__/logs/climate_%A_%a.out
#SBATCH --error=__BASE__/logs/climate_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=__TIME__
#SBATCH --array=__ARRAY__
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com
NAME_LIST=(__LIST__)
ST=${NAME_LIST[$SLURM_ARRAY_TASK_ID]}
export PYTHONUNBUFFERED=1 CN_PERVIOUS=75 CN_IMPERVIOUS=98 OUTFLOW=1 OUTFLOW_SLOPE=0.005
source $SCRATCH/miniforge3/bin/activate
cd $SCRATCH/bgflood
echo "[$(date)] county scenario ${ST} (rain scaled, solver v2, 5 m)"
python3 flood_solver_v2.py $ST --dem dem/dem_warren_5m.tif --mann dem/mannings_n_county.tif --outdir runs/county_climate/$ST
rc=$?
echo "[$(date)] ${ST} exit code $rc"
exit $rc
SLEOF
mk() { sed -e "s|__BASE__|$BASE|g" -e "s|__TIME__|$1|" -e "s|__ARRAY__|$ARRAY|" -e "s|__LIST__|$LIST|" submit_climate.template > submit_climate.sh; }
if [[ $DRY -eq 1 ]]; then mk 24:00:00; echo "  DRY RUN - wrote submit_climate.sh, submitted nothing"; exit 0; fi
ok=0
for T in 24:00:00 20:00:00 12:00:00; do
  mk "$T"
  if OUT=$(sbatch submit_climate.sh 2>&1); then echo "$OUT" | tail -2; echo "  submitted with --time=$T"; ok=1; break; else echo "  rejected --time=$T: $(echo "$OUT" | tail -1)"; fi
done
[[ $ok -eq 1 ]] || fail "sbatch rejected every time limit"
echo; echo "== 3/3 Watching =="
echo "Queue: squeue -u ${USER:-$(whoami)}   Output: $BASE/runs/county_climate/<name>/   Cancel: scancel -n warrenclim"
