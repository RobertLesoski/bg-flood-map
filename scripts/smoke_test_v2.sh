#!/bin/bash
# =============================================================================
#  Real-terrain smoke test of solver v2 (about 15-40 min, on the dev queue). Run on a Vista LOGIN node:
#      bash smoke_test_v2.sh
#  1. shows what happened to the earlier jobs
#  2. runs 1 simulated hour on the REAL 2 m terrain with a test rain (100 mm/hr for 30 min, then dry)
#  3. checks the result automatically: finite values, exact water balance, sane flooded area and
#     infiltration share, water leaving the domain, and a rough projection of the full-run time
#  Nothing existing is touched. Output goes to runs/smoke_v2/<jobid>/.
# =============================================================================
set -u
BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
cd "$BASE" || { echo "FATAL: $BASE not found"; exit 1; }
mkdir -p runs logs
[[ -f flood_solver_v2.py ]] || { echo "FATAL: flood_solver_v2.py missing - run run_5_storms_v2.sh --dry-run first"; exit 1; }

echo "== 1/3  What happened to the earlier jobs =="
sacct -u "${USER:-$(whoami)}" -S now-5days --format=JobID%16,JobName%12,State%14,ExitCode,Elapsed -X 2>/dev/null | grep -E "JobID|---|bgflood" || echo "  (no matching jobs found in the last 5 days)"

echo; echo "== 2/3  Writing the test rain and the test job =="
python3 - <<'PYEOF'
import os
r = 100.0 / 3600 / 1000                      # 100 mm/hr in m/s
p = os.path.join(os.environ['SCRATCH'], 'bgflood', 'runs', 'storm_smoke.rain')
open(p, 'w').write(f"4\n0 {r:.9e}\n1800 {r:.9e}\n1801 0\n86400 0\n")
print('  wrote', p, '(50 mm in 30 min)')
PYEOF

cat > smoke_check.py <<'PYEOF'
import glob, json, math, os, sys
import numpy as np
import rasterio
B = os.path.join(os.environ['SCRATCH'], 'bgflood')
d = sorted(glob.glob(os.path.join(B, 'runs', 'smoke_v2', '*')), key=os.path.getmtime)[-1]
s = json.load(open(os.path.join(d, 'run_summary.json')))
fb = s['final_mass_balance']
rasters = sorted(glob.glob(os.path.join(d, 'depth_*.tif')))
last = rasterio.open(rasters[-1]).read(1)
v = last[last != -9999]
wet = (v > 0.1).mean() * 100
rain, runoff = s['total_rain_mm'], s['runoff_depth_mm']
infil = 100 * (1 - runoff / rain)
out_pct = 100 * fb['outflow_m3'] / max(fb['runoff_m3'], 1)
proj_h = s['wall_minutes'] / 60 * (86400 / 3600)
res = []
def chk(name, ok, detail):
    res.append(ok); print(('  PASS  ' if ok else '  FAIL  ') + name + '   ' + detail)
chk('rasters written', len(rasters) >= 1, f'{len(rasters)} files')
chk('no NaN / infinite depths', bool(np.isfinite(v).all()), f'max depth {v.max():.2f} m')
chk('depths below 25 m (no blow-up)', v.max() < 25, f'{v.max():.2f} m')
chk('water balance closes (|error| < 0.05 %)', abs(fb['mass_error_pct']) < 0.05, f"{fb['mass_error_pct']:+.5f} %")
chk('runoff is between 0 and the rain', 0 < runoff < rain, f'rain {rain:.1f} mm -> runoff {runoff:.1f} mm')
chk('infiltration share plausible (20-90 %)', 20 < infil < 90, f'{infil:.1f} % of the rain soaked in')
chk('some water but not the whole domain is flooded', 0 < wet < 40, f'{wet:.1f} % of cells deeper than 0.1 m')
chk('water leaves through the boundary', fb['outflow_m3'] > 0, f"{out_pct:.1f} % of runoff left the domain in this short test")
print(f"\n  rough projection of a full 24 h run: {proj_h:.1f} h wall (early time steps are the largest, so the real run will take longer; limit is 40 h)")
print('\n  SMOKE TEST ' + ('PASSED' if all(res) else 'FAILED'))
sys.exit(0 if all(res) else 1)
PYEOF

cat > smoke_job.template <<'SLEOF'
#!/bin/bash
#SBATCH --job-name=bgsmoke
#SBATCH --output=__BASE__/logs/smoke_%j.out
#SBATCH --error=__BASE__/logs/smoke_%j.err
#SBATCH --partition=__PART__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
export PYTHONUNBUFFERED=1 CN_PERVIOUS=75 CN_IMPERVIOUS=98 OUTFLOW=1 OUTFLOW_SLOPE=0.005
source $SCRATCH/miniforge3/bin/activate
cd $SCRATCH/bgflood
python3 -c "import flood_solver_v2 as s; s.run_storm('smoke', total_time=3600, out_dir='runs/smoke_v2/'+__import__('os').environ.get('SLURM_JOB_ID','local'))" && python3 smoke_check.py
SLEOF

echo; echo "== 3/3  Running the smoke test (waits for the job; Ctrl+C is safe, the job keeps running) =="
for PART in gh-dev gh; do
  sed -e "s|__BASE__|$BASE|g" -e "s|__PART__|$PART|g" smoke_job.template > smoke_job.sh
  OUT=$(sbatch --wait smoke_job.sh 2>&1); rc=$?
  if echo "$OUT" | grep -q "sbatch: error"; then            # submission rejected: try the next partition
    echo "  partition $PART rejected the job: $(echo "$OUT" | grep 'sbatch: error' | tail -1)"
    [[ "$PART" == "gh" ]] && echo "  (no partition accepted the test job)"
    continue
  fi
  echo "  job ran on partition $PART (sbatch exit code $rc)"; break
done
LOG=$(ls -t logs/smoke_*.out 2>/dev/null | head -1)
echo; echo "== Result (from $LOG) =="
tail -n 25 "$LOG" 2>/dev/null || echo "  no log found"
if grep -q "SMOKE TEST PASSED" "$LOG" 2>/dev/null; then
  echo; echo ">>> SMOKE TEST PASSED - safe to submit the five full runs."; exit 0
else
  echo; echo ">>> SMOKE TEST DID NOT PASS - do NOT submit the full runs. Paste the output above to Claude."; exit 1
fi
