#!/bin/bash
# =============================================================================
#  Bowling Green flood model - the 5 return-period storms (5, 10, 25, 50, 100 yr)
#  One script, run on a Vista LOGIN node:      bash run_5_storms.sh
#  Dry run (do everything except submit):      bash run_5_storms.sh --dry-run
#
#  It (1) writes correct rain files, (2) patches flood_solver.py, (3) moves any old/bad
#  output aside, (4) runs a preflight, (5) submits ONE 5-task SLURM array.
#  Nothing is deleted. The 2yr run is not touched.
# =============================================================================
set -u
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
STAMP=$(date +%Y%m%d_%H%M%S)
cd "$BASE" || { echo "FATAL: $BASE not found"; exit 1; }
mkdir -p runs logs
fail() { echo; echo "FATAL: $*"; exit 1; }

[[ -f flood_solver.py ]]                      || fail "flood_solver.py missing in $BASE"
[[ -f dem/dem_bg_2m.tif && -f dem/mannings_n.tif ]] || fail "dem/dem_bg_2m.tif or dem/mannings_n.tif missing"
[[ -f "$SCRATCH/miniforge3/bin/activate" ]]   || fail "$SCRATCH/miniforge3/bin/activate missing"
command -v python3 >/dev/null                 || fail "python3 not found"
if command -v squeue >/dev/null 2>&1 && [[ -n "$(squeue -u "${USER:-$(whoami)}" -h -n bgflood5 2>/dev/null)" ]]; then
  fail "bgflood5 jobs are already queued or running (squeue -u \$USER). Re-running now could move live output. Cancel them first (scancel -n bgflood5) if you really want to restart."
fi

# -----------------------------------------------------------------------------
echo "== 1/5  Writing rain files (2yr-run hyetograph, NOAA Atlas 14 depths, units = m/s) =="
cat > make_rain.py <<'PYEOF'
import os, sys
BASE = os.path.join(os.environ['SCRATCH'], 'bgflood')
STORMS = {'5yr': 4.2, '10yr': 4.9, '25yr': 5.9, '50yr': 6.7, '100yr': 7.6}   # 24-h depth, inches (NOAA Atlas 14, Warren Co. KY)
# Same temporal distribution as the completed 2yr run: 11-point cumulative curve, linear in time (fraction of 24 h -> fraction of depth).
# Verified against the 2yr rasters: mean depth over the domain was 30.3 mm (h10) and 37.2 mm (h11); this curve predicts 30.4 and 37.4 mm.
# (The official NRCS Type II table would have given 16.1 and 20.9 mm, i.e. not what the 2yr run used.)
_TF = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
_PF = [0, .04, .10, .19, .31, .50, .70, .79, .86, .92, 1.0]
TYPE_II = [(round(f * 24, 6), p) for f, p in zip(_TF, _PF)]
assert all(b[1] >= a[1] and b[0] > a[0] for a, b in zip(TYPE_II, TYPE_II[1:])), 'table not monotone'
assert TYPE_II[-1] == (24, 1.000)

def cum(h):
    for (h0, c0), (h1, c1) in zip(TYPE_II, TYPE_II[1:]):
        if h0 <= h <= h1:
            return c0 + (c1 - c0) * (h - h0) / (h1 - h0)
    raise ValueError(h)

STEP = 300.0                      # 5-minute intensity steps
N = int(86400 / STEP)
ok = True
for name, depth_in in STORMS.items():
    P = depth_in * 0.0254         # metres
    pts = []
    for k in range(N):
        t0, t1 = k * STEP, (k + 1) * STEP
        rate = (cum(t1 / 3600.0) - cum(t0 / 3600.0)) * P / STEP        # m/s, constant over the step
        pts.append((t0, rate))
        pts.append((t1 - 1.0, rate))
    pts.append((86400.0, 0.0))
    path = os.path.join(BASE, 'runs', f'storm_{name}.rain')
    with open(path, 'w') as fh:
        fh.write(f'{len(pts)}\n')
        for t, r in pts:
            fh.write(f'{t:.1f} {r:.12e}\n')
    # read back exactly as the solver does (m/s) and verify
    with open(path) as fh:
        n = int(fh.readline())
        rd = [tuple(map(float, fh.readline().split())) for _ in range(n)]
    total_mm = sum((rd[i + 1][0] - rd[i][0]) * (rd[i + 1][1] + rd[i][1]) / 2 for i in range(n - 1)) * 1000
    exp_mm = depth_in * 25.4
    peak = max(r for _, r in rd) * 3.6e6
    good = abs(total_mm - exp_mm) / exp_mm < 0.003 and rd[-1][0] == 86400.0 and 0 < peak < 600 and n == len(pts)
    ok &= good
    print(f"  {name:>5}: {total_mm:7.2f} mm (target {exp_mm:6.2f})  5-min peak {peak:6.1f} mm/hr   {'OK' if good else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
python3 make_rain.py || fail "rain file verification failed"

# -----------------------------------------------------------------------------
echo; echo "== 2/5  Patching flood_solver.py =="
cat > patch_solver5.py <<'PYEOF'
import os, re, shutil, sys, time
PATH = os.path.join(os.environ['SCRATCH'], 'bgflood', 'flood_solver.py')
src = open(PATH, encoding='utf-8').read()
log = []

def edit(name, pat, repl, done=None, required=True):
    global src
    if done and re.search(done, src):
        log.append(f'  already  {name}'); return
    new, n = re.subn(pat, repl, src)
    if n == 0:
        if required:
            sys.exit(f'ABORT: cannot apply "{name}" and it is not already applied - solver left unchanged.')
        log.append(f'  n/a      {name}'); return
    src = new; log.append(f'  applied  {name}')

# debugging workarounds that hid the earlier 1000x rain bug
edit('remove 0.5 s dt floor', r'dt = max\(min\((.*)\), 0\.5\)', r'dt = min(\1)', required=False)
edit('remove 20 m depth cap', r'cp\.minimum\((cp\.maximum\(H_d \+ dt \* \(r_rate - dQx/dx - dQy/dy\), 0\.0\)), 20\.0\)', r'\1', required=False)
# scenario names ("5yr") instead of integers
edit('rain path storm_<name>.rain', r"f'storm_\{rp\}yr\.rain'", "f'storm_{rp}.rain'", done=r"f'storm_\{rp\}\.rain'")
edit('output dir runs/output/<name>', r"f'\{rp\}yr'", "f'{rp}'", done=r"'output',\s*f'\{rp\}'")
edit('scenario name from argv', r'rp = int\(sys\.argv\[1\]\)', 'rp = sys.argv[1]', done=r'rp = sys\.argv\[1\]')
edit('log label start', r'=== \{rp\}-yr storm ===', '=== {rp} storm ===', required=False)
edit('log label done', r'Done \{rp\}-yr', 'Done {rp}', required=False)
edit('24 h run length', r'(def run_storm\(rp, total_time=)\d+', r'\g<1>86400', done=r'def run_storm\(rp, total_time=86400')
# unit guard: abort on implausible rain intensity (the 1000x error was ~17,000 mm/hr; design-storm 5-min peak is < 300)
guard = ('    peak_mmhr = rain_r.max()*3600*1000\n'
         '    total_mm = float(((rain_r[1:]+rain_r[:-1])*0.5*np.diff(rain_t)).sum())*1000\n'
         '    print(f"Total rain: {total_mm:.1f} mm ({total_mm/25.4:.2f} in)", flush=True)\n'
         '    if peak_mmhr > 600:\n'
         '        print(f"ABORT: peak rain {peak_mmhr:.0f} mm/hr is implausible - rain file units wrong (solver expects m/s)", flush=True)\n'
         '        sys.exit(2)\n')
if 'ABORT: peak rain' in src:
    if re.search(r'peak_mmhr > 250', src):
        src = src.replace('peak_mmhr > 250', 'peak_mmhr > 600'); log.append('  applied  unit guard threshold 250 -> 600 mm/hr')
    else:
        log.append('  already  unit guard')
else:
    edit('unit guard', r'(\n    print\(f"Peak rain: [^\n]*\n)', lambda m: m.group(1) + guard)
# always save the final (24 h) state
final = ('    if not os.path.exists(os.path.join(out_dir, f\'depth_{int(t):06d}s.tif\')):\n'
         '        H_np = cp.asnumpy(H_d); H_np[~valid] = -9999.0\n'
         '        with rasterio.open(os.path.join(out_dir, f\'depth_{int(t):06d}s.tif\'), \'w\', **profile) as dst:\n'
         '            dst.write(H_np, 1)\n'
         '        print(f"  final state saved at t={t/3600:.2f}hr", flush=True)\n\n')
if 'final state saved' in src:
    log.append('  already  final-state save')
else:
    edit('final-state save', r'(\n)(    files = sorted\(\[f for f in os\.listdir\(out_dir\))', lambda m: m.group(1) + final + m.group(2))

# structural sanity: the loop must advance time (an earlier edit had silently dropped this)
for what, pat in [('time loop', r'while t < total_time'), ('t += dt', r'\bt \+= dt'), ('step += 1', r'\bstep \+= 1'),
                  ('rain read as m/s', r'rain_r\.append\(rr\)\s*(#.*)?\n')]:
    if not re.search(pat, src):
        sys.exit(f'ABORT: solver sanity check failed: "{what}" not found - solver left unchanged.')
if re.search(r'rain_r\.append\(rr\s*/', src):
    sys.exit('ABORT: solver divides rain by 1000 - unit convention differs from the rain files - solver left unchanged.')
compile(src, PATH, 'exec')
if '\n'.join(log).count('applied') == 0:
    print('\n'.join(log)); print('  solver already fully patched - no change'); sys.exit(0)
bak = PATH + '.bak_' + time.strftime('%Y%m%d_%H%M%S')
shutil.copy2(PATH, bak)
open(PATH, 'w', encoding='utf-8').write(src)
print('\n'.join(log)); print('  backup:', bak)
PYEOF
python3 patch_solver5.py || fail "solver patch failed (solver unchanged) - paste the message above to Claude"

# -----------------------------------------------------------------------------
echo; echo "== 3/5  Moving any old/bad output aside (nothing deleted) =="
for rp in 5yr 10yr 25yr 50yr 100yr; do
  d="runs/output/$rp"
  if [[ -d "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null)" ]]; then
    mkdir -p runs/old_runs
    mv "$d" "runs/old_runs/${rp}_$STAMP" && echo "  moved $d -> runs/old_runs/${rp}_$STAMP"
  else
    echo "  $rp: clean"
  fi
done

# -----------------------------------------------------------------------------
echo; echo "== 4/5  Preflight =="
python3 - <<'PYEOF' || fail "preflight failed"
import os, re, sys, py_compile
B = os.path.join(os.environ['SCRATCH'], 'bgflood')
bad = []
def chk(label, cond):
    print(('  ok    ' if cond else '  FAIL  ') + label)
    if not cond: bad.append(label)
s = open(os.path.join(B, 'flood_solver.py'), encoding='utf-8').read()
try:
    py_compile.compile(os.path.join(B, 'flood_solver.py'), doraise=True); chk('solver compiles', True)
except Exception as e:
    chk(f'solver compiles ({e})', False)
chk('solver: rain read as plain m/s', bool(re.search(r'rain_r\.append\(rr\)', s)) and not re.search(r'rain_r\.append\(rr\s*/', s))
chk('solver: scenario rain path', "f'storm_{rp}.rain'" in s)
chk('solver: scenario output dir', bool(re.search(r"'output',\s*f'\{rp\}'", s)))
chk('solver: string scenario arg', 'rp = sys.argv[1]' in s)
chk('solver: 24 h default', 'total_time=86400' in s)
chk('solver: unit guard', 'ABORT: peak rain' in s)
chk('solver: final-state save', 'final state saved' in s)
chk('solver: no 20 m cap', not re.search(r'20\.0\) \* V_d', s))
chk('solver: no dt floor', not re.search(r'dt = max\(min\(', s))
chk('solver: time advances (t += dt, step += 1)', bool(re.search(r'\bt \+= dt', s)) and bool(re.search(r'\bstep \+= 1', s)))
for rp in ('5yr', '10yr', '25yr', '50yr', '100yr'):
    chk(f'rain file storm_{rp}.rain', os.path.exists(os.path.join(B, 'runs', f'storm_{rp}.rain')))
    d = os.path.join(B, 'runs', 'output', rp)
    chk(f'output/{rp} empty or absent', (not os.path.isdir(d)) or not os.listdir(d))
chk('dem_bg_2m.tif', os.path.exists(os.path.join(B, 'dem', 'dem_bg_2m.tif')))
chk('mannings_n.tif', os.path.exists(os.path.join(B, 'dem', 'mannings_n.tif')))
os.makedirs(os.path.join(B, 'runs', 'output'), exist_ok=True)
chk('output folder writable', os.access(os.path.join(B, 'runs', 'output'), os.W_OK))
print('\n  PREFLIGHT ' + ('FAILED' if bad else 'PASSED'))
sys.exit(1 if bad else 0)
PYEOF

# -----------------------------------------------------------------------------
echo; echo "== 5/5  SLURM array (5 tasks: 5yr 10yr 25yr 50yr 100yr, one H100 each) =="
cat > submit_5storms.template <<'SLEOF'
#!/bin/bash
#SBATCH --job-name=bgflood5
#SBATCH --output=__BASE__/logs/storm_%A_%a.out
#SBATCH --error=__BASE__/logs/storm_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=__TIME__
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=robert.lesoski138@topper.wku.edu

RP_LIST=(5yr 10yr 25yr 50yr 100yr)
RP=${RP_LIST[$SLURM_ARRAY_TASK_ID]}

export PYTHONUNBUFFERED=1
source $SCRATCH/miniforge3/bin/activate
cd $SCRATCH/bgflood

echo "[$(date)] Starting ${RP} storm on $(hostname)"
python3 flood_solver.py $RP
rc=$?
echo "[$(date)] ${RP} finished with exit code $rc"
exit $rc
SLEOF

if [[ $DRY -eq 1 ]]; then
  sed -e "s|__BASE__|$BASE|g" -e "s|__TIME__|36:00:00|" submit_5storms.template > submit_5storms.sh
  echo "  DRY RUN - wrote submit_5storms.sh but did not submit."
  echo "  Submit with:  sbatch $BASE/submit_5storms.sh"
  exit 0
fi

submitted=0
for T in 36:00:00 24:00:00 12:00:00; do
  sed -e "s|__BASE__|$BASE|g" -e "s|__TIME__|$T|" submit_5storms.template > submit_5storms.sh
  if OUT=$(sbatch submit_5storms.sh 2>&1); then
    echo "$OUT" | tail -2
    echo "  submitted with --time=$T"
    submitted=1; break
  else
    echo "  sbatch rejected --time=$T: $(echo "$OUT" | tail -1)"
  fi
done
[[ $submitted -eq 1 ]] || fail "sbatch rejected every time limit - paste the message above to Claude"

echo
echo "Check queue:      squeue -u ${USER:-$(whoami)}"
echo "Watch a run:      tail -f $BASE/logs/storm_<JOBID>_0.out   (task 0 = 5yr ... task 4 = 100yr)"
echo "Healthy start:    'Total rain: 106.7 mm' (5yr), 'Peak rain' about 9-16 mm/hr, H staying under a few metres"
echo "Each run writes 25 hourly rasters (t=0 ... 24 h) to $BASE/runs/output/<name>/"
