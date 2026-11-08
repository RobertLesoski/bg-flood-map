#!/bin/bash
# =============================================================================
#  ALL of Warren County, KY - every storm, one command.
#  Solver v2 (SCS infiltration + boundary outflow + mass balance) on a 5 m county grid.
#
#  Storms (12 = one H100 job each):
#     design : 2yr 5yr 10yr 25yr 50yr 100yr   (24 h, NOAA Atlas 14, same hyetograph as the city runs)
#     history: ev1 ev2 ev3 ev4 ev5 ev6        (KBWG hourly gauge blocks, see storms.json)
#
#  Run on a Vista LOGIN node, from the folder holding this file:
#       bash run_county_all.sh --dry-run     check everything, write the job file, submit nothing
#       bash run_county_all.sh               build DEM + rain, preflight, submit all 12
#       bash run_county_all.sh --only 25yr   (or --only "25yr ev4") submit just those storms
#       bash run_county_all.sh --skip-dem    DEM/roughness already built, do not rebuild
#       bash run_county_all.sh --throttle 4  run at most 4 storms at the same time (default: all 12)
#
#  Needs on Vista (already there from the city runs):
#       $SCRATCH/bgflood/flood_solver_v2.py        (written by run_5_storms_v2.sh)
#       $SCRATCH/bgflood/dem/dem_bg_2m.tif         (2 m city DEM, metres, EPSG:32616)
#       $SCRATCH/bgflood/dem/mannings_n.tif        (city roughness)
#       $SCRATCH/bgflood/runs/storm_{5yr..100yr}.rain, storm_ev{1..6}.rain
#       $SCRATCH/miniforge3  (python3 with cupy, rasterio, numpy; gdalwarp on PATH or in that env)
#  The county DEM is fetched here from USGS 3DEP (public, no login) - the login node has internet.
#
#  Output: $SCRATCH/bgflood/runs/county_v2/<storm>/depth_<s>s.tif (hourly) + max_depth.tif + run_summary.json
#  Download after the jobs finish, then see COUNTY_RUN.md for the conversion to the viewer.
# =============================================================================
set -u
DRY=0; SKIP_DEM=0; ONLY=""; THROTTLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY=1 ;;
    --skip-dem)  SKIP_DEM=1 ;;
    --only)      shift; ONLY="${1:-}" ;;
    --throttle)  shift; THROTTLE="${1:-}" ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
  shift
done

BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
cd "$BASE" || { echo "FATAL: $BASE not found"; exit 1; }
mkdir -p runs/county_v2 logs dem
fail() { echo; echo "FATAL: $*"; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_STORMS=(2yr 5yr 10yr 25yr 50yr 100yr ev1 ev2 ev3 ev4 ev5 ev6)
if [[ -n "$ONLY" ]]; then
  read -r -a STORMS <<< "$ONLY"
  for s in "${STORMS[@]}"; do
    [[ " ${ALL_STORMS[*]} " == *" $s "* ]] || fail "unknown storm '$s' (choose from: ${ALL_STORMS[*]})"
  done
else
  STORMS=("${ALL_STORMS[@]}")
fi

[[ -f "$SCRATCH/miniforge3/bin/activate" ]] || fail "$SCRATCH/miniforge3/bin/activate missing"
# shellcheck disable=SC1091
source "$SCRATCH/miniforge3/bin/activate"
command -v python3 >/dev/null || fail "python3 not found"
[[ -f flood_solver_v2.py ]] || fail "flood_solver_v2.py missing - run run_5_storms_v2.sh once first (it writes the solver)"
grep -q -- '--dem' flood_solver_v2.py && grep -q -- '--mann' flood_solver_v2.py && grep -q -- '--outdir' flood_solver_v2.py \
  || fail "flood_solver_v2.py has no --dem/--mann/--outdir options (old version)"
if command -v squeue >/dev/null 2>&1 && [[ -n "$(squeue -u "${USER:-$(whoami)}" -h -n warrenv2 2>/dev/null)" ]]; then
  fail "warrenv2 jobs are already queued or running. Cancel first (scancel -n warrenv2) to restart."
fi

# -----------------------------------------------------------------------------
echo "== 1/5  Rain files =="
cat > make_rain_county.py <<'PYEOF'
# Writes any missing design-storm rain file (m/s) and checks every rain file the county runs need.
import os, sys
B = os.path.join(os.environ['SCRATCH'], 'bgflood', 'runs')
DEPTH_IN = {'2yr': 3.5, '5yr': 4.2, '10yr': 4.9, '25yr': 5.9, '50yr': 6.7, '100yr': 7.6}   # NOAA Atlas 14, 24 h
_TF = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
_PF = [0, .04, .10, .19, .31, .50, .70, .79, .86, .92, 1.0]
TYPE = [(round(f * 24, 6), p) for f, p in zip(_TF, _PF)]
def cum(h):
    for (h0, c0), (h1, c1) in zip(TYPE, TYPE[1:]):
        if h0 <= h <= h1:
            return c0 + (c1 - c0) * (h - h0) / (h1 - h0)
    raise ValueError(h)
def write_design(name, d_in):
    P, STEP, pts = d_in * 0.0254, 300.0, []
    for k in range(int(86400 / STEP)):
        t0, t1 = k * STEP, (k + 1) * STEP
        rate = (cum(t1 / 3600.0) - cum(t0 / 3600.0)) * P / STEP
        pts += [(t0, rate), (t1 - 1.0, rate)]
    pts.append((86400.0, 0.0))
    with open(os.path.join(B, f'storm_{name}.rain'), 'w') as fh:
        fh.write(f'{len(pts)}\n')
        for t, r in pts: fh.write(f'{t:.1f} {r:.12e}\n')
def read(path):
    with open(path) as fh:
        n = int(fh.readline()); return [tuple(map(float, fh.readline().split())) for _ in range(n)]
def stats(pts):
    mm = sum((pts[i+1][0]-pts[i][0]) * (pts[i+1][1]+pts[i][1]) / 2 for i in range(len(pts)-1)) * 1000
    return mm, max(r for _, r in pts) * 3.6e6
bad = 0
for name in ['2yr', '5yr', '10yr', '25yr', '50yr', '100yr', 'ev1', 'ev2', 'ev3', 'ev4', 'ev5', 'ev6']:
    path = os.path.join(B, f'storm_{name}.rain')
    if name in DEPTH_IN:
        # the 2yr file on Vista was once overwritten with a wrong-unit file, so design files are re-verified and rewritten if off
        if os.path.exists(path):
            mm, peak = stats(read(path))
            if abs(mm - DEPTH_IN[name] * 25.4) / (DEPTH_IN[name] * 25.4) > 0.003 or not (0 < peak < 600):
                print(f'  {name}: existing file is wrong ({mm:.1f} mm, peak {peak:.0f} mm/hr) - rewriting'); write_design(name, DEPTH_IN[name])
        else:
            write_design(name, DEPTH_IN[name])
        mm, peak = stats(read(path)); want = DEPTH_IN[name] * 25.4
        good = abs(mm - want) / want < 0.003 and 0 < peak < 600
        print(f"  {'ok  ' if good else 'FAIL'}  {name:>5}: {mm:7.2f} mm (target {want:6.2f}), peak {peak:5.1f} mm/hr")
    else:
        if not os.path.exists(path):
            print(f'  FAIL  {name}: {path} missing - upload scripts/storms_out/storm_{name}.rain from your PC'); bad += 1; continue
        pts = read(path); mm, peak = stats(pts)
        good = 20 < mm < 600 and 0 < peak < 250 and pts[-1][0] >= 86399
        print(f"  {'ok  ' if good else 'FAIL'}  {name:>5}: {mm:7.2f} mm, peak {peak:5.1f} mm/hr (historical block)")
    bad += (not good)
sys.exit(1 if bad else 0)
PYEOF
python3 make_rain_county.py || fail "rain files not usable (see FAIL lines above)"

# -----------------------------------------------------------------------------
echo; echo "== 2/5  County DEM + roughness (5 m, UTM 16N) =="
if [[ -f "$HERE/build_county_grid.sh" ]]; then [[ "$HERE" -ef "$PWD" ]] || cp "$HERE/build_county_grid.sh" build_county_grid.sh; else echo "FATAL: build_county_grid.sh must sit next to this script"; exit 1; fi
chmod +x build_county_grid.sh
if [[ $SKIP_DEM -eq 1 || $DRY -eq 1 ]]; then
  echo "  skipped (dry run or --skip-dem) - run: bash build_county_grid.sh when ready"
  if [[ $DRY -eq 1 ]]; then echo "  dry run: leaving DEM files alone"; fi
else
  bash build_county_grid.sh || fail "county DEM build failed (message above)"
fi
if [[ $DRY -eq 0 ]]; then
  [[ -f dem/dem_warren_5m.tif && -f dem/mannings_n_county.tif ]] || fail "dem/dem_warren_5m.tif or dem/mannings_n_county.tif missing"
fi

# -----------------------------------------------------------------------------
echo; echo "== 3/5  Preflight =="
python3 - <<'PYEOF' || fail "preflight failed"
import os, re, sys
B = os.path.join(os.environ['SCRATCH'], 'bgflood')
s = open(os.path.join(B, 'flood_solver_v2.py'), encoding='utf-8').read()
bad = []
def chk(label, cond):
    print(('  ok    ' if cond else '  FAIL  ') + label)
    if not cond: bad.append(label)
chk('v2: SCS curve-number infiltration', 'scs_runoff_m' in s)
chk('v2: boundary outflow', 'OUTFLOW' in s and 'q_out' in s)
chk('v2: mass balance logged', 'mass_error_pct' in s)
chk('v2: rain read as plain m/s', bool(re.search(r'rain_r\.append\(rr\)', s)) and not re.search(r'rain_r\.append\(rr\s*/', s))
chk('v2: unit guard', 'ABORT: peak rain' in s)
chk('v2: 24 h default', 'total_time=86400' in s)
chk('v2: --dem/--mann/--outdir', all(f in s for f in ("'--dem'", "'--mann'", "'--outdir'")))
os.makedirs(os.path.join(B, 'runs', 'county_v2'), exist_ok=True)
chk('runs/county_v2 writable', os.access(os.path.join(B, 'runs', 'county_v2'), os.W_OK))
free = os.statvfs(B); gb = free.f_bavail * free.f_frsize / 1e9
print(f"  {'ok    ' if gb > 200 else 'WARN  '}scratch free space {gb:,.0f} GB (12 storms need up to about 200 GB; run fewer with --only if short)")
dem = os.path.join(B, 'dem', 'dem_warren_5m.tif')
if os.path.exists(dem):
    import rasterio
    with rasterio.open(dem) as d:
        mb = d.bounds; print(f"  info  county grid {d.width} x {d.height} cells = {d.width*d.height/1e6:.0f} M, EPSG {d.crs.to_epsg()}, "
                              f"{(mb.right-mb.left)/1000:.1f} x {(mb.top-mb.bottom)/1000:.1f} km")
        chk('county DEM is UTM 16N, 5 m', d.crs.to_epsg() == 32616 and abs(d.res[0] - 5) < 1e-6)
        chk('county grid under 150 M cells (one-GPU memory)', d.width * d.height < 150e6)
print('\n  PREFLIGHT ' + ('FAILED' if bad else 'PASSED'))
sys.exit(1 if bad else 0)
PYEOF

# -----------------------------------------------------------------------------
echo; echo "== 4/5  SLURM array (${#STORMS[@]} storms: ${STORMS[*]}; one H100 each) =="
LIST="${STORMS[*]}"
ARRAY="0-$(( ${#STORMS[@]} - 1 ))"; [[ -n "$THROTTLE" ]] && ARRAY="${ARRAY}%${THROTTLE}"
cat > submit_county_all.template <<'SLEOF'
#!/bin/bash
#SBATCH --job-name=warrenv2
#SBATCH --output=__BASE__/logs/county_%A_%a.out
#SBATCH --error=__BASE__/logs/county_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=__TIME__
#SBATCH --array=__ARRAY__
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com

STORM_LIST=(__LIST__)
ST=${STORM_LIST[$SLURM_ARRAY_TASK_ID]}

export PYTHONUNBUFFERED=1
export CN_PERVIOUS=75
export CN_IMPERVIOUS=98
export OUTFLOW=1
export OUTFLOW_SLOPE=0.005
source $SCRATCH/miniforge3/bin/activate
cd $SCRATCH/bgflood

echo "[$(date)] county ${ST} (solver v2, 5 m) on $(hostname)"
python3 flood_solver_v2.py $ST \
    --dem dem/dem_warren_5m.tif \
    --mann dem/mannings_n_county.tif \
    --outdir runs/county_v2/$ST
rc=$?
echo "[$(date)] county ${ST} finished with exit code $rc"
exit $rc
SLEOF
mk() { sed -e "s|__BASE__|$BASE|g" -e "s|__TIME__|$1|" -e "s|__ARRAY__|$ARRAY|" -e "s|__LIST__|$LIST|" submit_county_all.template > submit_county_all.sh; }

if [[ $DRY -eq 1 ]]; then
  mk 24:00:00
  echo "  DRY RUN - wrote submit_county_all.sh but did not submit. Storms: $LIST"
  exit 0
fi

submitted=0
for T in 24:00:00 20:00:00 12:00:00 08:00:00; do
  mk "$T"
  if OUT=$(sbatch submit_county_all.sh 2>&1); then
    echo "$OUT" | tail -2; echo "  submitted ${#STORMS[@]} storms with --time=$T"; submitted=1; break
  else
    echo "  sbatch rejected --time=$T: $(echo "$OUT" | tail -1)"
  fi
done
[[ $submitted -eq 1 ]] || fail "sbatch rejected every time limit - paste the message above to Claude"

echo; echo "== 5/5  Watching =="
echo "Queue:         squeue -u ${USER:-$(whoami)}"
echo "Watch a run:   tail -f $BASE/logs/county_<JOBID>_<n>.out    (n = position in: $LIST)"
echo "Healthy log:   'mass err' within about +/-0.5% each hour, 'Peak rain' sane, max depth in the low metres"
echo "Output:        $BASE/runs/county_v2/<storm>/"
echo "Cancel all:    scancel -n warrenv2"
