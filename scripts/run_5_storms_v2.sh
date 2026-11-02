#!/bin/bash
# =============================================================================
#  Bowling Green flood model, solver v2 (SCS infiltration + boundary outflow + built-in mass balance)
#  5, 10, 25, 50 and 100-year storms.   Run on a Vista LOGIN node:   bash run_5_storms_v2.sh
#  Dry run (everything except submit):                                bash run_5_storms_v2.sh --dry-run
#
#  Nothing existing is touched: the old runs and flood_solver.py stay as they are.
#  New output goes to runs/output_v2/<storm>/.
#  Assumptions (edit the ENV lines in submit_5storms_v2.sh to change them):
#     CN_PERVIOUS=75  (open land, AMC II, assumed typical for Warren Co. soils - NOT calibrated)
#     CN_IMPERVIOUS=98 (roads + buildings, detected from the Manning's n raster)
#     OUTFLOW=1 with energy slope 0.005 at the domain edge
# =============================================================================
set -u
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
BASE="${SCRATCH:?SCRATCH is not set - run this on Vista}/bgflood"
cd "$BASE" || { echo "FATAL: $BASE not found"; exit 1; }
mkdir -p runs logs
fail() { echo; echo "FATAL: $*"; exit 1; }

[[ -f dem/dem_bg_2m.tif && -f dem/mannings_n.tif ]] || fail "dem/dem_bg_2m.tif or dem/mannings_n.tif missing"
[[ -f "$SCRATCH/miniforge3/bin/activate" ]]         || fail "$SCRATCH/miniforge3/bin/activate missing"
command -v python3 >/dev/null                       || fail "python3 not found"
if command -v squeue >/dev/null 2>&1 && [[ -n "$(squeue -u "${USER:-$(whoami)}" -h -n bgflood5v2 2>/dev/null)" ]]; then
  fail "bgflood5v2 jobs are already queued or running. Cancel first (scancel -n bgflood5v2) if you want to restart."
fi

echo "== 1/4  Writing flood_solver_v2.py =="
[[ -f flood_solver_v2.py ]] && cp flood_solver_v2.py "flood_solver_v2.py.bak_$(date +%Y%m%d_%H%M%S)"
cat > flood_solver_v2.py <<'SOLVEREOF'
"""
flood_solver_v2.py - 2D local-inertial (LISFLOOD-FP "acceleration") flood solver on the GPU, with
  * SCS Curve Number infiltration (cumulative-rain method, per land-cover class)
  * free outflow at the domain boundary (normal-depth / Manning), so water can leave
  * a built-in mass balance written to the log at every output, and a summary json

Usage (on Vista, from $SCRATCH/bgflood):   python3 flood_solver_v2.py 5yr
  reads   runs/storm_<name>.rain           (line 1 = N, then "seconds  rate_m_per_s")
          dem/dem_bg_2m.tif, dem/mannings_n.tif
  writes  runs/output_v2/<name>/depth_<seconds>s.tif (hourly + final), max_depth.tif, run_summary.json

Environment overrides (assumptions, NOT calibrated - see the README note in the log):
  CN_PERVIOUS=75   curve number for open land / soil  (AMC II)
  CN_IMPERVIOUS=98 curve number for roads and buildings (detected from the Manning's n raster)
  OUTFLOW=1        0 closes the domain edges
  OUTFLOW_SLOPE=0.005   energy slope used for the boundary outflow
  ALPHA=0.3        CFL number
"""
import argparse, json, os, sys, time, traceback
import numpy as np
import rasterio

BASE = os.path.dirname(os.path.abspath(__file__))
G = 9.81
H_MIN = 1e-3
ALPHA = float(os.environ.get('ALPHA', 0.3))
CN_PERV = float(os.environ.get('CN_PERVIOUS', 75))
CN_IMP = float(os.environ.get('CN_IMPERVIOUS', 98))
OUTFLOW = os.environ.get('OUTFLOW', '1') != '0'
S0 = float(os.environ.get('OUTFLOW_SLOPE', 0.005))


def scs_runoff_m(p_m, cn):
    """Cumulative SCS runoff depth (m) for cumulative rainfall p_m (m); Ia = 0.2 S."""
    s = (1000.0 / cn - 10.0) * 0.0254
    ia = 0.2 * s
    return 0.0 if p_m <= ia else (p_m - ia) ** 2 / (p_m - ia + s)


def run_storm(name, total_time=86400, out_interval=3600, dem_path=None, man_path=None, rain_path=None, out_dir=None):
    print(f"\n=== {name} storm (solver v2) ===", flush=True)
    t0 = time.time()
    import cupy as cp
    print(f"CuPy {cp.__version__} | GPU {cp.cuda.Device().id}", flush=True)
    dem_path = dem_path or os.path.join(BASE, 'dem', 'dem_bg_2m.tif')
    man_path = man_path or os.path.join(BASE, 'dem', 'mannings_n.tif')
    rain_path = rain_path or os.path.join(BASE, 'runs', f'storm_{name}.rain')
    out_dir = out_dir or os.path.join(BASE, 'runs', 'output_v2', name)
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(dem_path) as src:
        Z = src.read(1).astype(np.float32)
        dx = float(src.res[0]); dy = float(src.res[1])
        nodata = src.nodata if src.nodata is not None else -9999
        profile = src.profile.copy()
    with rasterio.open(man_path) as src:
        N = src.read(1).astype(np.float32)
    Z[Z == nodata] = np.nan
    valid = ~np.isnan(Z)
    Z[~valid] = 999.0
    rows, cols = Z.shape
    n_valid = int(valid.sum())
    cell_area = dx * dy
    print(f"Grid: {cols}x{rows} | Valid: {n_valid:,} | dx={dx} m", flush=True)

    # land-cover class from the Manning's n raster: roads (n~0.016) and buildings (n~0.08) are impervious
    imp = valid & ((N < 0.02) | ((N > 0.07) & (N < 0.09)))
    n_imp = int(imp.sum()); n_perv = n_valid - n_imp
    print(f"Land cover: pervious {n_perv:,} (CN {CN_PERV:g}) | impervious {n_imp:,} (CN {CN_IMP:g})", flush=True)

    # rain: seconds + m/s
    rain_t, rain_r = [], []
    with open(rain_path) as f:
        n = int(f.readline())
        for _ in range(n):
            tt, rr = map(float, f.readline().split())
            rain_t.append(tt); rain_r.append(rr)       # already m/s
    rain_t = np.array(rain_t); rain_r = np.array(rain_r)
    peak_mmhr = rain_r.max() * 3600 * 1000
    total_mm = float(((rain_r[1:] + rain_r[:-1]) * 0.5 * np.diff(rain_t)).sum()) * 1000
    print(f"Peak rain: {peak_mmhr:.2f} mm/hr", flush=True)
    print(f"Total rain: {total_mm:.1f} mm ({total_mm/25.4:.2f} in)", flush=True)
    if peak_mmhr > 600:
        print(f"ABORT: peak rain {peak_mmhr:.0f} mm/hr is implausible - rain file units wrong (solver expects m/s)", flush=True)
        sys.exit(2)

    # domain boundary: valid cells touching an invalid cell or the array edge; nexp = number of exposed faces
    vp = np.pad(valid, 1, constant_values=False)
    nexp = ((~vp[:-2, 1:-1]).astype(np.uint8) + (~vp[2:, 1:-1]) + (~vp[1:-1, :-2]) + (~vp[1:-1, 2:])).astype(np.uint8)
    nexp[~valid] = 0
    brow, bcol = np.nonzero(nexp > 0)
    bn = nexp[brow, bcol].astype(np.float32)
    print(f"Boundary cells: {len(brow):,} | outflow {'ON' if OUTFLOW else 'OFF'} (slope {S0})", flush=True)

    Z_d = cp.asarray(Z); N_d = cp.asarray(N); V_d = cp.asarray(valid.astype(np.float32))
    C_d = cp.asarray(imp.astype(np.float32))
    H_d = cp.zeros((rows, cols), dtype=cp.float32)
    Qx_d = cp.zeros((rows, cols), dtype=cp.float32)
    Qy_d = cp.zeros((rows, cols), dtype=cp.float32)
    brow_d = cp.asarray(brow); bcol_d = cp.asarray(bcol); bn_d = cp.asarray(bn)
    nb_d = N_d[brow_d, bcol_d] if len(brow) else None
    profile.update(dtype='float32', count=1, nodata=-9999)

    t = 0.0; step = 0; next_out = 0.0
    p_cum = 0.0                      # cumulative rainfall, m (same for every cell)
    out_vol = 0.0                    # cumulative outflow volume, m3
    mass = []
    print("Simulation starting...", flush=True)

    def balance():
        rain_vol = p_cum * n_valid * cell_area
        runoff_vol = (scs_runoff_m(p_cum, CN_PERV) * n_perv + scs_runoff_m(p_cum, CN_IMP) * n_imp) * cell_area
        stored = float(H_d.sum()) * cell_area
        err = runoff_vol - out_vol - stored
        rel = 100.0 * err / runoff_vol if runoff_vol > 0 else 0.0
        return dict(t=t, rain_m3=rain_vol, runoff_m3=runoff_vol, infiltrated_m3=rain_vol - runoff_vol,
                    outflow_m3=out_vol, stored_m3=stored, mass_error_pct=rel)

    while t < total_time:
        h_max = max(float(cp.max(H_d)), H_MIN)
        dt = min(ALPHA * min(dx, dy) / (np.sqrt(G * h_max) + 1e-6), 30.0, total_time - t)

        r_rate = float(np.interp(t, rain_t, rain_r))
        p_old = p_cum
        p_cum += r_rate * dt
        q_perv = (scs_runoff_m(p_cum, CN_PERV) - scs_runoff_m(p_old, CN_PERV)) / dt      # m/s of runoff
        q_imp = (scs_runoff_m(p_cum, CN_IMP) - scs_runoff_m(p_old, CN_IMP)) / dt
        eta = Z_d + H_d

        deta_x = cp.diff(eta, axis=1)
        h_flow_x = cp.maximum(cp.maximum(eta[:, :-1], eta[:, 1:]) - cp.maximum(Z_d[:, :-1], Z_d[:, 1:]), 0.0)
        n_x = (N_d[:, :-1] + N_d[:, 1:]) * 0.5
        denom_x = 1.0 + dt * G * n_x ** 2 * cp.abs(Qx_d[:, :-1]) / (h_flow_x ** (7.0 / 3) + 1e-10)
        Qx_new = cp.where(h_flow_x > H_MIN, (Qx_d[:, :-1] - dt * G * h_flow_x * deta_x / dx) / denom_x, 0.0)
        vol_lim = cp.minimum(H_d[:, :-1], H_d[:, 1:]) * dx / (dt + 1e-10)
        Qx_d[:, :-1] = cp.sign(Qx_new) * cp.minimum(cp.abs(Qx_new), vol_lim)

        deta_y = cp.diff(eta, axis=0)
        h_flow_y = cp.maximum(cp.maximum(eta[:-1, :], eta[1:, :]) - cp.maximum(Z_d[:-1, :], Z_d[1:, :]), 0.0)
        n_y = (N_d[:-1, :] + N_d[1:, :]) * 0.5
        denom_y = 1.0 + dt * G * n_y ** 2 * cp.abs(Qy_d[:-1, :]) / (h_flow_y ** (7.0 / 3) + 1e-10)
        Qy_new = cp.where(h_flow_y > H_MIN, (Qy_d[:-1, :] - dt * G * h_flow_y * deta_y / dy) / denom_y, 0.0)
        vol_lim_y = cp.minimum(H_d[:-1, :], H_d[1:, :]) * dy / (dt + 1e-10)
        Qy_d[:-1, :] = cp.sign(Qy_new) * cp.minimum(cp.abs(Qy_new), vol_lim_y)

        r_eff = q_perv + (q_imp - q_perv) * C_d                                           # effective rain, m/s

        # Outflow limiter: no cell may export more water than it holds plus what it receives as rain.
        # Without it a cell with several outflows can go negative and be clamped to zero, which creates water.
        tot = cp.zeros((rows, cols), dtype=cp.float32)                                    # outflow rate per cell, m/s
        tot[:, :-1] += cp.maximum(Qx_d[:, :-1], 0.0) / dx
        tot[:, 1:] += cp.maximum(-Qx_d[:, :-1], 0.0) / dx
        tot[:-1, :] += cp.maximum(Qy_d[:-1, :], 0.0) / dy
        tot[1:, :] += cp.maximum(-Qy_d[:-1, :], 0.0) / dy
        f_lim = cp.where(tot * dt > H_d + dt * r_eff, (H_d + dt * r_eff) / (tot * dt + 1e-30), 1.0)
        Qx_d[:, :-1] = cp.where(Qx_d[:, :-1] >= 0, Qx_d[:, :-1] * f_lim[:, :-1], Qx_d[:, :-1] * f_lim[:, 1:])
        Qy_d[:-1, :] = cp.where(Qy_d[:-1, :] >= 0, Qy_d[:-1, :] * f_lim[:-1, :], Qy_d[:-1, :] * f_lim[1:, :])

        dQx = cp.diff(Qx_d, axis=1, prepend=cp.zeros((rows, 1), dtype=cp.float32))
        dQy = cp.diff(Qy_d, axis=0, prepend=cp.zeros((1, cols), dtype=cp.float32))
        H_d = cp.maximum(H_d + dt * (r_eff - dQx / dx - dQy / dy), 0.0) * V_d

        if OUTFLOW and len(brow):                                                         # free outflow, Manning normal depth
            hb = H_d[brow_d, bcol_d]
            q_out = (hb ** (5.0 / 3.0)) * np.sqrt(S0) / nb_d                              # m2/s per exposed face
            dh = cp.minimum(q_out * dt / dx * bn_d, hb)
            H_d[brow_d, bcol_d] = hb - dh
            out_vol += float(dh.sum()) * cell_area
        t += dt; step += 1

        if step == 1:
            print(f"Step1: dt={dt:.2f}s r={r_rate:.3e}m/s H={float(H_d.max()):.4f}m", flush=True)
        if step % 1000 == 0:
            print(f"  step{step} t={t:.0f}s dt={dt:.2f}s H={float(H_d.max()):.3f}m", flush=True)

        if t >= next_out:
            H_np = cp.asnumpy(H_d); H_np[~valid] = -9999.0
            with rasterio.open(os.path.join(out_dir, f'depth_{int(t):06d}s.tif'), 'w', **profile) as dst:
                dst.write(H_np, 1)
            wet = int((H_np > H_MIN).sum())
            maxh = float(H_np[H_np > 0].max()) if wet > 0 else 0.0
            b = balance(); mass.append(b)
            flag = '' if abs(b['mass_error_pct']) < 1.0 else '  <-- MASS BALANCE WARNING'
            print(f"  t={t/3600:.1f}hr|wet={wet:,}|max_h={maxh:.3f}m|runoff={b['runoff_m3']:.3e} out={b['outflow_m3']:.3e} "
                  f"stored={b['stored_m3']:.3e} m3|mass err {b['mass_error_pct']:+.3f}%|{time.time()-t0:.0f}s{flag}", flush=True)
            next_out += out_interval

    if not os.path.exists(os.path.join(out_dir, f'depth_{int(t):06d}s.tif')):
        H_np = cp.asnumpy(H_d); H_np[~valid] = -9999.0
        with rasterio.open(os.path.join(out_dir, f'depth_{int(t):06d}s.tif'), 'w', **profile) as dst:
            dst.write(H_np, 1)
        mass.append(balance())
        print(f"  final state saved at t={t/3600:.2f}hr", flush=True)

    files = sorted(f for f in os.listdir(out_dir) if f.startswith('depth_') and f.endswith('.tif'))
    if files:
        H_max = np.max([rasterio.open(os.path.join(out_dir, f)).read(1) for f in files], axis=0)
        H_max[~valid] = -9999.0
        with rasterio.open(os.path.join(out_dir, 'max_depth.tif'), 'w', **profile) as dst:
            dst.write(H_max, 1)
        print(f"Max depth: {H_max[H_max > 0].max():.2f}m", flush=True)
    final = mass[-1] if mass else {}
    summary = dict(storm=name, solver='v2', total_rain_mm=total_mm, cn_pervious=CN_PERV, cn_impervious=CN_IMP,
                   outflow=OUTFLOW, outflow_slope=S0, valid_cells=n_valid, impervious_cells=n_imp,
                   runoff_depth_mm=final.get('runoff_m3', 0) / (n_valid * cell_area) * 1000 if n_valid else 0,
                   final_mass_balance=final, wall_minutes=(time.time() - t0) / 60)
    with open(os.path.join(out_dir, 'run_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f"Done {name} in {(time.time()-t0)/60:.1f}min | final mass error {final.get('mass_error_pct', 0):+.3f}%", flush=True)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--dem'); ap.add_argument('--mann'); ap.add_argument('--outdir')
    a = ap.parse_args()
    try:
        run_storm(a.name, dem_path=a.dem, man_path=a.mann, out_dir=a.outdir)
    except SystemExit:
        raise
    except Exception as e:
        print(f"FATAL: {e}", flush=True); traceback.print_exc(); sys.exit(1)
SOLVEREOF
python3 -m py_compile flood_solver_v2.py || fail "flood_solver_v2.py does not compile"
echo "  written and compiles ($(wc -l < flood_solver_v2.py) lines)"

echo; echo "== 2/4  Checking the rain files (units m/s, depth, 24 h) =="
python3 - <<'PYEOF' || fail "rain file check failed - the 5 rain files from run_5_storms.sh must exist in runs/"
import os, sys
B = os.path.join(os.environ['SCRATCH'], 'bgflood', 'runs')
want = {'5yr': 4.2, '10yr': 4.9, '25yr': 5.9, '50yr': 6.7, '100yr': 7.6}
bad = 0
for k, d_in in want.items():
    p = os.path.join(B, f'storm_{k}.rain')
    if not os.path.exists(p):
        print(f'  FAIL  missing {p}'); bad += 1; continue
    with open(p) as fh:
        n = int(fh.readline()); pts = [tuple(map(float, fh.readline().split())) for _ in range(n)]
    mm = sum((pts[i+1][0]-pts[i][0]) * (pts[i+1][1]+pts[i][1]) / 2 for i in range(n-1)) * 1000
    peak = max(r for _, r in pts) * 3.6e6
    ok = abs(mm - d_in*25.4) / (d_in*25.4) < 0.003 and pts[-1][0] >= 86399 and 0 < peak < 600
    print(f"  {'ok  ' if ok else 'FAIL'}  {k:>5}: {mm:7.2f} mm (target {d_in*25.4:6.2f}), peak {peak:5.1f} mm/hr")
    bad += (not ok)
sys.exit(1 if bad else 0)
PYEOF

echo; echo "== 3/4  Preflight =="
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
chk('v2: time advances', bool(re.search(r'\bt \+= dt', s)) and bool(re.search(r'\bstep \+= 1', s)))
chk('v2: 24 h default', 'total_time=86400' in s)
chk('v2: output to runs/output_v2', "'output_v2'" in s)
os.makedirs(os.path.join(B, 'runs', 'output_v2'), exist_ok=True)
chk('output_v2 writable', os.access(os.path.join(B, 'runs', 'output_v2'), os.W_OK))
print('\n  PREFLIGHT ' + ('FAILED' if bad else 'PASSED'))
sys.exit(1 if bad else 0)
PYEOF

echo; echo "== 4/4  SLURM array (5 tasks: 5yr 10yr 25yr 50yr 100yr, one H100 each) =="
cat > submit_5storms_v2.template <<'SLEOF'
#!/bin/bash
#SBATCH --job-name=bgflood5v2
#SBATCH --output=__BASE__/logs/v2_%A_%a.out
#SBATCH --error=__BASE__/logs/v2_%A_%a.err
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
export CN_PERVIOUS=75
export CN_IMPERVIOUS=98
export OUTFLOW=1
export OUTFLOW_SLOPE=0.005
source $SCRATCH/miniforge3/bin/activate
cd $SCRATCH/bgflood

echo "[$(date)] Starting ${RP} (solver v2) on $(hostname)"
python3 flood_solver_v2.py $RP
rc=$?
echo "[$(date)] ${RP} finished with exit code $rc"
exit $rc
SLEOF

if [[ $DRY -eq 1 ]]; then
  sed -e "s|__BASE__|$BASE|g" -e "s|__TIME__|40:00:00|" submit_5storms_v2.template > submit_5storms_v2.sh
  echo "  DRY RUN - wrote submit_5storms_v2.sh but did not submit."
  exit 0
fi

submitted=0
for T in 40:00:00 36:00:00 24:00:00 12:00:00; do
  sed -e "s|__BASE__|$BASE|g" -e "s|__TIME__|$T|" submit_5storms_v2.template > submit_5storms_v2.sh
  if OUT=$(sbatch submit_5storms_v2.sh 2>&1); then
    echo "$OUT" | tail -2; echo "  submitted with --time=$T"; submitted=1; break
  else
    echo "  sbatch rejected --time=$T: $(echo "$OUT" | tail -1)"
  fi
done
[[ $submitted -eq 1 ]] || fail "sbatch rejected every time limit - paste the message above to Claude"

echo
echo "Check queue:   squeue -u ${USER:-$(whoami)}"
echo "Watch a run:   tail -f $BASE/logs/v2_<JOBID>_0.out   (task 0 = 5yr ... task 4 = 100yr)"
echo "Healthy log:   'mass err' within about +/-0.5% at every hour; runoff well below rain; max_h in the low metres"
echo "Output:        $BASE/runs/output_v2/<storm>/   (25 hourly rasters + max_depth.tif + run_summary.json)"
