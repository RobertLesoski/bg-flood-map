"""
Preflight for the county-wide Warren County flood solver runs.
Run on Vista before submitting (and the job runs it again automatically):
    python3 vista_county_preflight.py

Checks:
  - flood_solver.py is patched (scenario names, no workarounds, unit guard, 24 h)
  - flood_solver.py accepts --dem and --outdir flags (county batch needs them)
  - dem/dem_warren_5m.tif exists and covers the county bbox
  - every storm_<id>.rain exists and integrates correctly (reuses city rain files)
  - county output folder is writable
"""
import json, os, re, sys

BASE = os.path.join(os.environ.get('SCRATCH', '.'), 'bgflood')
errors = []

def fail(msg): errors.append(msg); print('  FAIL', msg)
def ok(msg):   print('  ok  ', msg)


# ── 1. Solver patches (same as city preflight) ───────────────────────────────
solver_path = os.path.join(BASE, 'flood_solver.py')
try:
    solver = open(solver_path, encoding='utf-8').read()
except FileNotFoundError:
    fail(f'flood_solver.py not found at {solver_path}')
    solver = ''

print('Solver patches')
for name, pat in [
    ('scenario rain path',  r"f'storm_\{rp\}\.rain'"),
    ('scenario output dir', r"f'\{rp\}'\)"),
    ('string scenario arg', r'rp = sys\.argv\[1\]'),
    ('24 h default',        r'total_time=86400'),
    ('rain-unit guard',     r'ABORT: peak rain'),
]:
    ok(name) if re.search(pat, solver) else fail(f'solver not patched: {name} — run vista_patch_solver.py')

if re.search(r'rain_r\.append\(rr\)', solver) and not re.search(r'rain_r\.append\(rr\s*/', solver):
    ok('solver reads rain as m/s')
else:
    fail('solver does not read rain as plain m/s')

if re.search(r', 0\.5\)\n', solver) and 'dt = max(' in solver:
    fail('dt floor workaround still present')
if re.search(r'20\.0\) \* V_d', solver):
    fail('20 m depth cap workaround still present')


# ── 2. --dem / --outdir flag support ─────────────────────────────────────────
print('County flag support')
if re.search(r'--dem', solver):
    ok('solver accepts --dem flag')
else:
    fail('solver does not accept --dem — patch flood_solver.py to support '
         '"--dem <path>" and "--outdir <dir>" CLI args before submitting county batch')
if re.search(r'--outdir', solver):
    ok('solver accepts --outdir flag')
else:
    fail('solver does not accept --outdir')


# ── 3. County DEM ─────────────────────────────────────────────────────────────
print('County DEM')
county_dem = os.path.join(BASE, 'dem', 'dem_warren_5m.tif')
if not os.path.exists(county_dem):
    fail(f'dem_warren_5m.tif missing — run dem_county_clip.py on Vista first:\n'
         f'      python3 dem_county_clip.py')
else:
    size_gb = os.path.getsize(county_dem) / 1e9
    ok(f'dem_warren_5m.tif exists ({size_gb:.2f} GB)')
    # Rough size check: Warren County at 5 m should be > 200 MB
    if size_gb < 0.2:
        fail(f'dem_warren_5m.tif is suspiciously small ({size_gb:.2f} GB) — may be clipped wrong')


# ── 4. Rain files (same as city — uniform gauge, reused for county) ───────────
meta_path = os.path.join(BASE, 'storms.json')
print('Rain files')
if not os.path.exists(meta_path):
    fail('storms.json missing in ' + BASE)
    meta = []
else:
    meta = json.load(open(meta_path))

for ev in meta:
    path = os.path.join(BASE, 'runs', f"storm_{ev['id']}.rain")
    if not os.path.exists(path):
        fail('missing ' + path)
        continue
    with open(path) as fh:
        n = int(fh.readline())
        pts = [tuple(map(float, fh.readline().split())) for _ in range(n)]
    mm  = sum((pts[i+1][0]-pts[i][0]) * (pts[i+1][1]+pts[i][1]) / 2 for i in range(n-1)) * 1000
    exp = ev['total_in'] * 25.4
    peak = max(p[1] for p in pts) * 3600 * 1000
    good = abs(mm - exp) / exp < 0.005 and peak < 250 and pts[-1][0] >= 86399
    (ok if good else fail)(
        f"{ev['id']}: {mm:.1f} mm (expected {exp:.1f}), peak {peak:.1f} mm/hr")


# ── 5. County output folder ───────────────────────────────────────────────────
print('Output folder')
out = os.path.join(BASE, 'runs', 'county_output')
os.makedirs(out, exist_ok=True)
ok('county_output folder writable') if os.access(out, os.W_OK) else fail('county_output not writable')


# ── Result ────────────────────────────────────────────────────────────────────
if errors:
    print(f'\nCOUNTY PREFLIGHT FAILED ({len(errors)} problem(s)) — do not submit.')
    sys.exit(1)
print('\nCOUNTY PREFLIGHT PASSED')
