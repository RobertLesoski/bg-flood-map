"""
Preflight for the historical-storm runs.  Run on Vista before submitting (and the job runs it again):
    python3 vista_preflight.py

Checks, exiting non-zero on any failure:
  - flood_solver.py has been patched (scenario names, no workarounds, unit guard, 24 h)
  - flood_solver.py reads rain values as m/s (the unit these rain files use)
  - every storm_<id>.rain exists, integrates to the gauge total in storms.json (+/-0.5 %),
    and has a plausible peak intensity
  - DEM / roughness inputs exist and the output folder is writable
"""
import json, os, re, sys

BASE = os.path.join(os.environ.get('SCRATCH', '.'), 'bgflood')
errors = []

def fail(msg):
    errors.append(msg)
    print('  FAIL', msg)

def ok(msg):
    print('  ok  ', msg)

solver = open(os.path.join(BASE, 'flood_solver.py'), encoding='utf-8').read()
print('Solver')
for name, pat in [('scenario rain path', r"f'storm_\{rp\}\.rain'"),
                  ('scenario output dir', r"f'\{rp\}'\)"),
                  ('string scenario arg', r'rp = sys\.argv\[1\]'),
                  ('24 h default', r'total_time=86400'),
                  ('rain-unit guard', r'ABORT: peak rain'),
                  ('final-state save', r'final state saved')]:
    ok(name) if re.search(pat, solver) else fail(f'solver not patched: {name} - run vista_patch_solver.py')
if re.search(r'rain_r\.append\(rr\)', solver) and not re.search(r'rain_r\.append\(rr\s*/', solver):
    ok('solver reads rain as m/s')
else:
    fail('solver does not read rain as plain m/s - rain files would be wrong')
if re.search(r', 0\.5\)\n', solver) and 'dt = max(' in solver:
    fail('dt floor workaround still present')
if re.search(r'20\.0\) \* V_d', solver):
    fail('20 m depth cap workaround still present')

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
    mm = sum((pts[i+1][0]-pts[i][0]) * (pts[i+1][1]+pts[i][1]) / 2 for i in range(n-1)) * 1000
    exp = ev['total_in'] * 25.4
    peak = max(p[1] for p in pts) * 3600 * 1000
    good = abs(mm - exp) / exp < 0.005 and peak < 250 and pts[-1][0] >= 86399
    (ok if good else fail)(f"{ev['id']}: {mm:.1f} mm (expected {exp:.1f}), peak {peak:.1f} mm/hr, {ev['label']}")

print('Inputs')
for rel in ('dem/dem_bg_2m.tif', 'dem/mannings_n.tif'):
    ok(rel) if os.path.exists(os.path.join(BASE, rel)) else fail('missing ' + rel)
out = os.path.join(BASE, 'runs', 'output')
os.makedirs(out, exist_ok=True)
ok('output folder writable') if os.access(out, os.W_OK) else fail('output folder not writable')

if errors:
    print(f'\nPREFLIGHT FAILED ({len(errors)} problem(s)) - do not submit.')
    sys.exit(1)
print('\nPREFLIGHT PASSED')
