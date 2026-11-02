r"""
process_v2.py - one command to turn the downloaded solver-v2 rasters into viewer layers and a summary.

  python scripts\process_v2.py                 # every storm folder found (5yr 10yr 25yr 50yr 100yr)
  python scripts\process_v2.py --storms 5yr    # just some storms

Expects the rasters in:
  C:\\Users\\Minec\\Desktop\\DSOC AI\\Bowling Green Flood Map\\BG FLOOD MAP VISTA SUPERCOMPUTER\\v2\\<storm>\\
      depth_<seconds>s.tif (25 of them), max_depth.tif, run_summary.json         (what fetch_v2.sh downloads)

For every storm it:  checks the run health (mass balance) -> converts the hourly rasters to tier polygons
-> reprojects them into layers/flood_hpc/v2_<storm>/ for the viewer -> deletes each intermediate UTM file
-> computes flood statistics and compares the maximum extent with the FEMA flood zones.
It is safe to re-run: hours already in the viewer folder are skipped.
Writes:  <v2 folder>\\SUMMARY_v2.md  and  SUMMARY_v2.json
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
DSOC = os.environ.get("DSOC_ROOT", r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")
ROOT = os.path.join(DSOC, "BG FLOOD MAP VISTA SUPERCOMPUTER", "v2")
FEMA = os.path.join(DSOC, 'layers', 'fema.geojson')
OUT_LAYERS = os.path.join(DSOC, 'layers', 'flood_hpc')
STORMS = ['5yr', '10yr', '25yr', '50yr', '100yr']
DEPTH_IN = {'5yr': 4.2, '10yr': 4.9, '25yr': 5.9, '50yr': 6.7, '100yr': 7.6}
WET = 0.1                                   # m, same as the shallow tier threshold

# WGS84 / UTM 16N inverse (numpy version of reproject_hpc.utm_to_wgs84)
_A = 6378137.0; _F = 1 / 298.257223563; _E2 = 2 * _F - _F * _F; _EP2 = _E2 / (1 - _E2); _K0 = 0.9996
_LON0 = math.radians(-87.0); _E1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))


def utm_to_wgs84(e, n):
    x = e - 500000.0
    M = n / _K0
    mu = M / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    phi1 = (mu + (3 * _E1 / 2 - 27 * _E1 ** 3 / 32) * np.sin(2 * mu) + (21 * _E1 ** 2 / 16 - 55 * _E1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * _E1 ** 3 / 96) * np.sin(6 * mu) + (1097 * _E1 ** 4 / 512) * np.sin(8 * mu))
    s, c, tn = np.sin(phi1), np.cos(phi1), np.tan(phi1)
    N1 = _A / np.sqrt(1 - _E2 * s ** 2); T1 = tn ** 2; C1 = _EP2 * c ** 2
    R1 = _A * (1 - _E2) / (1 - _E2 * s ** 2) ** 1.5; D = x / (N1 * _K0)
    lat = phi1 - (N1 * tn / R1) * (D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * _EP2) * D ** 4 / 24
                                   + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * _EP2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = _LON0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * _EP2 + 24 * T1 ** 2) * D ** 5 / 120) / c
    return np.degrees(lon), np.degrees(lat)


def run(cmd, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run(cmd, env=env).returncode


def fema_geoms():
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union
    fc = json.load(open(FEMA, encoding='utf-8'))
    sfha, anyz = [], []
    for f in fc['features']:
        g = shape(f['geometry']).buffer(0)
        anyz.append(g)
        if f['properties'].get('FLD_ZONE') in ('AE', 'A', 'AO', 'AH'):
            sfha.append(g)
    return unary_union(sfha), unary_union(anyz)


def statistics(storm, folder, geoms):
    p = os.path.join(folder, 'max_depth.tif')
    if not os.path.exists(p):
        return None
    import shapely
    with rasterio.open(p) as src:
        a = src.read(1); tr = src.transform; res = src.res[0]
    valid = a != -9999
    v = a[valid]
    wet = v[v > WET]
    out = dict(valid_km2=v.size * res * res / 1e6, wet_km2=wet.size * res * res / 1e6,
               wet_pct=100.0 * wet.size / max(v.size, 1), median_wet_depth_m=float(np.median(wet)) if wet.size else 0.0,
               p99_depth_m=float(np.percentile(wet, 99)) if wet.size else 0.0, max_depth_m=float(wet.max()) if wet.size else 0.0,
               cells_ge_1p2m_pct_of_wet=100.0 * float((wet >= 1.2).sum()) / max(wet.size, 1))
    # FEMA comparison on a 1-in-10 sample grid
    step = 10
    rr, cc = np.mgrid[0:a.shape[0]:step, 0:a.shape[1]:step]
    xs = tr.c + (cc + 0.5) * tr.a; ys = tr.f + (rr + 0.5) * tr.e
    lon, lat = utm_to_wgs84(xs.ravel(), ys.ravel())
    sub = a[::step, ::step].ravel()
    ok = sub != -9999
    sfha, anyz = geoms
    in_sfha = shapely.contains_xy(sfha, lon, lat) & ok
    in_any = shapely.contains_xy(anyz, lon, lat) & ok
    is_wet = (sub > WET) & ok
    out['fema'] = dict(
        sfha_km2=float(in_sfha.sum()) * (res * step) ** 2 / 1e6,
        sfha_flooded_pct=100.0 * float((is_wet & in_sfha).sum()) / max(int(in_sfha.sum()), 1),
        flooded_in_sfha_pct=100.0 * float((is_wet & in_sfha).sum()) / max(int(is_wet.sum()), 1),
        flooded_in_any_fema_zone_pct=100.0 * float((is_wet & in_any).sum()) / max(int(is_wet.sum()), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--storms', nargs='*', default=STORMS)
    ap.add_argument('--keep-intermediate', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    summary = {}
    geoms = None
    for s in a.storms:
        folder = os.path.join(ROOT, s)
        print(f'\n{"=" * 70}\n  {s}   ({folder})\n{"=" * 70}', flush=True)
        if not os.path.isdir(folder):
            print('  [skip] folder not found - download it first (fetch_v2.sh)')
            continue
        tifs = [f for f in os.listdir(folder) if f.startswith('depth_') and f.endswith('.tif')]
        info = {'depth_rasters': len(tifs)}
        sj = os.path.join(folder, 'run_summary.json')
        if os.path.exists(sj):
            rs = json.load(open(sj))
            fb = rs.get('final_mass_balance', {})
            info.update(rain_mm=rs.get('total_rain_mm'), runoff_mm=rs.get('runoff_depth_mm'),
                        mass_error_pct=fb.get('mass_error_pct'), outflow_m3=fb.get('outflow_m3'), stored_m3=fb.get('stored_m3'),
                        infiltrated_m3=fb.get('infiltrated_m3'), rain_m3=fb.get('rain_m3'), wall_minutes=rs.get('wall_minutes'))
            flag = '' if abs(fb.get('mass_error_pct', 99)) < 1 else '   <-- CHECK: mass error above 1%'
            print(f"  run health: rain {rs.get('total_rain_mm', 0):.1f} mm -> runoff {rs.get('runoff_depth_mm', 0):.1f} mm, "
                  f"mass error {fb.get('mass_error_pct', float('nan')):+.4f}%{flag}")
        else:
            print('  [warn] run_summary.json missing (run may be incomplete)')
        if len(tifs) < 25:
            print(f'  [warn] only {len(tifs)} depth rasters (expected 25) - the run may not have finished or the download is partial')

        prefix = f'v2_{s}'
        vdir = os.path.join(OUT_LAYERS, prefix)
        done = sorted(int(f[7:9]) for f in os.listdir(vdir) if f.startswith('flood_h') and f.endswith('.geojson')) if os.path.isdir(vdir) else []
        skip = ','.join(str(h) for h in done)
        print(f'  converting (hours already in the viewer: {len(done)})...', flush=True)
        rc = run([sys.executable, os.path.join(HERE, 'tif_to_geojson.py'), folder, prefix, '--skip-hours', skip])
        if rc != 0:
            print('  [FAIL] conversion failed'); continue
        print('  reprojecting for the viewer...', flush=True)
        rc = run([sys.executable, os.path.join(HERE, 'reproject_hpc.py'), prefix], {} if a.keep_intermediate else {'DELETE_SRC': '1'})
        if rc != 0:
            print('  [FAIL] reprojection failed'); continue
        if geoms is None and os.path.exists(FEMA):
            print('  loading FEMA zones...', flush=True)
            geoms = fema_geoms()
        st = statistics(s, folder, geoms) if geoms else None
        if st:
            info.update(st)
            print(f"  flooded (max): {st['wet_km2']:.2f} km2 = {st['wet_pct']:.1f}% of the domain | median depth {st['median_wet_depth_m']:.2f} m | "
                  f"p99 {st['p99_depth_m']:.2f} m | max {st['max_depth_m']:.2f} m")
            fm = st['fema']
            print(f"  vs FEMA: {fm['sfha_flooded_pct']:.0f}% of the FEMA 1%-annual-chance zone (AE/A) is flooded; "
                  f"{fm['flooded_in_sfha_pct']:.0f}% of the flooded area is inside it; {fm['flooded_in_any_fema_zone_pct']:.0f}% inside any FEMA zone")
        summary[s] = info

    # write the summary
    os.makedirs(ROOT, exist_ok=True)
    json.dump(summary, open(os.path.join(ROOT, 'SUMMARY_v2.json'), 'w'), indent=1)
    L = ['# Solver v2 runs - summary', '',
         'Model: 2D local-inertial, SCS curve-number infiltration (CN 75 open / 98 roads+buildings, uncalibrated), free outflow at the domain edge.',
         '', '| Storm | Rain (mm) | Runoff (mm) | Infiltrated % | Mass error % | Flooded km2 (>0.1 m) | Median depth (m) | Max depth (m) | FEMA 1% zone flooded % | Flooded inside FEMA 1% zone % |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for s, i in summary.items():
        rain, ro = i.get('rain_mm'), i.get('runoff_mm')
        inf = f"{100 * (1 - ro / rain):.1f}" if rain and ro is not None else '-'
        fm = i.get('fema', {})
        L.append(f"| {s} | {rain or 0:.1f} | {ro or 0:.1f} | {inf} | {i.get('mass_error_pct', float('nan')):+.4f} | {i.get('wet_km2', 0):.2f} | "
                 f"{i.get('median_wet_depth_m', 0):.2f} | {i.get('max_depth_m', 0):.2f} | {fm.get('sfha_flooded_pct', 0):.0f} | {fm.get('flooded_in_sfha_pct', 0):.0f} |")
    L += ['', 'Notes:',
          '- The FEMA 1% zone includes flooding from the Barren River and creeks that this model does not simulate (no river inflow or channel flow), so a low "FEMA zone flooded" figure is expected.',
          '- Maximum depths sit in closed depressions (quarries, sinkholes, ponds): the model has no drains or underground flow.',
          '- Comparison with the FEMA zones is a plausibility check, not a calibration.']
    open(os.path.join(ROOT, 'SUMMARY_v2.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'\nSummary written to {os.path.join(ROOT, "SUMMARY_v2.md")}   ({(time.time() - t0) / 60:.1f} min)')
    print('Next: hard-refresh the viewer (Ctrl+F5) and pick a storm from the Storm menu.')


if __name__ == '__main__':
    main()
