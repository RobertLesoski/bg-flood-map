"""
Reproject HPC flood GeoJSON files from UTM Zone 16N (EPSG:32616) → WGS84 (EPSG:4326).

Usage:
  python reproject_hpc.py                  # defaults to 2yr (city run)
  python reproject_hpc.py 5yr             # city storm prefix
  python reproject_hpc.py all             # all city storm prefixes
  python reproject_hpc.py county_ev1      # single county storm event
  python reproject_hpc.py county_all      # all county storm events (ev1–ev6)

City run prefixes:   2yr 5yr 10yr 25yr 50yr 100yr
County run prefixes: county_ev1 … county_ev6
  (county source files expected in SRC_DIR/county_output/<ev>/)
Solver-v2 prefixes:  v2_5yr v2_10yr v2_25yr v2_50yr v2_100yr   (or v2_all)
  (source: BG FLOOD MAP VISTA SUPERCOMPUTER/v2/<storm>/<prefix>_depth_<sec>s.geojson; writes only to the viewer's layers folder,
   skips hours already done, and with env DELETE_SRC=1 removes each intermediate UTM file after it is converted)

Outputs to:  layers/flood_hpc/<prefix>/flood_h01.geojson ...
"""

import json, math, os, sys


SRC_DIR = os.environ.get("DSOC_ROOT", r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")
BASE_DST = r"C:\Users\Minec\Documents\BG_3D_Map\layers\flood_hpc"
DSOC_DST = os.path.join(SRC_DIR, "layers", "flood_hpc")

ALL_PREFIXES        = ['2yr', '5yr', '10yr', '25yr', '50yr', '100yr']
ALL_COUNTY_PREFIXES = ['county_ev1', 'county_ev2', 'county_ev3',
                       'county_ev4', 'county_ev5', 'county_ev6']
ALL_V2_PREFIXES     = ['v2_5yr', 'v2_10yr', 'v2_25yr', 'v2_50yr', 'v2_100yr']

# ── WGS84 / UTM Zone 16N parameters ──────────────────────────────────────────
A    = 6378137.0
F    = 1.0 / 298.257223563
B    = A * (1.0 - F)
E2   = 2.0 * F - F * F
EP2  = E2 / (1.0 - E2)
K0   = 0.9996
ZONE = 16
LON0 = math.radians((ZONE - 1) * 6 - 180 + 3)   # -87° for Zone 16N
E1   = (1.0 - math.sqrt(1.0 - E2)) / (1.0 + math.sqrt(1.0 - E2))

HOUR_SECONDS = {h: f'{h * 3600:06d}' for h in range(1, 25)}   # hours 1-24 (a source file is skipped if absent)

TIER_DEPTH = {
    'shallow':  (0.1, 0.3),
    'moderate': (0.3, 0.6),
    'deep':     (0.6, 1.2),
    'severe':   (1.2, 5.0),
}


def utm_to_wgs84(easting, northing):
    x  = easting - 500000.0
    y  = northing
    M  = y / K0
    mu = M / (A * (1.0 - E2/4.0 - 3.0*E2**2/64.0 - 5.0*E2**3/256.0))
    phi1 = (mu
            + (3.0*E1/2.0  - 27.0*E1**3/32.0)       * math.sin(2.0*mu)
            + (21.0*E1**2/16.0 - 55.0*E1**4/32.0)   * math.sin(4.0*mu)
            + (151.0*E1**3/96.0)                      * math.sin(6.0*mu)
            + (1097.0*E1**4/512.0)                    * math.sin(8.0*mu))
    sin1 = math.sin(phi1); tan1 = math.tan(phi1); cos1 = math.cos(phi1)
    N1 = A / math.sqrt(1.0 - E2*sin1**2)
    T1 = tan1**2; C1 = EP2*cos1**2
    R1 = A*(1.0-E2)/(1.0-E2*sin1**2)**1.5
    D  = x / (N1*K0)
    lat = phi1 - (N1*tan1/R1)*(D**2/2.0
          - (5+3*T1+10*C1-4*C1**2-9*EP2)*D**4/24.0
          + (61+90*T1+298*C1+45*T1**2-252*EP2-3*C1**2)*D**6/720.0)
    lon = LON0 + (D - (1+2*T1+C1)*D**3/6.0
          + (5-2*C1+28*T1-3*C1**2+8*EP2+24*T1**2)*D**5/120.0) / cos1
    return round(math.degrees(lon), 7), round(math.degrees(lat), 7)


def reproject_ring(ring):
    return [utm_to_wgs84(x, y) for x, y in ring]

def reproject_geometry(geom):
    gt = geom['type']
    if gt == 'Polygon':
        geom['coordinates'] = [reproject_ring(r) for r in geom['coordinates']]
    elif gt == 'MultiPolygon':
        geom['coordinates'] = [[reproject_ring(r) for r in poly]
                                for poly in geom['coordinates']]
    return geom


def process_storm(prefix):
    print(f'\n{"="*60}')
    print(f'Processing: {prefix} storm')
    print(f'{"="*60}')

    # County runs store their Vista output under county_output/<ev>/
    # City runs store under <prefix>/ at the top level of SRC_DIR
    is_v2 = prefix.startswith('v2_')
    if is_v2:
        src_subdir = os.path.join(SRC_DIR, 'BG FLOOD MAP VISTA SUPERCOMPUTER', 'v2', prefix[len('v2_'):])
    elif prefix.startswith('county_'):
        ev = prefix[len('county_'):]   # e.g. "ev3"
        src_subdir = os.path.join(SRC_DIR, 'county_output', ev)
    else:
        # City runs live under "BG FLOOD MAP VISTA SUPERCOMPUTER/<folder>/" ("2 year" for the 2yr run)
        city_root = os.path.join(SRC_DIR, 'BG FLOOD MAP VISTA SUPERCOMPUTER')
        src_subdir = os.path.join(city_root, '2 year' if prefix == '2yr' else prefix)

    # Output goes into a per-storm subfolder for multi-storm support
    dst_dir = os.path.join(BASE_DST, prefix)
    if not is_v2:
        os.makedirs(dst_dir, exist_ok=True)
    # Also write to DSOC AI folder for the local server
    dsoc_dir = os.path.join(DSOC_DST, prefix)
    os.makedirs(dsoc_dir, exist_ok=True)

    found = 0
    for hour, sec_str in sorted(HOUR_SECONDS.items()):
        # City files: <prefix>_depth_<sec>s.geojson  County files: depth_<sec>s.geojson
        if prefix.startswith('county_'):
            fname = f'depth_{sec_str}s.geojson'
        else:
            fname = f'{prefix}_depth_{sec_str}s.geojson'
        src_path = os.path.join(src_subdir, fname)
        if not os.path.exists(src_path):
            print(f'  [skip] Hour {hour}: {fname} not found in {src_subdir}')
            continue
        size_mb = os.path.getsize(src_path) / 1e6
        if os.path.getsize(src_path) == 0:       # a 0-byte file is a failed conversion; an empty FeatureCollection (dry hour) is valid
            print(f'  [skip] Hour {hour}: file is 0 bytes')
            continue
        if is_v2 and os.path.exists(os.path.join(dsoc_dir, f'flood_h{hour:02d}.geojson')) and \
           os.path.getmtime(os.path.join(dsoc_dir, f'flood_h{hour:02d}.geojson')) > os.path.getmtime(src_path):
            print(f'  [done] Hour {hour}: already converted')
            found += 1
            continue

        dst_name = f'flood_h{hour:02d}.geojson'
        dst_path  = os.path.join(dst_dir, dst_name)
        dsoc_path = os.path.join(dsoc_dir, dst_name)
        print(f'  [hour {hour}] {size_mb:.1f} MB → {prefix}/{dst_name} ...', end=' ', flush=True)

        with open(src_path, 'r', encoding='utf-8') as fh:
            fc = json.load(fh)

        feats = fc.get('features', [])
        for i, feat in enumerate(feats):
            if feat.get('geometry'):
                reproject_geometry(feat['geometry'])
            p = feat.setdefault('properties', {})
            p['hour'] = hour
            p['sim_seconds'] = int(sec_str)
            p['storm'] = prefix
            tier = p.get('tier', 'shallow')
            lo, hi = TIER_DEPTH.get(tier, (0.1, 0.3))
            p['depth_min_m'] = lo
            if tier != 'severe':        # severe is open-ended (>= 1.2 m); no upper bound exists
                p['depth_max_m'] = hi
            if i % 20000 == 0 and i > 0:
                print(f'{i//1000}k..', end='', flush=True)

        data = json.dumps(fc, separators=(',', ':'))
        for path in ([dsoc_path] if is_v2 else [dst_path, dsoc_path]):
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(data)

        out_mb = os.path.getsize(dsoc_path) / 1e6
        print(f'done → {out_mb:.1f} MB  ({len(feats):,} features)')
        if is_v2 and os.environ.get('DELETE_SRC') == '1':
            os.remove(src_path)
        found += 1

    if found == 0:
        print(f'  No files found for "{prefix}" in {SRC_DIR}')
    else:
        print(f'\n  {found} hours processed → {dst_dir}')
    return found


# ── Main ──────────────────────────────────────────────────────────────────────
arg = sys.argv[1] if len(sys.argv) > 1 else '2yr'

if arg == 'all':
    for p in ALL_PREFIXES:
        process_storm(p)
elif arg == 'county_all':
    for p in ALL_COUNTY_PREFIXES:
        process_storm(p)
elif arg == 'v2_all':
    for p in ALL_V2_PREFIXES:
        process_storm(p)
else:
    valid = ALL_PREFIXES + ALL_COUNTY_PREFIXES + ALL_V2_PREFIXES
    if arg not in valid:
        print(f'Unknown prefix "{arg}".')
        print(f'  City:   {", ".join(ALL_PREFIXES)} or "all"')
        print(f'  County: {", ".join(ALL_COUNTY_PREFIXES)} or "county_all"')
        sys.exit(1)
    process_storm(arg)

print('\nAll done.')
print('Next: hard-refresh the viewer (Ctrl+F5).')
