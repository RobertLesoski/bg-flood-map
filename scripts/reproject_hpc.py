"""
Reproject HPC flood GeoJSON files from UTM Zone 16N (EPSG:32616) → WGS84 (EPSG:4326).
Pure Python / stdlib only — no geopandas or pyproj required.
Outputs one file per hour to the BG_3D_Map layers/flood_hpc/ directory.
"""

import json
import math
import os
import glob
import sys

SRC_DIR = r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map\BG FLOOD MAP VISTA SUPERCOMPUTER"
DST_DIR = r"C:\Users\Minec\Documents\BG_3D_Map\layers\flood_hpc"

os.makedirs(DST_DIR, exist_ok=True)

# WGS84 ellipsoid + UTM Zone 16N parameters
A    = 6378137.0
F    = 1.0 / 298.257223563
B    = A * (1.0 - F)
E2   = 2.0 * F - F * F
EP2  = E2 / (1.0 - E2)
K0   = 0.9996
ZONE = 16
LON0 = math.radians((ZONE - 1) * 6 - 180 + 3)  # -87° for Zone 16N
E1   = (1.0 - math.sqrt(1.0 - E2)) / (1.0 + math.sqrt(1.0 - E2))


def utm_to_wgs84(easting, northing):
    x = easting - 500000.0
    y = northing  # northern hemisphere; no false northing removal

    M  = y / K0
    mu = M / (A * (1.0 - E2/4.0 - 3.0*E2**2/64.0 - 5.0*E2**3/256.0))

    phi1 = (mu
            + (3.0*E1/2.0 - 27.0*E1**3/32.0)        * math.sin(2.0*mu)
            + (21.0*E1**2/16.0 - 55.0*E1**4/32.0)   * math.sin(4.0*mu)
            + (151.0*E1**3/96.0)                      * math.sin(6.0*mu)
            + (1097.0*E1**4/512.0)                    * math.sin(8.0*mu))

    sin_phi1 = math.sin(phi1)
    tan_phi1 = math.tan(phi1)
    cos_phi1 = math.cos(phi1)

    N1 = A / math.sqrt(1.0 - E2 * sin_phi1**2)
    T1 = tan_phi1**2
    C1 = EP2 * cos_phi1**2
    R1 = A * (1.0 - E2) / (1.0 - E2 * sin_phi1**2)**1.5
    D  = x / (N1 * K0)

    lat = phi1 - (N1 * tan_phi1 / R1) * (
          D**2/2.0
        - (5.0 + 3.0*T1 + 10.0*C1 - 4.0*C1**2 - 9.0*EP2) * D**4/24.0
        + (61.0 + 90.0*T1 + 298.0*C1 + 45.0*T1**2 - 252.0*EP2 - 3.0*C1**2) * D**6/720.0)

    lon = LON0 + (
          D
        - (1.0 + 2.0*T1 + C1) * D**3/6.0
        + (5.0 - 2.0*C1 + 28.0*T1 - 3.0*C1**2 + 8.0*EP2 + 24.0*T1**2) * D**5/120.0
    ) / cos_phi1

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


# Hour → seconds mapping (only the "clean" hourly files)
HOUR_SECONDS = {
    1: '003600', 2: '007200', 3: '010800', 4: '014400',
    5: '018000', 6: '021600', 7: '025200', 8: '028800', 9: '032400',
}

for hour, sec_str in sorted(HOUR_SECONDS.items()):
    src_path = os.path.join(SRC_DIR, f'2yr_depth_{sec_str}s.geojson')
    if not os.path.exists(src_path):
        print(f'  [skip] Hour {hour} not found: {src_path}')
        continue

    size_mb = os.path.getsize(src_path) / 1e6
    if size_mb < 0.01:
        print(f'  [skip] Hour {hour} is empty ({size_mb:.3f} MB)')
        continue

    dst_path = os.path.join(DST_DIR, f'flood_h{hour:02d}.geojson')
    print(f'[{hour}/9] Hour {hour} ({size_mb:.1f} MB) → {os.path.basename(dst_path)} ...', end=' ', flush=True)

    with open(src_path, 'r', encoding='utf-8') as fh:
        fc = json.load(fh)

    feats = fc.get('features', [])
    total = len(feats)
    for i, feat in enumerate(feats):
        if feat.get('geometry'):
            reproject_geometry(feat['geometry'])
        # Enrich properties with hour info
        p = feat.setdefault('properties', {})
        p['hour'] = hour
        p['sim_seconds'] = int(sec_str)
        # Depth ranges per tier (meters)
        tier = p.get('tier', 'shallow')
        p['depth_min_m'] = {'shallow': 0.1, 'moderate': 0.3, 'deep': 0.6, 'severe': 1.2}.get(tier, 0.1)
        p['depth_max_m'] = {'shallow': 0.3, 'moderate': 0.6, 'deep': 1.2, 'severe': 5.0}.get(tier, 0.3)
        if i % 20000 == 0 and i > 0:
            print(f'{i//1000}k..', end='', flush=True)

    with open(dst_path, 'w', encoding='utf-8') as fh:
        json.dump(fc, fh, separators=(',', ':'))

    out_mb = os.path.getsize(dst_path) / 1e6
    print(f'done → {out_mb:.1f} MB  ({total:,} features)')

print('\nAll done. Files written to:', DST_DIR)
