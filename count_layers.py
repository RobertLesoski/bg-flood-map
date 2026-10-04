"""Count unique Layer values and elevation histogram — fast fiona stream."""
import fiona
from collections import Counter

GEOJSON_PATH = r"C:\AI_RecycleBin\{A1BC5211-A7CC-498C-9C4A-1D655B883451}\1\Downloads\BGKY_Contours_2FT_2004_Depression.geojson"

layer_counts = Counter()
elev_counts = Counter()

with fiona.open(GEOJSON_PATH) as src:
    for feat in src:
        p = feat['properties']
        layer_counts[p.get('Layer', '(null)')] += 1
        elev_counts[p.get('Elevation', -9999)] += 1

print("\n[Unique Layer values and feature counts]")
for layer, cnt in sorted(layer_counts.items(), key=lambda x: -x[1]):
    print(f"  {cnt:>8,}  {layer}")

print(f"\n[Elevation histogram — every 10 ft band]")
bands = {}
for elev, cnt in elev_counts.items():
    band = (elev // 10) * 10
    bands[band] = bands.get(band, 0) + cnt
for band in sorted(bands.keys()):
    bar = '#' * (bands[band] // 500)
    print(f"  {band:>4} ft: {bands[band]:>6,}  {bar}")

print(f"\n[Lowest 10 elevation values — karst floor / flood zones]")
for elev in sorted(elev_counts.keys())[:10]:
    print(f"  {elev} ft: {elev_counts[elev]:,} contour lines")
