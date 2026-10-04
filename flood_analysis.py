"""
Step 2: Hydrology & Flood Analysis — Bowling Green, KY
Produces: flood_risk.gpkg with layers:
  - sinkhole_basins        : polygonized karst depression contours
  - riverine_flood_zone    : Barren River / Drakes Creek low-elevation buffer
  - contours_depression    : raw depression LineStrings (for viz)
  - contours_low_elev      : standard contours <=450 ft (for viz)
  - roads                  : full OSM drive network
  - roads_at_risk          : road segments intersecting either flood zone

CRS pipeline:
  Input  : EPSG:4326  (WGS84, lon/lat)
  Working: EPSG:32616 (UTM Zone 16N, meters) for metric operations
  Output : EPSG:4326  (back to lon/lat for web viz)
"""

import sys
import fiona
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import shape
from shapely.ops import polygonize, unary_union
import warnings
import time

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
GEOJSON   = r"C:\AI_RecycleBin\{A1BC5211-A7CC-498C-9C4A-1D655B883451}\1\Downloads\BGKY_Contours_2FT_2004_Depression.geojson"
OUTPUT    = r"C:\Users\Minec\Documents\BG_3D_Map\flood_risk.gpkg"

DEPR_TAGS = {"Topo-Minr-Depr", "Topo-Majr-Depr",
             "Topo-Minr-Depr-Clip", "Topo-Majr-Depr-Clip"}

RIVERINE_ELEV_THRESHOLD = 450
MIN_BASIN_AREA_M2       = 200
BASIN_BUFFER_M          = 8
RIVERINE_BUFFER_M       = 200

UTM16N = "EPSG:32616"
WGS84  = "EPSG:4326"

BG_BBOX = (-86.536, 36.896, -86.303, 37.048)

# Alternative Overpass API endpoints — tried in order
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.karte.io/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

print("=" * 60)
print("BG Flood Analysis — Step 2")
print("=" * 60)

# ── Phase 1: Stream GeoJSON ───────────────────────────────────────────────────
print("\n[Phase 1] Streaming GeoJSON...")
t0 = time.time()

depr_rows = []
low_rows  = []

with fiona.open(GEOJSON) as src:
    total = len(src)
    for i, feat in enumerate(src):
        if i % 50000 == 0:
            print(f"  ...{i:,}/{total:,}", flush=True)
        props = feat["properties"]
        layer = props.get("Layer", "")
        elev  = props.get("Elevation", 9999)
        geom_d = feat["geometry"]
        if geom_d is None:
            continue
        geom = shape(geom_d)

        if layer in DEPR_TAGS:
            depr_rows.append({"elevation": elev, "layer": layer, "geometry": geom})
        elif elev <= RIVERINE_ELEV_THRESHOLD:
            low_rows.append({"elevation": elev, "layer": layer, "geometry": geom})

print(f"  Done in {time.time()-t0:.1f}s", flush=True)
print(f"  Depression contours : {len(depr_rows):,}")
print(f"  Low-elev contours   : {len(low_rows):,}")

gdf_depr = gpd.GeoDataFrame(depr_rows, crs=WGS84).to_crs(UTM16N)
gdf_low  = gpd.GeoDataFrame(low_rows,  crs=WGS84).to_crs(UTM16N)

# ── Phase 2: Polygonize depression contours ───────────────────────────────────
print("\n[Phase 2] Polygonizing sinkhole basins...", flush=True)
t0 = time.time()

def _risk_tier(elev):
    if elev <= 430: return "critical"
    if elev <= 470: return "high"
    if elev <= 510: return "moderate"
    return "low"

basin_records = []
for elev, group in gdf_depr.groupby("elevation"):
    merged = unary_union(list(group.geometry))
    for poly in polygonize(merged):
        area = poly.area
        if area < MIN_BASIN_AREA_M2:
            continue
        basin_records.append({
            "elevation_ft": elev,
            "area_m2":      round(area, 1),
            "risk_tier":    _risk_tier(elev),
            "geometry":     poly.buffer(BASIN_BUFFER_M),
        })

gdf_basins = gpd.GeoDataFrame(basin_records, crs=UTM16N)
print(f"  Sinkhole basins : {len(gdf_basins):,}")
print(gdf_basins.risk_tier.value_counts().to_string())
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

# ── Phase 3: Riverine flood zone ──────────────────────────────────────────────
print(f"\n[Phase 3] Riverine zone (contours ≤ {RIVERINE_ELEV_THRESHOLD} ft)...", flush=True)
t0 = time.time()

buffered     = gdf_low.geometry.buffer(RIVERINE_BUFFER_M)
flood_union  = unary_union(buffered)
gdf_riverine = gpd.GeoDataFrame(
    [{"zone_type": "riverine_lowland",
      "elev_threshold_ft": RIVERINE_ELEV_THRESHOLD,
      "buffer_m": RIVERINE_BUFFER_M,
      "risk_tier": "critical",
      "geometry": flood_union}],
    crs=UTM16N,
).explode(index_parts=False).reset_index(drop=True)

print(f"  Zone polygons : {len(gdf_riverine):,}")
print(f"  Total area    : {gdf_riverine.geometry.area.sum()/1e6:.2f} km²")
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

# ── Save terrain layers now — before network calls ────────────────────────────
print("\n[Saving terrain layers to GeoPackage]...", flush=True)
terrain_layers = {
    "sinkhole_basins":     gdf_basins.to_crs(WGS84),
    "riverine_flood_zone": gdf_riverine.to_crs(WGS84),
    "contours_depression": gdf_depr.to_crs(WGS84),
    "contours_low_elev":   gdf_low.to_crs(WGS84),
}
for name, gdf in terrain_layers.items():
    gdf.to_file(OUTPUT, layer=name, driver="GPKG")
    print(f"  ✓ {name:30s}  {len(gdf):>6,} features")

# ── Phase 4: Download OSM roads (try multiple endpoints) ─────────────────────
print("\n[Phase 4] Downloading OSM road network...", flush=True)

gdf_roads = None
for endpoint in OVERPASS_ENDPOINTS:
    print(f"  Trying {endpoint} ...", flush=True)
    try:
        ox.settings.overpass_url = endpoint
        ox.settings.timeout = 120
        G = ox.graph_from_bbox(
            bbox=(BG_BBOX[3], BG_BBOX[1], BG_BBOX[2], BG_BBOX[0]),
            network_type="drive",
            retain_all=False,
        )
        _, gdf_edges = ox.graph_to_gdfs(G)
        gdf_roads = gdf_edges[["name", "highway", "length", "geometry"]].copy()
        gdf_roads = gdf_roads.reset_index(drop=True).to_crs(UTM16N)
        print(f"  ✓ Road segments : {len(gdf_roads):,}")
        break
    except Exception as e:
        print(f"  ✗ Failed: {e.__class__.__name__}: {e}", flush=True)

if gdf_roads is None:
    print("\n  All Overpass endpoints failed.")
    print("  Terrain layers saved. Skipping road analysis.")
    print(f"  Re-run when internet is available, or download OSM PBF manually:")
    print(f"    https://download.geofabrik.de/north-america/us/kentucky-latest.osm.pbf")
    print(f"\nPartial output (terrain only): {OUTPUT}")
    sys.exit(0)

# ── Phase 5: Spatial join roads × flood zones ─────────────────────────────────
print("\n[Phase 5] Tagging roads intersecting flood zones...", flush=True)
t0 = time.time()

TIER_ORDER = {"critical": 0, "high": 1, "moderate": 2, "low": 3}

gdf_all_risk = pd.concat([
    gdf_basins[["risk_tier", "geometry"]].assign(zone_type="sinkhole"),
    gdf_riverine[["risk_tier", "geometry"]].assign(zone_type="riverine"),
], ignore_index=True).to_crs(UTM16N)

joined = gpd.sjoin(
    gdf_roads,
    gdf_all_risk[["risk_tier", "zone_type", "geometry"]],
    how="inner",
    predicate="intersects",
)
joined["_tier_rank"] = joined["risk_tier"].map(TIER_ORDER)
gdf_roads_risk = (
    joined.sort_values("_tier_rank")
          .groupby(joined.index, sort=False)
          .first()
          .drop(columns=["_tier_rank", "index_right"])
          .reset_index(drop=True)
)

print(f"  Roads at risk : {len(gdf_roads_risk):,} of {len(gdf_roads):,}")
print(gdf_roads_risk["risk_tier"].value_counts().to_string())
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

# ── Phase 6: Write road layers ────────────────────────────────────────────────
print("\n[Phase 6] Writing road layers...", flush=True)
road_layers = {
    "roads":         gdf_roads.to_crs(WGS84),
    "roads_at_risk": gdf_roads_risk.to_crs(WGS84),
}
for name, gdf in road_layers.items():
    gdf.to_file(OUTPUT, layer=name, driver="GPKG")
    print(f"  ✓ {name:30s}  {len(gdf):>6,} features")

print(f"\nComplete output: {OUTPUT}")
print("All 6 layers ready for Step 3 (parcel join) and Step 4 (3D viz).")
