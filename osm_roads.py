"""
Phase 4–6 roads: read local OSM PBF → clip to BG bbox → spatial join with flood zones
Appends 'roads' and 'roads_at_risk' layers to the existing flood_risk.gpkg

Usage:
    python osm_roads.py [path_to_kentucky-latest.osm.pbf]

If no path given, looks for kentucky-latest.osm.pbf in the same folder as this script.
"""

import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
OUTPUT     = SCRIPT_DIR / "flood_risk.gpkg"

PBF_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "kentucky-latest.osm.pbf"

BG_BBOX    = (-86.536, 36.896, -86.303, 37.048)   # (minx, miny, maxx, maxy)
UTM16N     = "EPSG:32616"
WGS84      = "EPSG:4326"
TIER_ORDER = {"critical": 0, "high": 1, "moderate": 2, "low": 3}

# ── Validate inputs ────────────────────────────────────────────────────────────
if not PBF_PATH.exists():
    print(f"ERROR: PBF not found: {PBF_PATH}")
    print("Download from: https://download.geofabrik.de/north-america/us/kentucky-latest.osm.pbf")
    sys.exit(1)

if not OUTPUT.exists():
    print(f"ERROR: {OUTPUT} not found — run flood_analysis.py first to generate terrain layers.")
    sys.exit(1)

print("=" * 60)
print("BG Roads — Phase 4–6 (local PBF)")
print("=" * 60)
print(f"  PBF   : {PBF_PATH}")
print(f"  Output: {OUTPUT}")

# ── Phase 4: Extract roads from PBF ───────────────────────────────────────────
print("\n[Phase 4] Extracting roads from PBF...", flush=True)
t0 = time.time()

try:
    from pyrosm import OSM
except ImportError:
    print("ERROR: pyrosm not installed.")
    print("  pip install pyrosm")
    sys.exit(1)

# pyrosm bbox: (minx, miny, maxx, maxy) same as shapely
osm = OSM(str(PBF_PATH), bounding_box=list(BG_BBOX))

# network_type="driving" returns edges GeoDataFrame
gdf_roads_raw = osm.get_network(network_type="driving")

if gdf_roads_raw is None or len(gdf_roads_raw) == 0:
    print("ERROR: No road features extracted. Check bbox or PBF coverage.")
    sys.exit(1)

# Keep useful columns (pyrosm names differ from osmnx)
keep = [c for c in ["name", "highway", "length", "geometry"] if c in gdf_roads_raw.columns]
gdf_roads = gdf_roads_raw[keep].copy().reset_index(drop=True)

# Ensure CRS is WGS84 (pyrosm sets it automatically)
if gdf_roads.crs is None:
    gdf_roads = gdf_roads.set_crs(WGS84)
elif gdf_roads.crs.to_epsg() != 4326:
    gdf_roads = gdf_roads.to_crs(WGS84)

gdf_roads_utm = gdf_roads.to_crs(UTM16N)
print(f"  ✓ Road segments : {len(gdf_roads_utm):,}  ({time.time()-t0:.1f}s)", flush=True)

# ── Phase 5: Spatial join roads × flood zones ──────────────────────────────────
print("\n[Phase 5] Loading flood zones & tagging roads...", flush=True)
t0 = time.time()

gdf_basins   = gpd.read_file(OUTPUT, layer="sinkhole_basins").to_crs(UTM16N)
gdf_riverine = gpd.read_file(OUTPUT, layer="riverine_flood_zone").to_crs(UTM16N)

gdf_all_risk = pd.concat([
    gdf_basins[["risk_tier", "geometry"]].assign(zone_type="sinkhole"),
    gdf_riverine[["risk_tier", "geometry"]].assign(zone_type="riverine"),
], ignore_index=True)

joined = gpd.sjoin(
    gdf_roads_utm,
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
gdf_roads_risk = gpd.GeoDataFrame(gdf_roads_risk, geometry="geometry").set_crs(UTM16N, allow_override=True)

print(f"  Roads at risk : {len(gdf_roads_risk):,} of {len(gdf_roads_utm):,}", flush=True)
print(gdf_roads_risk["risk_tier"].value_counts().to_string())
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

# ── Phase 6: Append road layers to GeoPackage ─────────────────────────────────
print("\n[Phase 6] Writing road layers...", flush=True)
road_layers = {
    "roads":         gdf_roads_utm.to_crs(WGS84),
    "roads_at_risk": gdf_roads_risk.to_crs(WGS84),
}
for name, gdf in road_layers.items():
    gdf.to_file(str(OUTPUT), layer=name, driver="GPKG")
    print(f"  ✓ {name:30s}  {len(gdf):>6,} features", flush=True)

print(f"\nComplete output: {OUTPUT}")
print("All 6 layers ready for Step 3 (parcel join) and Step 4 (3D viz).")
