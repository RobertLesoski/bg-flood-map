"""Fetch FEMA NFHL flood zones via OBJECTID batching (workaround for server query bug)."""
import json, time, warnings
from pathlib import Path
import geopandas as gpd
import requests
from shapely.geometry import box as sbox, shape

warnings.filterwarnings("ignore")

OUTPUT  = Path(__file__).parent / "flood_risk.gpkg"
WGS84   = "EPSG:4326"
BG_BBOX = (-86.536, 36.896, -86.303, 37.048)
BASE    = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
BATCH   = 100

env = json.dumps({"xmin": BG_BBOX[0], "ymin": BG_BBOX[1],
                  "xmax": BG_BBOX[2], "ymax": BG_BBOX[3],
                  "spatialReference": {"wkid": 4326}}, separators=(",", ":"))

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 BG-3D-Map/1.0"

# ── Step 1: get all OBJECTIDs in BG bbox ──────────────────────────────────────
print("Step 1: fetching OBJECTIDs in BG bbox...", flush=True)
r = session.post(BASE, data={
    "geometry": env, "geometryType": "esriGeometryEnvelope",
    "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
    "returnIdsOnly": "true", "f": "json",
}, timeout=60)
r.raise_for_status()
obj_ids = r.json().get("objectIds", [])
print(f"  Found {len(obj_ids):,} features in bbox", flush=True)

if not obj_ids:
    print("No features — check bbox.")
    raise SystemExit(1)

# ── Step 2: fetch features in batches by OBJECTID ─────────────────────────────
print(f"Step 2: fetching features in batches of {BATCH}...", flush=True)
all_features = []
def fetch_batch(ids, timeout=60):
    """Fetch features for a list of OBJECTIDs; returns list of features or None on failure."""
    where = f"OBJECTID IN ({','.join(str(x) for x in ids)})"
    r = session.post(BASE, data={
        "where": where,
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DFIRM_ID",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }, timeout=timeout)
    if not r.ok:
        return None
    data = r.json()
    if "error" in data:
        return None
    return data.get("features", [])

for i in range(0, len(obj_ids), BATCH):
    batch = obj_ids[i:i + BATCH]
    feats = fetch_batch(batch)
    if feats is None:
        # Retry in two halves
        print(f"  Batch {i//BATCH+1} failed — retrying as two halves...", flush=True)
        mid = len(batch) // 2
        for half in [batch[:mid], batch[mid:]]:
            hfeats = fetch_batch(half)
            if hfeats is None:
                # Last resort: one at a time
                print(f"    Half failed — trying individually ({len(half)} features)...", flush=True)
                for oid in half:
                    single = fetch_batch([oid])
                    if single:
                        all_features.extend(single)
                    time.sleep(0.05)
            else:
                all_features.extend(hfeats)
            time.sleep(0.2)
    else:
        all_features.extend(feats)
    print(f"  {len(all_features):,} / {len(obj_ids):,} fetched...", flush=True)
    time.sleep(0.1)

print(f"\n✓ Total features: {len(all_features):,}", flush=True)
if not all_features:
    print("No features collected.")
    raise SystemExit(1)

# ── Build GeoDataFrame ─────────────────────────────────────────────────────────
gdf = gpd.GeoDataFrame.from_features(all_features, crs=WGS84)
gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
print(f"✓ Valid geometries: {len(gdf):,}")
print(gdf["FLD_ZONE"].value_counts().to_string())

gdf.to_file(str(OUTPUT), layer="fema_flood_zones", driver="GPKG")
print(f"\nWritten → {OUTPUT}  layer: fema_flood_zones")
