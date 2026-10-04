"""
Step 3: Data Integration — Bowling Green, KY
Downloads and joins three authoritative datasets into flood_risk.gpkg:
  1. FEMA NFHL flood zones       → fema_flood_zones
  2. Warren County PVA parcels   → parcels, parcels_at_risk
  3. KyFromAbove 2ft DEM         → dem_bg.tif (GeoTIFF)

Sources:
  FEMA NFHL   : https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28
  Parcels     : https://webgis.bgky.org/server/rest/services/WARCO/Parcel_Reference/MapServer/0
  DEM         : https://kyraster.ky.gov/arcgis/rest/services/ElevationServices/Ky_DEM_KYAPED_2FT_Phase3/ImageServer
"""

import json
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
OUTPUT     = SCRIPT_DIR / "flood_risk.gpkg"
DEM_OUT    = SCRIPT_DIR / "dem_bg.tif"

BG_BBOX  = (-86.536, 36.896, -86.303, 37.048)   # (minx, miny, maxx, maxy) WGS84
UTM16N   = "EPSG:32616"
WGS84    = "EPSG:4326"

FEMA_URL    = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
PARCEL_URL  = "https://webgis.bgky.org/server/rest/services/WARCO/Parcel_Reference/MapServer/0/query"
DEM_URLS    = [
    "https://kyraster.ky.gov/arcgis/rest/services/ElevationServices/Ky_DEM_KYAPED_2FT_Phase3/ImageServer/exportImage",
    "https://kyraster.ky.gov/arcgis/rest/services/ElevationServices/Ky_DEM_KYAPED_2FT_Phase2/ImageServer/exportImage",
    "https://kyraster.ky.gov/arcgis/rest/services/ElevationServices/Ky_DEM_KYAPED_5FT_WGS84WM/ImageServer/exportImage",
]

PAGE_SIZE   = 1000
SESSION     = requests.Session()
SESSION.headers.update({"User-Agent": "BG-3D-Map/1.0"})

print("=" * 60)
print("BG Step 3 — Data Integration")
print("=" * 60)


# ── Helpers ────────────────────────────────────────────────────────────────────
def bbox_envelope(minx, miny, maxx, maxy):
    return json.dumps({"xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                        "spatialReference": {"wkid": 4326}})

def fetch_page(url, params, offset):
    params = dict(params, resultOffset=offset, resultRecordCount=PAGE_SIZE)
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def paginate_features(url, base_params, label):
    """Paginate an ArcGIS REST query, return list of GeoJSON-compatible feature dicts."""
    all_features = []
    offset = 0
    while True:
        data = fetch_page(url, base_params, offset)
        feats = data.get("features", [])
        all_features.extend(feats)
        print(f"  {label}: {len(all_features):,} fetched...", flush=True)
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.2)
    return all_features


# ── Phase 1: FEMA NFHL Flood Zones ────────────────────────────────────────────
print("\n[Phase 1] FEMA NFHL flood zones...", flush=True)
t0 = time.time()

fema_params = {
    "geometry":       bbox_envelope(*BG_BBOX),
    "geometryType":   "esriGeometryEnvelope",
    "inSR":           "4326",
    "spatialRel":     "esriSpatialRelIntersects",
    "outFields":      "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DFIRM_ID",
    "returnGeometry": "true",
    "outSR":          "4326",
    "f":              "geojson",
}

try:
    fema_feats = paginate_features(FEMA_URL, fema_params, "FEMA zones")
    if fema_feats:
        gdf_fema = gpd.GeoDataFrame.from_features(fema_feats, crs=WGS84)
        # Clip to BG bbox
        from shapely.geometry import box as sbox
        gdf_fema = gdf_fema[gdf_fema.geometry.intersects(
            sbox(*BG_BBOX)
        )].reset_index(drop=True)
        print(f"  ✓ FEMA zones: {len(gdf_fema):,}  ({time.time()-t0:.1f}s)", flush=True)
        print(gdf_fema["FLD_ZONE"].value_counts().to_string())
    else:
        print("  ! No FEMA features returned — check bbox/service availability")
        gdf_fema = None
except Exception as e:
    print(f"  ✗ FEMA download failed: {e}")
    gdf_fema = None


# ── Phase 2: Warren County Parcels ────────────────────────────────────────────
print("\n[Phase 2] Warren County PVA parcels...", flush=True)
t0 = time.time()

PARCEL_FIELDS = ",".join([
    "PVA_PARCEL", "ADDRESS", "SiteAddress", "full_property_add",
    "ZONING", "LAND_USE", "ACRES", "calc_acres",
    "sale_price", "year_purchase", "sale_code",
    "subdivision", "Jurisdiction", "district", "class",
])

parcel_params = {
    "geometry":       bbox_envelope(*BG_BBOX),
    "geometryType":   "esriGeometryEnvelope",
    "inSR":           "4326",
    "spatialRel":     "esriSpatialRelIntersects",
    "outFields":      PARCEL_FIELDS,
    "returnGeometry": "true",
    "outSR":          "4326",
    "f":              "geojson",
}

try:
    parcel_feats = paginate_features(PARCEL_URL, parcel_params, "parcels")
    if parcel_feats:
        gdf_parcels = gpd.GeoDataFrame.from_features(parcel_feats, crs=WGS84)
        gdf_parcels = gdf_parcels[gdf_parcels.geometry.notna()].reset_index(drop=True)
        # Clean sale_price: coerce to numeric
        gdf_parcels["sale_price"] = pd.to_numeric(gdf_parcels["sale_price"], errors="coerce")
        print(f"  ✓ Parcels: {len(gdf_parcels):,}  ({time.time()-t0:.1f}s)", flush=True)
        valid_prices = gdf_parcels["sale_price"].dropna()
        if len(valid_prices):
            print(f"  sale_price: min=${valid_prices.min():,.0f}  median=${valid_prices.median():,.0f}  max=${valid_prices.max():,.0f}")
    else:
        print("  ! No parcel features returned")
        gdf_parcels = None
except Exception as e:
    print(f"  ✗ Parcel download failed: {e}")
    gdf_parcels = None


# ── Phase 3: KyFromAbove DEM ───────────────────────────────────────────────────
print("\n[Phase 3] KyFromAbove 2ft DEM download...", flush=True)
t0 = time.time()

# Request ~5m resolution GeoTIFF (manageable size for web viz)
# BG area ~20km x 17km → 4096x3500 pixels at ~5m
dem_params = {
    "bbox":         f"{BG_BBOX[0]},{BG_BBOX[1]},{BG_BBOX[2]},{BG_BBOX[3]}",
    "bboxSR":       4326,
    "size":         "4096,3500",
    "imageSR":      32616,
    "format":       "tiff",
    "pixelType":    "F32",
    "noData":       -9999,
    "f":            "image",
}

dem_ok = False
for dem_url in DEM_URLS:
    phase = "Phase3" if "Phase3" in dem_url else "Phase2"
    print(f"  Trying {phase}...", flush=True)
    try:
        r = SESSION.get(dem_url, params=dem_params, timeout=180, stream=True)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "tiff" in content_type or "image" in content_type:
            with open(DEM_OUT, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_mb = DEM_OUT.stat().st_size / 1e6
            if size_mb < 0.1:
                print(f"  ✗ {phase} returned {size_mb:.2f} MB (no coverage for this area)", flush=True)
                DEM_OUT.unlink(missing_ok=True)
            else:
                print(f"  ✓ DEM saved: {size_mb:.1f} MB → {DEM_OUT}  ({time.time()-t0:.1f}s)", flush=True)
                dem_ok = True
                break
        else:
            print(f"  ✗ Unexpected content-type: {content_type[:80]}", flush=True)
    except Exception as e:
        print(f"  ✗ {phase} failed: {e}", flush=True)

if not dem_ok:
    print("  ! DEM download failed. Continuing without DEM.", flush=True)


# ── Phase 4: Spatial join — parcels × FEMA zones ──────────────────────────────
if gdf_parcels is not None and gdf_fema is not None:
    print("\n[Phase 4] Joining parcels to FEMA flood zones...", flush=True)
    t0 = time.time()
    gdf_p_utm = gdf_parcels.to_crs(UTM16N)
    gdf_f_utm = gdf_fema.to_crs(UTM16N)

    joined_fema = gpd.sjoin(
        gdf_p_utm,
        gdf_f_utm[["FLD_ZONE", "SFHA_TF", "geometry"]],
        how="left",
        predicate="intersects",
    )
    # Keep worst (SFHA = Special Flood Hazard Area) if multiple hits
    joined_fema = (
        joined_fema.sort_values("SFHA_TF", ascending=False)
        .groupby(joined_fema.index, sort=False)
        .first()
        .drop(columns=["index_right"], errors="ignore")
    )
    gdf_parcels_joined = gpd.GeoDataFrame(joined_fema, geometry="geometry", crs=UTM16N)
    sfha_count = (gdf_parcels_joined["SFHA_TF"] == "T").sum()
    print(f"  Parcels in SFHA (Special Flood Hazard Area): {sfha_count:,}")
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
else:
    gdf_parcels_joined = gdf_parcels.to_crs(UTM16N) if gdf_parcels is not None else None


# ── Phase 5: Spatial join — parcels × sinkhole basins ─────────────────────────
if gdf_parcels_joined is not None:
    print("\n[Phase 5] Joining parcels to sinkhole basins...", flush=True)
    t0 = time.time()
    gdf_sinkholes = gpd.read_file(str(OUTPUT), layer="sinkhole_basins").to_crs(UTM16N)
    joined_sink = gpd.sjoin(
        gdf_parcels_joined,
        gdf_sinkholes[["risk_tier", "geometry"]].rename(columns={"risk_tier": "sinkhole_risk"}),
        how="left",
        predicate="intersects",
    )
    TIER_ORDER = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
    joined_sink["_tr"] = joined_sink["sinkhole_risk"].map(TIER_ORDER).fillna(99)
    gdf_parcels_final = (
        joined_sink.sort_values("_tr")
        .groupby(joined_sink.index, sort=False)
        .first()
        .drop(columns=["_tr", "index_right"], errors="ignore")
    )
    gdf_parcels_final = gpd.GeoDataFrame(gdf_parcels_final, geometry="geometry", crs=UTM16N)
    sink_count = gdf_parcels_final["sinkhole_risk"].notna().sum()
    print(f"  Parcels overlapping sinkhole basins: {sink_count:,}")
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Subset at-risk parcels (guard for columns that may be absent if upstream phase failed)
    mask = gdf_parcels_final["sinkhole_risk"].notna()
    if "FLD_ZONE" in gdf_parcels_final.columns:
        mask = mask | gdf_parcels_final["FLD_ZONE"].notna()
    gdf_at_risk = gdf_parcels_final[mask].copy()
else:
    gdf_parcels_final = None
    gdf_at_risk = None


# ── Phase 6: Write to GeoPackage ──────────────────────────────────────────────
print("\n[Phase 6] Writing layers to GeoPackage...", flush=True)
layers_written = 0

if gdf_fema is not None:
    gdf_fema.to_file(str(OUTPUT), layer="fema_flood_zones", driver="GPKG")
    print(f"  ✓ fema_flood_zones              {len(gdf_fema):>6,} features")
    layers_written += 1

if gdf_parcels_final is not None:
    gdf_parcels_final.to_crs(WGS84).to_file(str(OUTPUT), layer="parcels", driver="GPKG")
    print(f"  ✓ parcels                       {len(gdf_parcels_final):>6,} features")
    layers_written += 1

if gdf_at_risk is not None:
    gdf_at_risk.to_crs(WGS84).to_file(str(OUTPUT), layer="parcels_at_risk", driver="GPKG")
    print(f"  ✓ parcels_at_risk               {len(gdf_at_risk):>6,} features")
    layers_written += 1


print(f"\n{layers_written} new layers added to: {OUTPUT}")
if dem_ok:
    print(f"DEM saved to: {DEM_OUT}")
print("\nAll data ready for Step 4 (3D visualization pipeline).")
