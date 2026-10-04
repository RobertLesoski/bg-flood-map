"""
Export ALL 40,987 parcels + OSM building footprints for the 3D viewer.
- Samples min_rain.tif at each parcel centroid → parcel_min_rain_in column
- Extracts OSM building footprints → cleaner 3D shapes than parcel polygons
- Joins OSM buildings to parcels to get property values on buildings
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path

GPKG         = Path(r"C:\Users\Minec\Documents\BG_3D_Map\flood_risk.gpkg")
PBF_PATH     = Path(r"C:\Users\Minec\Documents\BG_3D_Map\kentucky-latest.osm.pbf")
MIN_RAIN_TIF = Path(r"C:\Users\Minec\Documents\BG_3D_Map\layers\min_rain.tif")
OUT_DIR      = Path(r"C:\Users\Minec\Documents\BG_3D_Map\layers")
UTM16N       = "EPSG:32616"
WGS84        = "EPSG:4326"
BG_BBOX      = (-86.536, 36.896, -86.303, 37.048)

# ── Helper: sample raster at GeoDataFrame centroids (vectorized, any CRS) ────
def sample_raster_at_centroids(gdf, tif_path, col_name, nodata_fill=999.0):
    import pyproj
    with rasterio.open(tif_path) as src:
        data  = src.read(1).astype(np.float32)
        nd    = src.nodata
        tf    = src.transform
        h, w  = data.shape
        raster_crs = src.crs.to_epsg()

    # Reproject centroids to raster CRS
    gdf_cents = gdf.copy()
    gdf_cents["_geom"] = gdf_cents.geometry.centroid
    cents_gdf = gpd.GeoDataFrame(gdf_cents[["_geom"]].rename(columns={"_geom": "geometry"}),
                                  geometry="geometry", crs=gdf.crs)
    if gdf.crs.to_epsg() != raster_crs:
        cents_gdf = cents_gdf.to_crs(epsg=raster_crs)

    xs = cents_gdf.geometry.x.values.astype(np.float64)
    ys = cents_gdf.geometry.y.values.astype(np.float64)

    # Manual affine inverse: col = (x - c) / a,  row = (y - f) / e
    cols_px = np.floor((xs - tf.c) / tf.a).astype(np.int64)
    rows_px = np.floor((ys - tf.f) / tf.e).astype(np.int64)

    in_bounds = (rows_px >= 0) & (rows_px < h) & (cols_px >= 0) & (cols_px < w)
    vals = np.full(len(gdf), nodata_fill, dtype=np.float32)

    idx = np.where(in_bounds)[0]
    if len(idx):
        sampled = data[rows_px[idx], cols_px[idx]]
        if nd is not None:
            sampled[np.isclose(sampled, nd)] = nodata_fill
        sampled[~np.isfinite(sampled)] = nodata_fill
        sampled[sampled < 0] = nodata_fill
        vals[idx] = sampled

    out = gdf.copy()
    out[col_name] = vals.tolist()
    return out

# ── 1. All parcels ────────────────────────────────────────────────────────────
print("[1] Loading all parcels...", flush=True)
gdf_p = gpd.read_file(str(GPKG), layer="parcels")
print(f"    {len(gdf_p):,} parcels", flush=True)

if gdf_p.crs and gdf_p.crs.to_epsg() != 4326:
    gdf_p = gdf_p.to_crs(WGS84)

# Compute height_m from sale_price (parcels keep value-proportional for color only)
if "sale_price" in gdf_p.columns:
    gdf_p["height_m"] = gdf_p["sale_price"].fillna(0).apply(
        lambda p: round(float(min(np.sqrt(max(float(p), 0) / 1000) * 2, 200)), 1)
    )
else:
    gdf_p["height_m"] = 5.0

# Sample flood risk at centroids
if MIN_RAIN_TIF.exists():
    print("    Sampling flood risk at parcel centroids...", flush=True)
    gdf_p = sample_raster_at_centroids(gdf_p, MIN_RAIN_TIF, "flood_at_rain_in", nodata_fill=9.0)
    gdf_p["flood_at_rain_in"] = gdf_p["flood_at_rain_in"].round(2)
else:
    gdf_p["flood_at_rain_in"] = 9.0

# Estimated damage per parcel at 2" rainfall
# FEMA guidance: ~15% structural damage per foot of flooding for residential
# We approximate: if flood_at_rain_in <= 2.0, estimate damage_pct based on how far below 2" threshold
def estimate_damage_pct(flood_at_rain, rain=2.0):
    if flood_at_rain > rain:
        return 0
    margin = rain - flood_at_rain          # how far below threshold (inches of rain "headroom")
    depth_ft = min(margin * 1.5, 6.0)     # crude flood depth proxy
    return round(min(depth_ft * 12, 80), 1)  # % damage, capped at 80%

gdf_p["damage_pct_2in"] = gdf_p["flood_at_rain_in"].apply(lambda x: estimate_damage_pct(x, 2.0))
gdf_p["damage_usd_2in"] = (gdf_p["sale_price"].fillna(0) * gdf_p["damage_pct_2in"] / 100).round(0).astype(int)

# Keep only essential fields to reduce file size
keep_cols = ["geometry", "PVA_PARCEL", "ADDRESS", "SiteAddress", "ACRES",
             "LAND_USE", "ZONING", "sale_price", "year_purchase", "Jurisdiction",
             "height_m", "flood_at_rain_in", "damage_pct_2in", "damage_usd_2in"]
keep_cols = [c for c in keep_cols if c in gdf_p.columns]
gdf_p = gdf_p[keep_cols]

# Simplify: 8m tolerance in UTM, convert back
gdf_p_utm = gdf_p.to_crs(UTM16N)
gdf_p_utm.geometry = gdf_p_utm.geometry.simplify(8, preserve_topology=True)
gdf_p = gdf_p_utm.to_crs(WGS84)
gdf_p = gdf_p[gdf_p.geometry.notna()].reset_index(drop=True)

out = OUT_DIR / "all_parcels.geojson"
gdf_p.to_file(str(out), driver="GeoJSON")
mb = out.stat().st_size / 1e6
print(f"    ✓ all_parcels.geojson  {len(gdf_p):,} features  {mb:.1f} MB", flush=True)

# Quick stats at 2" rain
at_risk = gdf_p[gdf_p["flood_at_rain_in"] <= 2.0]
total_dmg = at_risk["damage_usd_2in"].sum()
print(f"    At 2\" rain: {len(at_risk):,} parcels at risk, "
      f"est. damage ${total_dmg:,.0f}", flush=True)

# ── 2. OSM building footprints ────────────────────────────────────────────────
print("\n[2] Extracting OSM building footprints...", flush=True)
try:
    from pyrosm import OSM
    osm = OSM(str(PBF_PATH), bounding_box=list(BG_BBOX))
    bldgs = osm.get_buildings()
    if bldgs is not None and len(bldgs) > 0:
        if bldgs.crs and bldgs.crs.to_epsg() != 4326:
            bldgs = bldgs.to_crs(WGS84)

        # Keep useful OSM fields
        osm_keep = ["geometry"]
        for col in ["name", "building", "amenity", "addr:street", "addr:housenumber",
                    "height", "building:levels"]:
            if col in bldgs.columns:
                osm_keep.append(col)
        bldgs = bldgs[osm_keep]

        # Spatial join: get sale_price + ADDRESS from nearest parcel
        bldgs_utm = bldgs.to_crs(UTM16N)
        parcels_utm = gdf_p[["geometry", "sale_price", "ADDRESS", "height_m",
                              "flood_at_rain_in", "damage_pct_2in", "damage_usd_2in",
                              "LAND_USE"]].to_crs(UTM16N)
        joined = gpd.sjoin(bldgs_utm, parcels_utm, how="left", predicate="intersects")
        # Drop duplicate index columns
        joined = joined[[c for c in joined.columns if c != "index_right"]]
        joined = joined.drop_duplicates(subset=["geometry"], keep="first")
        bldgs_out = joined.to_crs(WGS84)

        # Simplify building footprints (1m tolerance)
        bldgs_utm2 = bldgs_out.to_crs(UTM16N)
        bldgs_utm2.geometry = bldgs_utm2.geometry.simplify(1, preserve_topology=True)
        bldgs_out = bldgs_utm2.to_crs(WGS84)
        bldgs_out = bldgs_out[bldgs_out.geometry.notna()].reset_index(drop=True)

        # ── Realistic physical height (OSM > type table > land_use > default) ──
        # Heights in meters; US story ≈ 3.0 m floor-to-floor
        BLDG_TYPE_H = {
            # Residential
            "house": 6.0, "detached": 6.0, "semidetached_house": 6.0,
            "residential": 6.0, "terrace": 6.0, "bungalow": 3.5,
            "cabin": 4.0, "farm": 5.0, "farmhouse": 6.0,
            # Multi-unit residential
            "apartments": 10.0, "dormitory": 16.0, "barracks": 10.0,
            "hotel": 18.0,
            # Education
            "school": 7.0, "university": 10.0, "college": 10.0,
            "kindergarten": 4.5,
            # Commercial / office
            "retail": 5.0, "commercial": 7.0, "office": 10.0,
            "supermarket": 6.0, "shop": 4.5, "kiosk": 3.5,
            "restaurant": 4.5, "fast_food": 4.5,
            # Civic / worship
            "church": 14.0, "cathedral": 25.0, "chapel": 8.0,
            "mosque": 12.0, "synagogue": 10.0, "temple": 10.0,
            "government": 10.0, "civic": 8.0, "public": 8.0,
            "fire_station": 7.0, "police": 7.0,
            # Industrial
            "industrial": 8.0, "warehouse": 9.0, "factory": 10.0,
            "storage_tank": 12.0, "silo": 20.0, "barn": 8.0,
            "greenhouse": 4.5, "hangar": 12.0,
            # Sport / entertainment
            "stadium": 20.0, "grandstand": 18.0, "sports_hall": 9.0,
            "sports_centre": 9.0,
            # Hospitals / medical
            "hospital": 16.0, "clinic": 8.0,
            # Utility / transport
            "garage": 4.0, "parking": 7.5, "carport": 3.0,
            "transportation": 6.0, "train_station": 10.0,
            "bridge": 6.0, "roof": 4.0,
            # Construction / misc
            "construction": 6.0, "ruins": 3.0,
        }

        LAND_USE_H = {
            "Residential": 6.0, "Commercial": 6.0, "Industrial": 8.0,
            "Agricultural": 5.0, "Institutional": 9.0, "Educational": 8.0,
        }

        def compute_height(row):
            # 1) OSM explicit height string (e.g. "12.5" or "12.5 m")
            h = row.get("height")
            if h is not None:
                try:
                    hv = float(str(h).replace(" m","").replace("m","").strip())
                    if hv > 0:
                        return round(min(hv, 300), 1)
                except Exception:
                    pass

            # 2) OSM building:levels
            lvl = row.get("building:levels")
            if lvl is not None:
                try:
                    lv = float(str(lvl).strip())
                    if lv > 0:
                        return round(min(lv * 3.0, 300), 1)
                except Exception:
                    pass

            # 3) Building type table
            btype = str(row.get("building") or "").lower().strip()
            if btype and btype != "yes" and btype != "nan":
                h_type = BLDG_TYPE_H.get(btype)
                if h_type:
                    return h_type

            # 4) Land use from parcel join
            lu = str(row.get("LAND_USE") or "").strip()
            for key, hv in LAND_USE_H.items():
                if key.lower() in lu.lower():
                    return hv

            # 5) Reasonable default: single-story BG structure
            return 5.5

        bldgs_out["height_m"] = bldgs_out.apply(compute_height, axis=1)

        out_b = OUT_DIR / "buildings.geojson"
        bldgs_out.to_file(str(out_b), driver="GeoJSON")
        mb_b = out_b.stat().st_size / 1e6
        print(f"    ✓ buildings.geojson  {len(bldgs_out):,} footprints  {mb_b:.1f} MB", flush=True)
    else:
        print("    No building footprints found in PBF.", flush=True)
except Exception as e:
    print(f"    OSM buildings skipped: {e}", flush=True)

print("\nDone.", flush=True)
