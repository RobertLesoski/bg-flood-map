"""Export GeoPackage layers to GeoJSON for the 3D viewer."""
import numpy as np
import geopandas as gpd
from pathlib import Path

GPKG   = Path(r"C:\Users\Minec\Documents\BG_3D_Map\flood_risk.gpkg")
OUT    = Path(r"C:\Users\Minec\Documents\BG_3D_Map\layers")
UTM16N = "EPSG:32616"
OUT.mkdir(exist_ok=True)

EXPORTS = [
    # (gpkg_layer,         out_name,    simplify_m)
    ("parcels_at_risk",   "parcels",   3),
    ("roads_at_risk",     "roads",     1),
    ("fema_flood_zones",  "fema",      5),
    ("sinkhole_basins",   "sinkholes", 2),
    ("riverine_flood_zone","riverine", 5),
]

for layer, name, tol_m in EXPORTS:
    print(f"Exporting {layer}...", flush=True)
    try:
        gdf = gpd.read_file(str(GPKG), layer=layer)
    except Exception as e:
        print(f"  SKIP: {e}", flush=True)
        continue

    # Reproject to WGS84 if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Simplify in UTM (metric tolerance)
    gdf_utm = gdf.to_crs(UTM16N)
    gdf_utm.geometry = gdf_utm.geometry.simplify(tol_m, preserve_topology=True)
    gdf = gdf_utm.to_crs("EPSG:4326")

    # Precompute height_m for parcel extrusion
    if name == "parcels" and "sale_price" in gdf.columns:
        gdf["height_m"] = gdf["sale_price"].fillna(0).apply(
            lambda p: round(float(min(np.sqrt(max(float(p), 0) / 1000) * 2, 200)), 1)
        )

    # Drop null geometries
    gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)

    # Drop binary/blob columns that break GeoJSON serialization
    drop_cols = [c for c in gdf.columns if gdf[c].dtype == object
                 and c != "geometry"
                 and gdf[c].apply(lambda x: isinstance(x, (bytes, bytearray))).any()]
    if drop_cols:
        gdf = gdf.drop(columns=drop_cols)

    out_path = OUT / f"{name}.geojson"
    gdf.to_file(str(out_path), driver="GeoJSON")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  {name}.geojson  {len(gdf):,} features  {size_mb:.1f} MB", flush=True)

print("\nAll layers exported.", flush=True)
