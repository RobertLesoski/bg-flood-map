# Implementation Notes — BG 3D Flood Map v2

_Last updated: October 2026_

This document records what was added, what was already present, and what could not be fully implemented in-browser without additional infrastructure.

---

## Added in This Release

| Feature | File(s) | Status |
|---|---|---|
| TypeScript/Vite scaffold | `app/package.json`, `app/tsconfig.json`, `app/vite.config.ts` | ✅ Complete |
| Flood elevation slider (feet MSL) | `app/src/main.ts` — `#elev-ctrl` | ✅ Complete |
| Water shader RGBA(0,105,148,0.65) | `app/src/main.ts` — `WATER_COLOR` | ✅ Complete |
| Building vulnerability colors (grey/amber/crimson) | `app/src/main.ts` — `buildingVulnColor()` | ✅ Complete |
| FEMA 100-yr (AE/A) vs 500-yr (X) distinct layers | `app/src/main.ts` — `addFemaLayers()` | ✅ Complete |
| Sinkhole depth coloring by risk tier | `app/src/main.ts` — `SINK_COLOR` | ✅ Complete |
| Per-layer opacity sliders | Sidebar `[data-opacity-layer]` / `[data-opacity-group]` | ✅ Complete |
| Hillshade layer with multi-direction selector | `app/src/main.ts` — `addHillshadeLayer()` | ✅ Complete |
| Max depth stat in header | `#stat-depth` | ✅ Complete |
| Water depth badge in inspection modal | `showInfo()` — `depthRow` | ✅ Complete |
| Basemap selector (OSM / Satellite / Dark) | Sidebar `#basemap-sel` | ✅ Complete |
| Geolocate control | `map.addControl(GeolocateControl)` | ✅ Complete |
| Exaggeration range corrected to 1.0–3.0×, default 1.75× | `app/src/main.ts` | ✅ Complete |
| Python DEM preprocessing pipeline | `scripts/dem_preprocess.py` | ✅ Complete |
| Sinkhole depth, rim elevation, volume enrichment | `scripts/dem_preprocess.py` Step 6 | ✅ Complete |
| TWI computation (pysheds or Gaussian fallback) | `scripts/dem_preprocess.py` Step 3 | ✅ Complete |
| Multi-directional hillshade generation (GDAL + numpy fallback) | `scripts/dem_preprocess.py` Step 4 | ✅ Complete |
| Terrain-RGB mbtiles generation | `scripts/dem_preprocess.py` Step 5 | ✅ Complete (requires rio-rgbify) |

---

## Already Implemented (viewer.html — Skipped to Avoid Duplication)

These features exist in the original `viewer.html` and are replicated in the new `app/`:

- 3D terrain mesh from AWS Terrarium tiles
- Navigation controls (pan, zoom, pitch 0–80°, rotation, compass)
- Vertical exaggeration slider (fixed to 1.0–3.0× in new app)
- Rainfall simulation slider (0–3")
- FEMA NFHL flood zone overlay
- Karst sinkhole layer
- Riverine flood zones
- At-risk roads (color-coded by elevation tier)
- 3D extruded OSM building footprints
- All-parcels flat fill layer
- Layer visibility toggles
- Feature inspection modal with terrain elevation query
- Stats bar (buildings/parcels at risk)

---

## Not Implemented (Requires External Infrastructure)

### 1. KyFromAbove Terrain-RGB Tile Server
**Requested:** _"Mapbox Terrain-RGB or Terrarium elevation raster tiles derived from KyFromAbove (KYAPED) 5-ft LiDAR DEMs"_

**Status:** ⚠️ Preprocessing script written; serving infrastructure not set up.

The app currently uses the free AWS Terrarium tiles (`elevation-tiles-prod` S3 bucket) which cover the area at 1–10m resolution. To switch to KyFromAbove 2-ft LiDAR tiles:

1. Run `scripts/dem_preprocess.py` — it generates `layers/dem_meters.tif` and calls `rio-rgbify` to produce `layers/terrain_rgb.mbtiles`
2. Install tileserver-gl: `npm install -g tileserver-gl`
3. Serve: `tileserver-gl layers/terrain_rgb.mbtiles --port 8080`
4. In `app/src/main.ts`, change the `terrain` source `tiles` URL to `http://localhost:8080/tiles/terrain_rgb/{z}/{x}/{y}.png`

**Why not auto-done:** The mbtiles file would be ~2–8 GB for zoom levels 5–15 and requires running GDAL + rio-rgbify, which cannot be executed from within the TypeScript/HTML code.

---

### 2. Vector Tiles (.pbf) for Streaming
**Requested:** _"Streaming GeoJSON / Vector Tiles (.pbf) for building footprints, roads, and parcel boundaries"_

**Status:** ⚠️ Architecture is ready; tile generation not set up.

The app loads GeoJSON directly (same as viewer.html). To switch to .pbf vector tiles:

1. Install tippecanoe: `brew install tippecanoe` (Mac) or build from source on Windows
2. Generate tiles:
   ```bash
   tippecanoe -o layers/buildings.mbtiles -z15 -Z10 layers/buildings.geojson
   tippecanoe -o layers/parcels.mbtiles -z14 -Z8 layers/all_parcels.geojson
   tippecanoe -o layers/roads.mbtiles -z15 -Z10 layers/roads.geojson
   ```
3. Serve with tileserver-gl or Martin tile server
4. In `main.ts`, change source types from `geojson` to `vector` and add `source-layer` to each layer spec

**Why not auto-done:** tippecanoe is a native binary not available in the Python/Node environment, and a tile server must be running continuously.

---

### 3. KGS Karst/Sinkhole Official Layer
**Requested:** _"Integration of Kentucky Geological Survey (KGS) Karst/Sinkhole layers"_

**Status:** ⚠️ Using derived sinkholes; KGS official data requires download.

The current sinkhole layer (`layers/sinkholes.geojson`) is derived from the LiDAR DEM by `flood_analysis.py`. The KGS publishes the official Kentucky Sinkhole database:

- **KGS Sinkhole Dataset:** https://kgs.uky.edu/kgsmap/kgsgeoserver/viewer.asp
- **WFS endpoint:** `https://kgs.uky.edu/kgsweb/...` (contact KGS for WFS/WMS access)
- **Download:** The KGS geologic GIS data viewer allows shapefile download by county

To integrate: download the Warren County sinkhole shapefile from KGS, convert to GeoJSON with `ogr2ogr`, and add as an additional layer in `main.ts`.

---

### 4. TWI Raster as Live Basemap Option
**Requested:** _"Multi-directional hillshade and Topographic Wetness Index (TWI) basemap options"_

**Status:** ⚠️ TWI raster generation scripted; not yet served as tiles.

The sidebar shows a hillshade toggle with direction control (fully working, uses MapLibre's built-in hillshade layer type). TWI requires a pre-processed raster converted to tiles:

1. Run `scripts/dem_preprocess.py` — outputs `layers/twi.tif`
2. Convert to tiles: `gdal2tiles.py --zoom=8-15 layers/twi.tif layers/twi_tiles/`
3. Serve tiles (e.g., with `python -m http.server 8081` in the project root)
4. Add to `main.ts`:
   ```typescript
   map.addSource('twi-src', {
     type: 'raster',
     tiles: ['http://localhost:8081/layers/twi_tiles/{z}/{x}/{y}.png'],
     tileSize: 256
   })
   map.addLayer({ id: 'twi', type: 'raster', source: 'twi-src',
     paint: { 'raster-colorize-mix': [...] } // pseudo-color TWI
   })
   ```

---

## Local Run Instructions

### Quick Start (no build — uses existing viewer.html)
```
cd C:\Users\Minec\Documents\BG_3D_Map
python -m http.server 8000
# Open http://localhost:8000/viewer.html
```

### New TypeScript App (Vite dev server)

Node.js is required. Install it from https://nodejs.org (LTS version).

```
cd C:\Users\Minec\Documents\BG_3D_Map\app
npm install
npm run dev
# Opens http://localhost:3000 automatically
```

### Production Build
```
cd C:\Users\Minec\Documents\BG_3D_Map\app
npm run build
# Output lands in BG_3D_Map/dist/
# Serve with: python -m http.server 8000  (from BG_3D_Map/dist/)
```

### DEM Preprocessing
```
cd C:\Users\Minec\Documents\BG_3D_Map
conda activate sharppy-env
pip install richdem pysheds rio-rgbify  # one-time
python scripts\dem_preprocess.py
```

---

## Memory Optimization for Large Geospatial Datasets

| Dataset | Current Size | Recommended Approach |
|---|---|---|
| `all_parcels.geojson` (26 MB) | Loads into browser memory | Use vector tiles (.pbf) at zoom 8–14 |
| `buildings.geojson` (12 MB) | Loads into browser memory | Use vector tiles at zoom 12–16 |
| `flood_damage.geojson` (50 MB) | **Do not load in browser** | Query from backend API; join to parcels on demand |
| `min_rain.tif` (56 MB) | Pre-processed on server | Never load client-side; serve as raster tiles |

**For the viewer:** The current GeoJSON approach works fine at this scale (total ~65 MB for active layers). Switching to .pbf vector tiles would reduce initial load time by ~70% and enable zoom-level-dependent detail (fewer features at low zoom).

**For production:** Consider running a PostGIS/pg_tileserv stack:
```
parcels → pg_tileserv → /tiles/parcels/{z}/{x}/{y}.pbf
buildings → pg_tileserv → /tiles/buildings/{z}/{x}/{y}.pbf
```
This enables server-side filtering (e.g., return only flood_at_rain_in ≤ 1.0 buildings) and supports datasets of any size.
