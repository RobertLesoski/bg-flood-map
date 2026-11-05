# Viewer accuracy audit (branch viewer-accuracy-2, base desktop-integration ca72db2)

Independent refs: rasterio direct reads (COG), rasterio.warp aeqd/laea projections (pyproj not installed in anaconda env).
Script: scratchpad a1.py / a3.py (not committed).

| # | Check | N | max | mean | p99 | Status |
|---|-------|---|-----|------|-----|--------|
| 1 | Probe depth (TileDepthSource.depth_many, nearest, hour=round(s/3600)) vs COG cell, county 2yr h1/h6, 100yr h1/h12, ev4 h1/h6, ev6 h1/h3, ev1 h1/h6 | 300 per storm-hour (150 wet + 150 random) | 8.6e-8 m | <3e-9 m | 4.8e-8 m | PASS (float32 rounding only); 2yr/100yr/ev1 hour 1 has no wet cells |
| 3a | Path length vs aeqd | 50 shapes | 1.5e-7 rel | 4.5e-8 | 1.4e-7 | PASS |
| 3b | Polygon area vs laea (edges densified) | 50 | 1.1e-3 rel before fix; 1.1e-6 after | 1.2e-4 / 1.2e-7 | 7.3e-4 / 7.5e-7 | BUG FIXED |
| 3c | Unit conversions (m/ft/km/mi/m2/acre/mi2) | 7 values | exact to display precision | | | PASS |

Bugs fixed
- geodesy.polygon_area_m2: one chord per edge in authalic sin(lat) space -> up to 0.11 % area error on ~20 km polygons. Now densifies edges (k=32). Test: desktop/tests/test_audit_geodesy.py (fails on old code).

Resume checklist (not yet run)
- [ ] 1 peak probe (no max_depth.tif reader is used by the probe; check series peak path) + city flow grids
- [ ] 2 building card all_touched vs rasterio
- [ ] 4 profile ground/water vs DEM
- [ ] 5 point/building series + peaks
- [ ] 6 timeline counts vs whole-array
- [ ] 7 compare sign/legend
- [ ] 8 address/facilities/roads/report
- [ ] 9 storm descriptions vs storm_info.json / STORM_DATA_SOURCES.md / VALIDATION_EV4.md
