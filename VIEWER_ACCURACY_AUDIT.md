# Viewer accuracy audit (branch viewer-accuracy-2, base desktop-integration ca72db2)

Independent refs: rasterio direct reads (COG), rasterio.warp aeqd/laea projections (pyproj not installed in anaconda env).
Script: scratchpad a1.py / a3.py (not committed).

| # | Check | N | max | mean | p99 | Status |
|---|-------|---|-----|------|-----|--------|
| 1 | Probe depth (TileDepthSource.depth_many, nearest, hour=round(s/3600)) vs COG cell, county 2yr h1/h6, 100yr h1/h12, ev4 h1/h6, ev6 h1/h3, ev1 h1/h6 | 300 per storm-hour (150 wet + 150 random) | 8.6e-8 m | <3e-9 m | 4.8e-8 m | PASS (float32 rounding only); 2yr/100yr/ev1 hour 1 has no wet cells |
| 3a | Path length vs aeqd | 50 shapes | 1.5e-7 rel | 4.5e-8 | 1.4e-7 | PASS |
| 3b | Polygon area vs laea (edges densified) | 50 | 1.1e-3 rel before fix; 1.1e-6 after | 1.2e-4 / 1.2e-7 | 7.3e-4 / 7.5e-7 | BUG FIXED |
| 3c | Unit conversions (m/ft/km/mi/m2/acre/mi2) | 7 values | exact to display precision | | | PASS |
| 1b | Probe card (TileService.depth_lnglat) peak (max_depth.tif) + nearest hour, random point inside random cell, same storms incl. ev6 h1 | 300 per case (11 cases) | 4.4e-16 m | ~0 | 1.1e-16 | PASS, status (no data vs depth) mismatches 0 |
| 1c | Default bilinear path (TileDepthSource.depth_many) vs independent centre bilinear w/ nodata renormalised | 300 per case | 4.3e-8 m | <2e-9 | 2.3e-8 | PASS |
| 1d | City flow grids 5yr/100yr h3/h6/h12 (FlowGridDepthSource) vs own gunzip+decode (t^2*8 m) at cell centres | 300 per case | 8.5e-10 m | <6e-11 | 4.3e-10 | PASS |
| 2 | Building card (building_depth) vs rasterio all_touched max: ev6 h1, ev6 peak, 100yr peak, ev4 h6 | 300 per case (696 wet) | 0 cm | 0 | 0 | PASS; label (depth/dry/no data) mismatches 0 |
| 4 | Profile (build_profile, 200 stations, 10 lines, ev6 h1 + 100yr h12): ground vs DEM bilinear / depth / surface / distance vs aeqd | 2000 stations | ground 1.3e-10 m, depth 5.3e-8 m, surface 5.3e-8 m, dist 5.0e-6 m | | | PASS |
| 5a | Point series (compute_row) 25 h vs independent bilinear, ev6 + 100yr | 100 points x 25 h | 1.1e-7 m | 3.5e-9 | 4.3e-8 | PASS; peak err 5.5e-8 m; peak-hour mismatch 0/100 |
| 5b | Building series: old corner+centroid samples vs all-touched (100yr peak) | 374 wet bldgs | 47 cm under-read | 1.5 cm | 20.8 cm | BUG FIXED (153/374 under-read) |
| 6 | Timeline vs whole-array (definitions: wet area = sum of cells >= 0.02 m x (res*cos lat)^2; road km = road length (1 m steps, nearest cell) with depth >= 0.10 m; flooded = buildings whose all-touched footprint max >= 0.10 m) | ev6 h3 / 100yr h12 / ev4 h6 | area 0.0017 km2 of 375 (4.5e-6 rel); road km 0.06-0.08 km (3.6 % on ev6 h3, 0.04 % on 100yr: sampling spacing); flooded OLD 83/1432/377 vs 164/1698/550 | | | area PASS, roads OK, flooded BUG FIXED |
| 7 | Compare B-A (ev4 h6 vs 100yr h12): sign vs independent tile block means; pack vs grids; legend | 200 wet cells | sign 82/82 agree where abs>0.1 m; value 0.43 m max / 0.05 mean / 0.34 p99 (bilinear vs block-mean resampling, by design); pack float16 3.5e-4 m | | | PASS (warm = B deeper, blue = shallower, grey < 2 cm) |
| 8 | Address search: "num street" query returns that record | 200 addresses | 0 misses | | | PASS. Depth vs grid NOT verified: index bid is an OSM id (max 1.5e9), mapping to scene buildings not audited |
| 9 | Storm blurbs vs storm_info.json (2yr,100yr,ev1,ev4,ev6): all numbers present or correct in-to-mm conversions; validation line vs VALIDATION_EV4.md (not validated, 21 roads, AUC 0.84, "weak sign of skill") | 5 storms | | | | PASS |

Bugs fixed
- geodesy.polygon_area_m2: chord per edge in authalic space, up to 0.11 % area error. Densify edges. test_audit_geodesy.py.
- series.building_target (tiles): corner+centroid samples under-read the building card by up to 47 cm. Now samples the card's all-touched tile cells when the footprint index is ready. test_audit_series_building.py.
- timeline flooded-building count: same sampling, under-counted 15-50 % vs the card statistic. prepare_points uses the all-touched index; TimelineWorker recomputes when the index arrives (separate disk cache key). test_audit_timeline_counts.py.

Resume checklist
- [x] 8 facilities: 98 facilities x 5 storms, tile point depth vs rasterio sample: 0 cm error, flooded counts equal (3-4). File d_ values (index source) disagree with tiles on 7-28 of 98 at the 10 cm line: different source, viewer prefers tiles.
- [ ] 8: address depth vs grid (bid -> scene building mapping), roads-at-risk list, exported report
- [ ] compare_grids uses one cell_ground_m^2 for the whole county (cos-lat variation ~0.3 %), not changed
- [ ] tile_sample shows 1 cm as depth while probes zero < 2 cm (labelling choice, not changed)
