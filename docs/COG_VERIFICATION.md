# COG verification (county real-depth tiles)

Date of the checks: 2026-11-05. Every number below was produced by a script run on Vista (`scripts/verify_cog_accuracy.py`, read-only windowed point reads on a login node, no Slurm job) or locally (`scripts/crosscheck_cogs.py`, `cog/test/real-check.html`). Machine-readable copy of everything: `layers/cog/verification.json`.

## 1. Verdict

* All 12 storms x (25 hourly + max_depth) COGs are structurally correct: EPSG:3857, uint16 centimetres, nodata 65535, 512 px tiles, overviews 2/4/8/16/32, GDAL COG layout, identical size and bounds in every file of a storm.
* **The tiles in `layers/cog/county` are the set `county_cog_tol` (Slurm job 1123323, GDAL warp tolerance 0.01 px), not `county_cog` (job 1122676, default tolerance 0.125 px).** The local copy is byte-identical (sha256, 324 of 324 files) to `county_cog_tol` on Vista and to none of `county_cog`.
* Against the solver rasters the local tiles are a faithful bilinear resample: error to the exactly expected bilinear value is at most 1 cm and the 99th percentile 1 cm over 22421 random points; `county_cog` (default tolerance) reaches 15 cm (99th percentile up to 6.0 cm; a first independent sample, picked from county_cog itself, found up to 21 cm). The earlier set is therefore the less accurate one, but both agree with the solver rasters to within about 6 cm in 99 % of the points.
* The honest limitation is not the file format but the **re-gridding**: the solver rasters are 5.0 m in UTM 16N, the COGs 4.99 m in Web Mercator. Same resolution, rotated grid (convergence about 0.35 deg), so every COG pixel is a bilinear mix of up to four solver cells. Compared with the single solver cell under a point the COG differs on average by 7 to 8 cm in the deep stratum, 6 cm in the mid stratum, 4 cm in the low stratum and 1 cm in the trace stratum (99th percentile up to 94 cm in the deep stratum, largest single difference 132 cm), in both directions equally (signed mean about 0). See section 5.
* The viewer reads the same numbers as rasterio: 516 of 516 points through the reader harness and 480 of 480 through the viewer's own `depthAt` path, all equal. Overlay, click query, risk card and the phone layout were exercised on the real tiles.

## 2. What was checked and how

* Sets: `county_cog` (job 1122676, 17:29-17:36 on 2026-11-04, make_cogs.py before the tolerance fix) and `county_cog_tol` (job 1123323, 22:39-22:43, `WarpedVRT(..., tolerance=0.01)`). No new job was submitted for this audit. `county_cog_tol_test` holds one file (100yr max_depth.tif) and was not used.
* Points (per storm): 560 to 660 on max_depth.tif and 980 to 1 350 on the 25 hourly files (about 60 per file). They are picked on a factor-8 overview of the COG with these target shares: deep 15 %, mid 15 %, low 14 %, trace 10 %, dry 8 %, near the domain edge 12 %, just outside the domain 4 %, tile boundary 22 %; the stratum is then taken from the full-resolution COG value, so counts per stratum differ slightly (tables below).
* At each point the COG pixel centre (EPSG:3857) is transformed with PROJ (pyproj) to UTM 16N and compared with the SOURCE raster (`runs/county_v2/<storm>/depth_*.tif`, `max_depth.tif`, float32 metres x 100 -> cm): (a) **bilinear** = bilinear interpolation of the four surrounding source cell centres (exactly what the converter asked GDAL for; only points with four valid neighbours), (b) **nearest** = the single source cell that contains the position (what a person reading the solver file would call 'the value'), (c) **hull** = the COG value lies between the minimum and maximum of those four cells (plus or minus 1 cm), a property every bilinear resample must have.
* Statistics are absolute differences in cm: max, mean and 99th percentile.

## 3. Structure (all 12 storms)

Per storm: 25 hourly files (`depth_<seconds>s.tif`, hours 0 to 24; names are not exact multiples of 3600 s) + `max_depth.tif` + `cog_meta.json` = 27 files, 26 rasters. Every raster: EPSG 3857, uint16, nodata 65535, tiled 512x512, DEFLATE with predictor 2, overviews [2, 4, 8, 16, 32], 10475 x 12156 px of 6.25 m, bounds (-9653845.26, 4390038.47, -9588376.51, 4466013.47), `LAYOUT=COG`. Both sets pass for all 12 storms (`structure` column below).

## 4. Accuracy per storm

### 4.1 Local tiles = `county_cog_tol` (what the viewer reads)

| storm | files | structure | points (max / hourly) | bilinear err max / p99 (cm) | bilinear mean (cm) | vs nearest cell: mean / p99 / max (cm) | hull violations |
|---|---|---|---|---|---|---|---|
| 2yr | 26 | OK | 659 / 1189 | 1 / 1.0 | 0.017 | 2.16 / 51 / 106 | 0 of 1536 |
| 5yr | 26 | OK | 659 / 1213 | 1 / 1.0 | 0.015 | 1.95 / 33 / 86 | 0 of 1567 |
| 10yr | 26 | OK | 659 / 1230 | 1 / 1.0 | 0.013 | 1.92 / 36 / 132 | 0 of 1593 |
| 25yr | 26 | OK | 659 / 1264 | 1 / 1.0 | 0.013 | 1.75 / 34 / 85 | 0 of 1619 |
| 50yr | 26 | OK | 659 / 1271 | 1 / 1.0 | 0.017 | 1.97 / 41 / 60 | 0 of 1631 |
| 100yr | 26 | OK | 659 / 1280 | 1 / 1.0 | 0.023 | 2.00 / 31 / 87 | 0 of 1626 |
| ev1 | 26 | OK | 560 / 977 | 1 / 0.0 | 0.004 | 1.03 / 37 / 47 | 0 of 1236 |
| ev2 | 26 | OK | 659 / 1099 | 1 / 1.0 | 0.014 | 1.71 / 34 / 93 | 0 of 1460 |
| ev3 | 26 | OK | 659 / 1102 | 1 / 1.0 | 0.014 | 1.75 / 35 / 74 | 0 of 1474 |
| ev4 | 26 | OK | 659 / 1306 | 1 / 1.0 | 0.018 | 2.02 / 29 / 80 | 0 of 1663 |
| ev5 | 26 | OK | 659 / 1350 | 1 / 1.0 | 0.021 | 2.12 / 32 / 107 | 0 of 1718 |
| ev6 | 26 | OK | 659 / 1331 | 1 / 1.0 | 0.014 | 1.93 / 30 / 86 | 0 of 1683 |

### 4.2 Earlier set `county_cog` (default warp tolerance, job 1122676)

| storm | files | structure | points (max / hourly) | bilinear err max / p99 (cm) | bilinear mean (cm) | vs nearest cell: mean / p99 / max (cm) | hull violations |
|---|---|---|---|---|---|---|---|
| 2yr | 26 | OK | 659 / 1189 | 12 / 5.0 | 0.276 | 2.20 / 53 / 110 | 0 of 1536 |
| 5yr | 26 | OK | 659 / 1213 | 11 / 4.4 | 0.254 | 1.95 / 36 / 94 | 0 of 1567 |
| 10yr | 26 | OK | 659 / 1230 | 14 / 6.0 | 0.249 | 1.94 / 39 / 141 | 1 of 1593 |
| 25yr | 26 | OK | 659 / 1264 | 9 / 4.0 | 0.224 | 1.75 / 35 / 88 | 0 of 1619 |
| 50yr | 26 | OK | 659 / 1271 | 15 / 4.0 | 0.253 | 1.98 / 42 / 63 | 0 of 1631 |
| 100yr | 26 | OK | 659 / 1280 | 15 / 5.0 | 0.285 | 2.03 / 32 / 88 | 1 of 1626 |
| ev1 | 26 | OK | 560 / 977 | 4 / 3.0 | 0.073 | 1.01 / 36 / 49 | 0 of 1236 |
| ev2 | 26 | OK | 659 / 1099 | 7 / 4.0 | 0.182 | 1.72 / 33 / 91 | 0 of 1460 |
| ev3 | 26 | OK | 659 / 1102 | 13 / 3.4 | 0.222 | 1.79 / 37 / 84 | 0 of 1474 |
| ev4 | 26 | OK | 659 / 1306 | 15 / 4.0 | 0.267 | 1.99 / 30 / 66 | 0 of 1663 |
| ev5 | 26 | OK | 659 / 1350 | 14 / 4.4 | 0.255 | 2.12 / 32 / 105 | 0 of 1718 |
| ev6 | 26 | OK | 659 / 1331 | 12 / 3.8 | 0.238 | 1.93 / 30 / 75 | 0 of 1683 |

The difference between the sets is the GDAL warp approximation: by default GDAL replaces the exact UTM to Mercator transform by a linear approximation that may be off by 0.125 source pixels; on noisy real depths that is up to 15 cm. With `tolerance=0.01` it is at most 1 cm, i.e. the rounding of the stored centimetre plus GDAL's fixed-point weights.

## 5. Why the tolerance against the nearest solver cell is larger, by depth stratum

The COG is bilinear, so its error against the bilinear expectation is about 1 cm. Against the nearest cell the error is bounded by the local depth gradient: a bilinear value lies between the four neighbouring cells, so it can differ from the cell under the point by at most the spread of those cells. The rain-on-grid solution is rough at the cell scale (micro-depressions, sinks, roads and roofs next to each other): the median spread of the four cells is 25.5 cm in the deep stratum of the 2-year max_depth sample, and individual neighbours differ by more than 1 m (example from the first run of the audit, 2yr max_depth, pixel col 2757 row 2651: COG 82 cm = the bilinear value, solver cell under it 171 cm, the four cells span 171 cm). The error is therefore concentrated where the water is deep and the surface rough, and is zero where it is dry:

Pooled over the 12 storms, hourly rasters, local tiles:

| depth stratum (COG value) | points | vs bilinear: max / mean abs / p99 range (cm) | vs nearest source cell: max / mean abs / p99 range (cm) | nearest: signed mean (cm) | share COG lower / higher by > 1 cm |
|---|---|---|---|---|---|
| deep(>=1.2m) | 913 | 1 / 0.077 / 0.0-1.0 | 78 / 7.30 / 6.0-57.2 | -0.23 | 35 % / 31 % |
| mid(0.3-1.19m) | 1330 | 1 / 0.053 / 0.8-1.0 | 107 / 6.61 / 26.4-67.2 | -0.78 | 33 % / 28 % |
| low(0.1-0.29m) | 735 | 1 / 0.031 / 0.0-1.0 | 35 / 4.38 / 17.9-29.9 | +0.51 | 26 % / 30 % |
| trace(0.01-0.09m) | 1070 | 1 / 0.015 / 0.0-1.0 | 14 / 1.52 / 6.0-9.9 | +0.65 | 9 % / 23 % |
| dry(0) | 8096 | 1 / 0.000 / 0.0-0.0 | 1 / 0.00 / 0.0-0.0 | -0.00 | 0 % / 0 % |
| within 3 px of the domain edge | 240 | 0 / 0.000 / 0.0-0.0 | 5 / 0.03 / 0.0-4.2 | -0.01 | 0 % / 0 % |
| first/last pixel row or column of a 512 px tile | 3839 | 1 / 0.002 / 0.0-0.0 | 86 / 0.24 / 0.0-13.8 | +0.01 | 2 % / 2 % |

Pooled, max_depth.tif, local tiles:

| depth stratum (COG value) | points | vs bilinear: max / mean abs / p99 range (cm) | vs nearest source cell: max / mean abs / p99 range (cm) | nearest: signed mean (cm) | share COG lower / higher by > 1 cm |
|---|---|---|---|---|---|
| deep(>=1.2m) | 773 | 1 / 0.054 / 0.4-1.0 | 132 / 8.41 / 30.3-93.5 | -1.44 | 34 % / 27 % |
| mid(0.3-1.19m) | 994 | 1 / 0.046 / 0.1-1.0 | 74 / 5.86 / 21.7-64.7 | -0.45 | 30 % / 30 % |
| low(0.1-0.29m) | 446 | 1 / 0.034 / 0.0-1.0 | 36 / 3.73 / 8.0-29.6 | +0.61 | 27 % / 28 % |
| trace(0.01-0.09m) | 793 | 1 / 0.006 / 0.0-0.4 | 12 / 1.19 / 6.0-10.6 | +0.43 | 8 % / 16 % |
| dry(0) | 3656 | 1 / 0.001 / 0.0-0.0 | 1 / 0.01 / 0.0-1.0 | -0.01 | 0 % / 0 % |
| within 3 px of the domain edge | 118 | 0 / 0.000 / 0.0-0.0 | 3 / 0.09 / 0.0-2.8 | +0.07 | 0 % / 2 % |
| first/last pixel row or column of a 512 px tile | 1721 | 1 / 0.002 / 0.0-0.0 | 52 / 0.46 / 0.0-16.8 | -0.08 | 3 % / 3 % |

Reading the table: the signed mean of the nearest-cell difference is about zero and the shares 'COG lower' and 'COG higher' are about equal (25 to 35 % each in the deep, mid and low strata). The re-gridding scatters values; it does not damp peaks or inflate them on average. Still, a single pixel can read up to about 1 m away from the solver cell that sits under it in deep rough spots, so a pixel value is 'the model depth here, to within the local roughness', not an exact reproduction of one solver cell. Edge and tile-boundary pixels behave like all others (no seam errors: bilinear differences of at most 1 cm there; COG nodata coincides with the solver nodata at every one of the 22421 tested points).

Hull check: 0 of 18806 tested values lie outside the min/max of their four source cells (local tiles); 2 of 18806 for `county_cog`.

### max_depth.tif against the hourly files

In `flood_solver_v2.py` max_depth.tif is the maximum over the saved snapshots (hourly files and the final state), not over every solver time step. This was confirmed on the data: at 300 random points inside the model per storm the source max_depth equals the maximum of the 25 source hourly rasters to the cm (0 points differ by more than 1 cm in any storm), and the COG max_depth equals the maximum of the 25 COG hourly rasters (differences 0 or 1 cm, none below -1 cm). A peak between two saves (the real solver maximum) is not in either file.

## 6. Download check

sha256 of every file in `layers/cog/county` (324 files, 5.48 GB) against `sha256sum` on Vista: **mismatches against `county_cog_tol`: 0; against `county_cog`: 324 of 324 files.** The two Vista sets share no identical file.

## 7. Cross-check against the other depth products (same solver rasters)

`scripts/crosscheck_cogs.py`, windowed reads, local tiles. 300 random buildings per storm (150 chosen among those the address index lists as wet, 150 among the others, so the agreement shares below are for this half-wet sample, not county-wide rates), all 98 facilities (97 inside the model).

**Viewer flow grids (25 m cells = 4x4 block mean of 5 m, 8 bit sqrt coded) at hours 12 and 24.** About 295 random cells per storm and hour. COG mean over the cell minus grid depth: signed mean between -0.4 and +0.7 cm (no bias), mean absolute 0.7 to 3.5 cm, 99th percentile 4.9 to 53 cm; share within 5 cm 79 to 99 %. COG maximum over the cell minus grid: +5 to +16 cm on average and COG >= grid in 91 to 99 % of cells, because the grid is a mean and the COG keeps the peak cells.

| storm | hour 12: mean abs / p99 (cm) | hour 24: mean abs / p99 (cm) | COG max - grid, hour 24 (mean, cm) |
|---|---|---|---|
| 2yr | 1.08 / 7.7 | 2.63 / 23.0 | +11.1 |
| 5yr | 1.34 / 9.5 | 3.04 / 38.2 | +12.9 |
| 10yr | 1.60 / 16.2 | 2.95 / 31.1 | +13.7 |
| 25yr | 2.25 / 19.7 | 2.93 / 40.4 | +14.0 |
| 50yr | 2.50 / 30.6 | 2.71 / 20.3 | +10.9 |
| 100yr | 2.57 / 27.6 | 3.48 / 30.9 | +15.6 |
| ev1 | 0.66 / 4.9 | 1.01 / 6.4 | +7.6 |
| ev2 | 1.00 / 7.3 | 1.27 / 11.7 | +7.2 |
| ev3 | 0.88 / 9.7 | 2.35 / 22.1 | +10.5 |
| ev4 | 1.91 / 17.7 | 3.34 / 29.8 | +13.7 |
| ev5 | 2.55 / 24.6 | 2.92 / 20.2 | +15.0 |
| ev6 | 1.83 / 20.5 | 3.40 / 53.0 | +13.3 |

**Address index (building depth = max over the footprint of the flow grid over hours 4 to 24) against the COG footprint maximum (all-touched pixels of max_depth.tif).**

| storm | buildings | COG - index: mean (cm) / median abs / p99 abs | both flooded >= 10 cm: median COG/index ratio | index >= 10 cm and COG < 10 cm | index < 10 cm and COG >= 10 cm | agreement of the 10 cm flag |
|---|---|---|---|---|---|---|
| 2yr | 300 | +1.5 / 3 / 39 | 1.171 | 26 | 17 | 86 % |
| 5yr | 300 | -1.5 / 3 / 84 | 0.885 | 32 | 11 | 86 % |
| 10yr | 300 | +0.1 / 3 / 47 | 1.127 | 35 | 12 | 84 % |
| 25yr | 300 | -1.6 / 4 / 54 | 1.008 | 38 | 15 | 82 % |
| 50yr | 300 | -1.3 / 3 / 58 | 1.111 | 35 | 9 | 85 % |
| 100yr | 300 | -2.7 / 3 / 65 | 1.013 | 35 | 9 | 85 % |
| ev1 | 300 | +4.2 / 3 / 32 | 2.261 | 4 | 41 | 85 % |
| ev2 | 300 | +2.0 / 3 / 37 | 1.643 | 21 | 23 | 85 % |
| ev3 | 300 | -2.2 / 1 / 70 | 1.216 | 34 | 13 | 84 % |
| ev4 | 300 | +0.5 / 3 / 48 | 0.989 | 27 | 15 | 86 % |
| ev5 | 300 | -0.8 / 3 / 47 | 1.015 | 25 | 12 | 88 % |
| ev6 | 300 | -1.9 / 3 / 66 | 0.892 | 30 | 10 | 87 % |

The median difference is 1 to 4 cm, but in 12 to 18 % of the sampled buildings the two products disagree on the 10 cm flag, in BOTH directions: the grid (20 m cells, 8 bit) smears a deep pit or ditch next to a house into the house's cells and misses 5 m features; the COG sees the 5 m footprint pixels only. The index is the maximum over hourly snapshots of the block-averaged grid; the COG is the maximum over the saved snapshots at 5 m, so neither is 'right': they are different resolutions of the same run. For rare events (ev1) the COG flags more buildings (41 of 300 are >= 10 cm in the COG but not in the index) because 5 m cells keep thin ponding that the 20 m mean dilutes below 10 cm.

**Infrastructure (98 facilities, `d_<storm>` = max over the facility's outline or point on the grid) against the COG maximum over the same outline/point:**

| storm | COG - index: mean (cm) / median abs / max | facilities >= 10 cm: index / COG | agreement of the 10 cm flag |
|---|---|---|---|
| 2yr | +18.1 / 1 / 210 | 26 / 33 | 93 % |
| 5yr | +18.3 / 1 / 219 | 27 / 33 | 92 % |
| 10yr | +17.8 / 1 / 222 | 27 / 33 | 92 % |
| 25yr | +17.4 / 1 / 223 | 28 / 34 | 94 % |
| 50yr | +17.1 / 1 / 224 | 28 / 33 | 93 % |
| 100yr | +17.1 / 1 / 223 | 29 / 34 | 93 % |
| ev1 | +5.6 / 0 / 47 | 6 / 22 | 84 % |
| ev2 | +14.8 / 1 / 184 | 24 / 31 | 93 % |
| ev3 | +17.6 / 1 / 207 | 28 / 31 | 95 % |
| ev4 | +18.1 / 1 / 223 | 30 / 35 | 93 % |
| ev5 | +17.3 / 1 / 223 | 30 / 34 | 94 % |
| ev6 | +17.2 / 1 / 223 | 31 / 36 | 93 % |

Systematic difference: the median is 0 to 1 cm but the mean is +15 to +18 cm (COG deeper; +5.6 cm for ev1) because a handful of campus outlines (schools, large sites) contain a pond or ditch whose 5 m cells reach 2 m while the 20 m grid averages it down (largest differences 1.8 to 2.2 m). The COG finds 33 to 34 facilities at >= 10 cm in the design storms where the index finds 26 to 29. The facility file was built from the grid; it has not been rebuilt from the COG.

Facility depths here use the footprints rebuilt from the cached OSM extract exactly as `build_infrastructure.py` does (outlines of merged parts, points for nodes).

## 8. Viewer on the real tiles

* reader_vs_rasterio (cog/test/real-check.html, cog-depth.js in the browser vs rasterio on the same files): 516 of 516 points identical (2yr, 100yr, ev4 x hour 12 and max_depth x 86 points each, including nodata/outside and points inside a pixel); footprint max >= centre pixel 10 of 10 per file
* viewer depthAt path (CogStore.depthAt as used by the click query): 480 of 480 points identical to rasterio (2yr, 100yr, ev4; max_depth and hour 12; 80 points per file)
* overlay: drawn and looked at: 100yr desktop at zoom 12.5 and 14 (peak mode), ev4 at zoom 13 (peak) and 16.2 (hour 22 mode), 2yr at zoom 12, 100yr on a 375 px phone viewport at zoom 14.5; thin shallow water fades when zoomed out (averaged overviews) as the note says; at zoom 16 single 6.25 m pixels are visible (nearest resampling)
* click query: 100yr desktop pond: grid depth 0.91 m, solver peak 156 cm, solver hour 12 93 cm; phone: grid 0.13 m, solver peak 244 cm, hour 12 22 cm (the 20 m grid mean and the 5 m solver cell differ as section 7 explains)
* risk card: ?addr=15000 (618 Magnolia Street, county): rows labelled solver show 2 cm in every storm; rasterio footprint maximum (all-touched) of that building from layers/cog/county/<storm>/max_depth.tif = 2 cm for 2yr, 5yr, 100yr, ev4: identical
* phone layout 375 x 812: no horizontal overflow (document scrollWidth 375 = innerWidth); bottom sheet holds the real-depth switch and the About panel; click query card readable
* About this storm panel: renders for all 12 storms in the county domain (13 sections each, no undefined/NaN, COG and Slurm lines present), no overflow on the phone viewport
* defect found and fixed: cog/test/real-check.html pointed at ../layers/cog (404); now ../../layers/cog. No defect found in viewer.html or cog/cog-depth.js
* pytest: scripts/tests: 257 passed + 1 timing test (test_flowgrid TestPerformance::test_4000x4000) failed once while the machine was busy and passed alone; test_storm_info.py (9 tests) added; see the final run in WEB_COG_PROGRESS.md

## 9. Native resolution of the solver rasters and what the COG resampling costs

Read with rasterio on Vista from `runs/county_v2/<storm>/max_depth.tif` and the hourly files (same for all storms): **EPSG:32616 (UTM 16N), 5.0 m x 5.0 m cells, 10404 x 12043, float32 metres, nodata -9999, LZW tiled 256.** The COG is EPSG:3857 at 6.25 m, which on the ground at 37 N is 6.25 x cos(37 deg) = 4.99 m. The two grids therefore have the same resolution (ratio 0.998) but are not aligned: Mercator north is rotated against UTM north by the grid convergence (about 0.35 deg at the county), so the converter must interpolate. Consequences:

* No information is lost to coarsening (4.99 m vs 5.0 m) and no information is invented (no finer grid exists). The result is a smoothing of about one cell, equal in both directions on average (signed mean about 0).
* The fraction of a solver cell's depth that survives at a pixel is a weighted mean of up to four cells, so isolated one-cell features are blurred (example above: 171 cm cell read as 82 cm) and thin lines (ditches, curbs) may fade; section 5 gives the size of this by depth stratum.
* Overview levels (6.25 x 2^n m) are averages, as the viewer's note says; queries use full resolution.
* Option if exact cell values matter: a native 5 m UTM COG (no warp, nearest or identity) would reproduce every solver cell with zero resampling error; the browser reader would then need a UTM 16N inverse projection (a few dozen lines, testable against PROJ as the Mercator one is). Not built: no new jobs were allowed for this audit.

Terrain (the real resolution behind the 5 m model outside the city): see `STORM_DATA_SOURCES.md`; measured on Vista: `dem_warren_5m.tif` equals the 3DEP 1/3 arc-second product (about 10.3 x 8.2 m) resampled to 5 m outside the 20 x 20 km city DEM (mean absolute difference to 3DEP 0.21 m at 368 points), and equals the 2 m city DEM inside (0.06 m at 32 points). The depths outside the city therefore carry about 10 m of real terrain information on a 5 m grid.

## 10. Not verified

* Flood depths against observations (no gauge data, no calibration).
* `max_depth.tif` against every solver time step (only snapshots exist).
* The viewer's overlay colours against a second renderer (looked at in the Browser pane only, screenshots in the session).
* Points were sampled randomly, not exhaustively (22421 points per set); an area of larger error with fewer than about 1 in 1 000 pixels could be missed, but the structure of the error (bounded by the local spread) was confirmed by the hull check at every point.
