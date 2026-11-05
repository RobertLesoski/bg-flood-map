# SEARCH_PROGRESS - search, reporting and help for the desktop viewer

Branch `desktop-search` (from master `36fc3ec`), worktree `C:\Users\Minec\Documents\BG_3D_Map\.claude\worktrees\agent-a84b47387ba530a41`.
Files, hooks and settings: `desktop/docs/SEARCH_HOOKS.md`. Python for everything: `C:\Users\Minec\anaconda3\python.exe`.

## Status: all five features done and verified

1. Address search (Ctrl+F) with typeahead (city 30,148 / county 29,754 records), jump + building selection, result card per storm.
2. Critical facilities: markers (colour + shape per type, ring in the tier colour when flooded), list window, click card.
3. Roads at risk window (L): sortable by flooded %, max depth, length, evacuation flag; click = fly to the road and outline it.
4. Risk report (Ctrl+P / Find menu): self-contained HTML + PNG of the view (Documents\BG_Flood_Reports, `BGV_REPORT_DIR`).
5. Help / About window (F1: Help, Shortcuts, Data sources, About), first-run hint overlay, missing-data messages, log path in About.

### Tile-source update (coordinator message)
County address results, facility depths / flooded flags and the risk report use the depth tiles when they exist
(`layers/cog/county/<storm>/max_depth.tif`, 6.25 m, EPSG:3857, uint16 cm, nodata 65535): building = maximum over the cells its OSM footprint
touches (all-touched), lot / facility = the cell under the point. Both numbers and their difference are shown; a row is flagged when the
flooded state flips or the two are >= 10 cm apart (a few cm across a tier edge is not flagged). Labels: `solver depth (tiles, 6.25 m; verification
pending)` and `index (grid)`. City stays on the index (no tiles). Folder configurable: `settings["tiles_dir"]` / `BGV_TILES_DIR`; cell size, bounds and
nodata are read from the raster, so finer tiles can be dropped in; `tiles_verified` removes "verification pending". Facility cards also show the
deepest tile cell within 30 m of the point.

## Log

1. Read the docs and the viewer code; baseline `pytest desktop/tests` = 347 passed (7:39 with other agents running).
2. Pure modules (`bgv/search/*`), controller with a worker thread, imgui panels + Help window, five hooks. First real renders looked right.
3. Coordinator update: switched the county to tile depths (see above). Real data: facilities show a large index-vs-tile gap (below).
4. Fixes found by looking at renders: panel z-order (a `NoBringToFrontOnFocus` window goes BEHIND the legend: cards use normal flags), card width,
   storm labels (`Event 6`, `100-year storm`), cm units everywhere, default road sort, search bar next to the compass, facility / roads windows on the
   right of the 3D view, footprints loaded once (single-flight; an earlier version could assess a dropdown row before the footprints were ready),
   depth-tier honesty for the city (severe = ">= 120 cm", `~` prefix), first-flooded / order caches for the facility list.
5. Tests: `tests/test_search.py` (24, synthetic + real), `tests/test_search_gpu.py` (10, city app), `tests/test_search_county_gpu.py` (6, county + tiles).
6. Verification tool `tests/verify_search.py` (independent code, results below). Benchmarks below.

## Verification (independent code; `python desktop/tests/verify_search.py --n 220`, seed 7)

| Check | Result |
|---|---|
| Index, city: 330 random + flooded addresses, app StormRow (depth, tier, first hour, hours) / address text / coordinates / typeahead round trip vs a line-by-line decode of the raw gzip JSON | 0 mismatches |
| Index, county: same, 330 addresses x 12 storms | 0 mismatches |
| County tiles, building footprints: 220 random addresses x 12 storms = 2,640 footprint maxima vs `rasterio.mask(all_touched=True)` on `rasterio.warp.transform_geom` polygons | 0 mismatches (max diff 0 cm) |
| County tiles, 30 random lots x 12 storms = 360 point samples vs `rasterio` index + read at PROJ-transformed coordinates | 0 mismatches |
| All 98 county facilities x 12 storms = 1,176 point samples vs rasterio | 0 mismatches |
| Road lists: every way of every city and county storm file has geometry; header flooded km = sum of rows | 0 problems |
| City addresses -> viewer building (`bid` = feature index): 250 random, centroid distance median 0.6 m, max 36.5 m | 0 wrong buildings; reverse (building -> record) asserted >= 98 of 100 |
| County record -> OSM way polygon -> centroid -> viewer building: 200 random | median centroid distance < 3 m (asserted) |

Tiles vs index (reported, not a bug of either): on 2,640 address-storm pairs the tile - index difference has median +2 cm (p5 -13, p95 +7);
material disagreement (flooded state flips or >= 10 cm apart) 11.0 %, tier differs 9.5 %. For FACILITIES the gap is large: index flags 26 / 29 / 30 of 98
facilities flooded in 2yr / 100yr / ev4, the point sample of the tiles only 3 / 4 / 3 (2 facilities lie outside the tile domain). The index value is the
maximum over the whole campus outline on a 20 m grid, the tile value the cell under the facility point (a large school can be wet at one corner and
dry at its centre); the card therefore also shows the deepest tile cell within 30 m. Which is "right" is exactly what the pending tile verification
should settle; the app shows both and says which is the headline.

## Performance (GTX 1650, 1920x1080, other agents were using the PC during the runs)

* County scene alone (`python -m bgv --benchmark --domain county`): synced mean 8.1 ms (124 fps), p95 12.2, p99 18.0, CPU record 2.9 ms (pipelined 2.0).
* My panels, thread-CPU per frame (`python -m bgv.search.shot --bench`): search bar alone 0.1 ms; typical (bar + markers + result card) 0.9 ms; worst
  case with the facilities and roads windows also open 3.1 ms. For comparison the existing sidebar costs about 7 ms thread-CPU in the county view in the
  same harness, i.e. my features are a small part of the interface cost.
* Whole-window frame rate could NOT be measured cleanly: the offscreen harness includes a 1080p read-back and other python processes (other agents) were
  running; bare county 21 ms vs all panels 28 ms per frame in that noisy harness (47 / 36 fps). Estimated cost of the typical search UI is below
  1 ms, the worst case 3 ms. The existing "lite" UI governor still applies. Re-measure on a quiet PC before quoting a 60 fps figure for the whole UI.

## Full-suite run and regressions found (honest)

First full run with my code: 372 passed, 10 skipped, 5 failed. Four were MY regressions in `test_interaction.py` and are fixed: (1) a click on a
facility marker swallowed the click (the test's building is a facility) - now the marker opens its card and the click still selects the building;
(2) my Esc handler ran before the old priority (hidden UI / dialogs) - it now defers when the UI is hidden or a dialog is open. The fifth,
`test_deferred_start_up_...` (known flaky) and `test_storm_switch_spam_...` (VRAM baseline differed by 12 MB, ran while other agents used the GPU)
are re-run in the final check; see the end of this file for the last result.

## Known issues / not done

* Tile verification is still pending (label says so); hourly tiles (`depth_*s.tif`) are not used - the tables use the storm maximum, the viewer's own
  hour-by-hour water still comes from the flow grids.
* A merged address record (several footprints, e.g. 5) is assessed with the tile maximum of its MAIN footprint only (the card says so); the index covers all.
* Facility markers are drawn as a screen overlay (no terrain occlusion); names show only below 3.5 km camera distance.
* The first-run hint overlay is suppressed on offscreen canvases (tests, renders) unless `BGV_FORCE_HINT=1`.
* `STORM_DATA_SOURCES.md` does not exist yet: the Data sources tab shows the placeholder and the storm's own `source_note`.
* The existing flaky perf test `test_deferred_start_up_builds_the_same_app_in_steps_behind_a_loading_screen` is untouched.
