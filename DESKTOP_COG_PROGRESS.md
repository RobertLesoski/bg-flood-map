# DESKTOP_COG_PROGRESS - native solver tiles + visual polish in the desktop viewer (running log, resume from here)

Scope of this tree (after the coordinator's scope change): Phase 1 (native tiles) + visual / rendering polish. Cinematic, analysis and
search/help features are built by other agents in worktrees (branches desktop-cinematic, desktop-analysis, desktop-search) on top of the
hooks in `desktop/docs/EXTENSION_API.md`. Python: `C:\Users\Minec\anaconda3\python.exe`. Tests: `python -m pytest desktop/tests -q -p no:cacheprovider`
from the repo root (baseline before this work: 347 passed).

## Facts
- Real tiles: `layers\cog\county\<storm>\` (12 storms x 25 hourly + max_depth.tif + cog_meta.json), EPSG:3857, uint16 cm, 6.25 m, 512 px tiles, deflate + predictor 2,
  overviews x2..x32, nodata 65535. City COGs do not exist: the city view reads the county run (scope 'county', labelled). Label until verified:
  "solver depth (tiles, 6.25 m, verification pending)"; becomes plain when `settings["tiles_verified"]` is true or cog_meta.json has `"verified": true`.
- Finer (native) tiles later: set `settings["cog_folder"]` (or env BGV_COG_FOLDER, folder name under layers\) or `cog_dir` / BGV_COG_DIR (absolute); no code change.
  A `res_m <= 5` in cog_meta.json makes the label say "native tiles".

## Log
1. Extension registry `bgv/ui/registry.py` + hooks in manager/menus/panels_side; `desktop/docs/EXTENSION_API.md` written (stable contract).
2. `bgv/data/cog.py` (pure python reader, shared byte-capped LRU, vectorised sampling) - matches rasterio exactly (see verification below).
3. `bgv/data/tiles.py` (TileService, TileSet, footprint index on the tile grid cached in desktop/cache, per-building footprint maximum), `bgv/data/depthapi.py`
   (app.depth_at / building_depth / depth_profile / tile_info / app.tiles).
4. UI: probe tooltip shows solver depth (hour + peak, 'no water' vs 'no data'), building card rows, sidebar section "Solver depth tiles" with the
   tile-vs-grid flooded-building comparison, View menu switch. `tests/test_tiles.py` (15 tests), `tests/tile_verify.py` (spot check vs rasterio).
5. Verification run (tile_verify.py, 250 random points per storm, 2yr / 100yr / ev4, peak + hour 12, PROJ lng/lat -> mercator): 0 mismatches in 1500 comparisons.
   Tile vs flow grid (county flow grid is 20 m, 4x block-averaged): see desktop/docs/TILES.md.

6. Visual polish done: sun-aware sky + horizon band + Mie bloom tinting the haze (COMMON_WGSL `sky_base`), calmer shallow-water specular with a highlight shoulder
   (ponds no longer clip to flat white at sun-glint angles), building window facades (fade out under ~4 px per window), colour-blind-safe tier palette (cividis,
   same as the web viewer's default; View > Colour palette; buildings + roads via uniforms, UI via theme.set_palette), dark / light interface theme
   (View > Interface theme, theme.set_mode + apply_colours), hi-DPI checked at pixel ratio 2 (crisp, layout scales). Before / after in desktop/docs/shots/polish/.
7. Tests: suite 361 passed + 1 stale assertion (roads WGSL constant name, updated) -> now + test_ext_theme.py (5) + test_tiles.py (15).

8. Experimental `tiledepth` layer (solver depth draped on the terrain, 1024-cell clipmap window, two textures, worker reads): rendered and looked right, but NOT REGISTERED:
   registering it makes test_ui_gpu / test_interaction fail with an imgui 'Missing PopID' assert in the sidebar (root cause not found; not has_ui) and the county benchmark hitches (p99 46 ms, max 92 ms). To resume: add the registry line, find the unbalanced push_id, then test.
9. Per-building footprint maximum vs rasterio on real county buildings (300 random, 100yr): peak 299 identical, hour 12 298 identical, rest 1-2 cm sliver differences.
10. Perf (desktop/docs/PERF.md): city 104 fps synced / 176 free, county 106 / 182 (1080p, noisy machine), GPU +0.1-0.2 ms from the polish.

## Resume checklist
- [x] registry + EXTENSION_API.md (sections 1-6)
- [x] reader, service, API, probe, building card, statistics comparison, tests (test_tiles 15, test_tiles_gpu 3, test_ext_theme 5), verification (tile_verify.py, tile_verify_buildings.py)
- [x] visual polish: sun-aware sky, water glint shoulder + calmer shallow water, window facades, colour-blind palette, dark / light theme, hi-DPI checked; shots in desktop/docs/shots/polish/
- [x] perf re-measure -> desktop/docs/PERF.md; README / DESIGN / TILES.md updated
UNFINISHED / NOT DONE (honest list):
- Tile verification by the tile agent is still pending: labels say "verification pending" until `settings["tiles_verified"]` is true or cog_meta.json has "verified": true.
- No city COGs exist; the city view reads the county run (labelled). Native finer tiles: set settings["cog_folder"] / BGV_COG_FOLDER, untested with real finer files (only fixture-tested).
- Statistics comparison counts buildings at the displayed hour only (nearest hour file); at-risk list and headline numbers stay flow-grid based by design. No CSV of the comparison.
- Tile depth is not interpolated between hours (nearest hour file); the 3D water still comes from the flow grid.
- Overlay layer (experimental, unregistered): imgui PopID assert when registered; hitches while the window is re-read; hour mode only draws chunks wet in the flow grid; no hour interpolation.
- Polish skipped: camera inertia (already exponential glide), soft shadows / AO (planar sun shadows + fake AO exist), per-channel absorption of the ground seen through water (needs dual-source blending), hillshade rework.
- Light theme: overlays drawn over the 3D view (compass, scale bar, legend chips) keep their dark chips by design; the timeline chart colours were only checked once visually.
- Not built here (other worktrees): cinematic, analysis, search/help features.
# Session 3
- [x] 1 keyboard gate (manager.text_focused, app._on_key, test in test_search_gpu)
- [x] 3 default ev6 hour 1
- [ ] 2 utilities layer
- [ ] docs, final tests
- [x] URGENT blank-text UI: NOT reproducible (real window, both monitors, county; offscreen 1.0/1.25, replay, domain switch, atlas growth all fine). Hardened imgui_fast: re-upload a lost atlas texture, never replay a frame with a pending upload; regression test in test_search_gpu.
- [x] 2a: build_utilities.py + tests done, layers/utilities.geojson.gz built (599 features; no hydro plants, no tanks mapped). [ ] 2b desktop layer bgv/data/utilities.py + ui/ext_utilities.py + search hits + docs + perf
- [x] 2b desktop layer done + docs. [ ] final full desktop suite (once) + scripts tests (once)
## Session 3 resume checklist (wrap-up)
DONE + verified: (1) text-input gate (manager.text_focused/_text_key, app._on_key), cheat sheet lines; (3) default ev6 + hour 1, stored storm/hour ignored unless 'restore_last' (View menu), dry-hour label; (2) scripts/build_utilities.py (+tests, 599 features built), desktop layer bgv/data/utilities.py + bgv/ui/ext_utilities.py (View menu, sidebar legend, card, search hits), docs/UTILITIES.md, PERF notes; imgui_fast atlas hardening.
Full suite run once: 484 passed, 5 failed; scripts/tests 286 passed.
FAILING / TODO:
- test_county.py::test_pick_storm_and_cli_settings and test_interaction.py::test_storm_switch_spam... : likely expect the old default storm (pick_storm now prefers ev6 / ignores stored storm): update expectations.
- test_search_gpu.py new tests (focus gate, utilities search) pass alone but fail in module order (shared env state: text_focused stuck / cur_hits): isolate state (clear active id, c.focus) in the tests.
- test_perf deferred start-up: known flaky, rerun alone.
- Blank-text UI bug not reproduced; hardening only.
- Unverified: hydro plants/tanks absent (relations not read); fps noisy.
