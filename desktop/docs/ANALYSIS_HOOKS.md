# Analysis tools: hooks and data (desktop-analysis, integrated)

Tools: measure distance / area, elevation profile (cross-section), point / building depth series, flood timeline, storm compare
(swipe + difference layer). Code: `bgv/analysis/*` (pure python, unit tested), `bgv/ui/panels_analysis.py` (interface),
`bgv/gpu/layers/compare.py` (swipe / difference drawing), `bgv/analysis/layer.py` (`AnalysisLayer`, state + background services).

## Wiring after the integration

The original branch edited `ui/manager.py`, `ui/menus.py` directly. In `desktop-integration` those edits are replaced by ONE file,
`bgv/ui/ext_analysis.py`, that registers on the canonical `bgv/ui/registry.py` (see `EXTENSION_API.md`):

| Hook | Registry call | Function |
|---|---|---|
| Analysis menu | `register_top_menu("Analysis", ...)` | `panels_analysis.menu` |
| tools window + map overlays | `register_panel("analysis", ..., order=40)` | `panels_analysis.draw` |
| view rectangle while the UI is hidden | `register_overlay(...)` | `panels_analysis.draw_hidden` |
| Esc / Enter / Backspace while a map tool is active | `register_key_hook(...)` | `panels_analysis.on_key` |
| click on the 3D view while a map tool is active | `register_click_hook(...)` | `panels_analysis.handle_click` |

`register_top_menu`, `register_overlay`, `register_key_hook`, `register_click_hook`, `register_shortcuts` were added to `registry.py`
for this integration (whole top-level menus, always-drawn overlays, raw key / click events). Still direct edits in shared files:

* `bgv/gpu/layers/registry.py`: `bgv.gpu.layers.compare:CompareLayer` (after the water layer), `bgv.analysis.layer:AnalysisLayer` (before the overlay).
* `bgv/gpu/layers/terrain.py`: `TerrainLayer.update` ORs `app.compare_wet` into the wet-chunk table; `WaterLayer.draw` calls
  `app.compare_hook.water_begin / water_end` (swipe scissor; the difference mode hides storm A's water).
* `bgv/ui/panels_side.py`: `HIDDEN_LAYERS += ("compare", "analysis")` (driven from their own window).

## Depth data

`depth_source.make_depth_source(app)` -> `depth_at(lng, lat, storm, hour)` metres. Tiles where they exist (county, 6.25 m), the flow grid
for the city and as fallback. Integration changes: the tile folder follows the shared policy (`settings["cog_folder"]` / env
`BGV_COG_FOLDER`, aliases `tile_dir` / `BGV_TILE_DIR`); COG tile blocks are read by `bgv/data/cog.py` through the one shared tile cache
(`cog.TILE_CACHE`) when the file is a readable COG (other rasters, e.g. the DEM, still use rasterio); the label is
'solver depth (tiles, 6.25 m; verification pending)' until `WEB_COG_PROGRESS.md` has a line `Status: verified` (the older extra
markers were dropped).

## Tile vs flow grid

`python -m bgv.analysis.verify --out docs/ANALYSIS_TILE_VS_GRID.md` writes the comparison (point depths, domain totals); the generated
report is `ANALYSIS_TILE_VS_GRID.md` in this folder. The sidebar headline numbers still use the flow grid; tile numbers differ
(fewer flooded buildings), see `TILES.md`.
