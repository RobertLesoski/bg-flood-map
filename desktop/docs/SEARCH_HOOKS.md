# Search, reporting and help - what to merge

Branch `desktop-search` (worktree `agent-a84b47387ba530a41`), based on master `36fc3ec`. Almost everything is NEW files; the shared
files get five small hooks, each wrapped so a failure in the search code can never take the interface down.

## New files (no merge conflict possible)

| File | Role |
|---|---|
| `bgv/search/addrindex.py` | address index loader (single file or manifest + shards), typeahead, per-storm stride-5 arrays |
| `bgv/search/tiles.py` | depth tiles (county COG `max_depth.tif`): point / footprint / neighbourhood sampling, source labels, tiles-vs-index comparison |
| `bgv/search/assess.py` | per-address assessment: index values + tile values + comparison; building footprints (OSM way id -> polygon) |
| `bgv/search/facilities.py` | critical facilities, effective depth (tiles else index), first flooded storm, type styles |
| `bgv/search/roadrisk.py` | flooded-road lists (sort / filter) and way geometry for zoom-to-road |
| `bgv/search/report.py` | self-contained HTML risk report + PNG |
| `bgv/search/helpdoc.py` | Help topics, shortcut list, markdown renderer for `STORM_DATA_SOURCES.md`, data-file status |
| `bgv/search/controller.py` | state + background loading (one worker thread) behind the panels; no imgui |
| `bgv/search/shot.py` | developer tool: headless renders of the search UI (`python -m bgv.search.shot --help`) |
| `bgv/ui/panels_search.py` | search bar, typeahead dropdown, result / facility cards, facilities and roads windows, markers, first-run hints |
| `bgv/ui/help.py` | searchable Help / About window (Help, Shortcuts, Data sources, About tabs) |
| `tests/test_search.py`, `tests/test_search_gpu.py`, `tests/verify_search.py` | tests and the independent verification tool |

## Hooks in shared files (all marked with a `SEARCH` comment)

1. `bgv/ui/manager.py`
   * `UiManager.frame()`: `self.safe("search", panels_search.draw)` just before `dialogs.draw`, and `panels_search` added to the local import.
   * `UiManager.on_key()`: first statement after the key is parsed: `panels_search.on_key(self, ev)` (Ctrl+F, `/`, M, L, Ctrl+P, Esc).
   * `UiManager._click()`: `panels_search.on_click(self, x, y)` before `bridge.click_select` (a click on a facility marker opens its card; the click still selects the building).
2. `bgv/ui/menus.py`: a new `Find` menu before `Help`; its body is `panels_search.menu(ui)`.
3. `bgv/ui/dialogs.py`: `draw()` now calls `help.draw(ui, controller)` (the new Help / About window) instead of `_help` / `_about`, which
   remain as the fallback when the new window raises. `ui.show_help` / `ui.show_about` keep their meaning.
4. `bgv/gpu/layers/overlay.py`:
   * `post_frame`: `panels_search.post_frame(app, m)` first (writes a queued risk report: HTML + PNG screenshot of the view).
   * `shutdown`: stops the search worker (`ctl.close()`).

Nothing else was touched: no change to `app.py`, `state.py`, the layer registry, the GPU code, the data layer or the existing tests.

## Settings / environment

* `settings["search_ui"]` (written through `app.settings`, persisted by the existing `collect_settings`): `markers_on`, `types_off`, `onboarded`.
* `settings["tiles_dir"]` or env `BGV_TILES_DIR`: folder with `<domain>/<storm>/max_depth.tif` (default `<data root>/layers/cog`). The cell size,
  bounds and nodata are read from the raster, so finer native tiles can be dropped in. `settings["tiles_verified"] = true` removes
  "verification pending" from the label.
* `settings["report_dir"]` or env `BGV_REPORT_DIR`: where reports go (default `Documents\BG_Flood_Reports`).
* env `BGV_FORCE_HINT=1`: show the first-run hints even on an offscreen canvas (they are suppressed there so renders / tests stay clean).

## Data read (all optional; every absence gives a message, never an error)

`layers/address_index.json.gz`, `layers/county/address_index.json.gz`, `layers/infrastructure.geojson`, `layers/county/infrastructure.geojson`,
`layers/roads_flooded/<domain>/<storm>.json`, `layers/county/roads.geojson.gz` (way geometry, covers the city too), `layers/cog/county/<storm>/max_depth.tif`,
the cached OSM county scene (`desktop/cache/osm/*/buildings.geojson`, for footprints) and `STORM_DATA_SOURCES.md` in the repository root
(rendered by the Help window when present; re-checked every 5 s).

## Merge advice

Take the new files as they are. For the five hooks, the hunks are 2 to 8 lines each; if a later change moved those functions, re-insert the
marked lines. The `tests/test_search*.py` files only depend on the public functions named above.
