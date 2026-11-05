"""Analysis tools of the desktop viewer: measurement, elevation profile, point / building depth series, flood timeline, storm compare.

Pure-python modules here (numpy / rasterio, no imgui, no wgpu) are unit-tested; `layer.py` is the app integration (AnalysisLayer) and
`bgv/ui/panels_analysis.py` the interface; `bgv/gpu/layers/compare.py` draws the swipe / difference modes.

Depth data: `depth_source.make_depth_source(app)` -> `depth_at(lng, lat, storm, hour)`; COG tiles (county, 6.25 m, read-only windowed
reads, folder configurable by settings['tile_dir'] / env BGV_TILE_DIR) with the flow grid for the city and as fallback.  Tiles are
labelled 'verification pending' until WEB_COG_PROGRESS.md carries one of `depth_source.VERIFIED_MARKERS`.
`python -m bgv.analysis.verify [--out file.md]` prints the tile-vs-grid comparison (100+ points per case, domain-wide numbers).

HOOKS into shared files (everything else is new; every edit is marked ANALYSIS):
    bgv/gpu/layers/registry.py   2 lines: bgv.gpu.layers.compare:CompareLayer (after the water layer),
                                 bgv.analysis.layer:AnalysisLayer (before the overlay)
    bgv/gpu/layers/terrain.py    TerrainLayer.update ORs `app.compare_wet` into the wet-chunk table (4 lines);
                                 WaterLayer.draw calls `app.compare_hook.water_begin / water_end` (swipe scissor, difference hides A)
    bgv/ui/manager.py            _click -> panels_analysis.handle_click; on_key -> panels_analysis.on_key;
                                 frame -> panels_analysis.draw / draw_hidden
    bgv/ui/menus.py              panels_analysis.menu(ui) before the Help menu
    bgv/ui/panels_side.py        HIDDEN_LAYERS += ("compare", "analysis")
"""
