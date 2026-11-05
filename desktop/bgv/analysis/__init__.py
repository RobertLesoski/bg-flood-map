"""Analysis tools of the desktop viewer: measurement, elevation profile, point / building depth series, flood timeline, storm compare.

Pure-python modules here (numpy / rasterio, no imgui, no wgpu) are unit-tested; `layer.py` is the app integration (AnalysisLayer) and
`bgv/ui/panels_analysis.py` the interface; `bgv/gpu/layers/compare.py` draws the swipe / difference modes.

Depth data: `depth_source.make_depth_source(app)` -> `depth_at(lng, lat, storm, hour)`; COG tiles (county, 6.25 m, read-only windowed
reads, folder configurable by settings['cog_folder'] / env BGV_COG_FOLDER (aliases tile_dir / BGV_TILE_DIR)) with the flow grid for the city and as fallback.  Tiles are
labelled 'verification pending' until WEB_COG_PROGRESS.md carries `Status: verified`.
`python -m bgv.analysis.verify [--out file.md]` prints the tile-vs-grid comparison (100+ points per case, domain-wide numbers).

HOOKS: wired through ONE file, bgv/ui/ext_analysis.py, onto bgv/ui/registry.py (menu, panel, overlay, key and click hooks); the
remaining direct edits are two lines in gpu/layers/registry.py, the compare hooks in gpu/layers/terrain.py and HIDDEN_LAYERS in
ui/panels_side.py.  Full list: desktop/docs/ANALYSIS_HOOKS.md.
"""
