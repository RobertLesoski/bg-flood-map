# Cinematic tools - progress log (append only)

Branch `desktop-cinematic` (worktree agent-a44d87e8ec6e05408, base master 36fc3ec). Python: `C:\Users\Minec\anaconda3\python.exe`.
Baseline before any change: `python -m pytest desktop/tests -q -p no:cacheprovider` = 347 passed (7.6 min).

## Coordinator update (depth source)
Depth read-outs (time series, flood timeline, bookmark readouts, export captions, tour ranking) come from the real COG depth tiles
`layers/<tile_dir>/<domain>/<storm>/depth_*s.tif` (default `tile_dir` = `cog`; setting `cinematic.tile_dir` or env `BGV_TILE_DIR`),
read with rasterio (read-only, windowed, cached) by `bgv/data/depthtiles.py`.  City has no tiles: flow-grid fallback.
Source label: "solver depth (tiles, 6.25 m; verification pending)" until WEB_COG_PROGRESS.md carries a line `Status: verified`.

## Done so far
1. `bgv/data/depthtiles.py`: TileDepth (windowed rasterio reads, hour = round(sec/3600), max_depth.tif), FlowGridDepth (batched
   fallback, every hour decoded once for all points), SeriesCache (background, batches requests), open_depth_source.
2. `bgv/tools_cinematic.py`: bookmarks, ViewFlight, TimeLapse, tour stops / route / spline path / plan, shots for the exporter.
3. `bgv/export_cinematic.py`: offscreen export job (own render targets, window state restored), honesty stamp, PNG sequence, ffmpeg or script.
4. `bgv/gpu/layers/cinematic.py` (controller layer), `bgv/ui/panels_cinematic.py` (panel, menu, captions, progress bar, keys), `bgv/ui/hooks.py`.
5. Hooks into shared files (one line each): registry.py layer line, manager.py (hooks.on_key / draw_panels / draw_overlays), menus.py (hooks.draw_menus).

## Findings
- The shader already blends the two hourly flow frames around t (flow_at: codes interpolated, each hour decoded, hours mixed), so any
  rate shows continuous water; buildings recolour per snapshot hour (statistics are hourly).
- ffmpeg is not on PATH here, but `imageio_ffmpeg` ships a binary: `find_ffmpeg()` uses PATH first, then that.
- A pipe-captured ffmpeg stderr deadlocks once 64 KB is written: stderr goes to `ffmpeg.log` in the clip folder instead.
- City `infrastructure.geojson` `d_<storm>` (footprint maximum) differs from a 25 m point read of the flow grid; ranking uses the depth source.
