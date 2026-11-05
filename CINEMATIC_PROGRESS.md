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

## Status at the deadline (2026-11-05 09:30)
Working and verified (tests `desktop/tests/test_cinematic.py` 25 CPU tests, `test_cinematic_gpu.py` 12 offscreen GPU tests on real data):
- bookmarks (save / rename / delete / overwrite / jump with eased flight, per domain, persisted, Ctrl+B, Ctrl+1..9, Alt+1..9);
- tour (route build in a worker, spline path, captions, hour sweep, storm escalation, pause, N / P skip, cancel by camera input, state restored);
- time-lapse (rate 0.05..12 h/s, range, loop / ping-pong / once, scrub, flow-buffer gate) and a continuity test: the picture at 6.00 +/- 0.01 h
  is almost the picture at 6.00 h (the shader already blended hourly frames; nothing needed to change);
- export: PNG sequence, MP4 through ffmpeg (found on PATH, else the `imageio_ffmpeg` binary), `make_mp4.bat/.sh` otherwise, cancel, progress bar,
  4K frame (3840x2160 verified), window state restored, driven by the UI frame loop; frames inspected by eye (stamp, captions, county tiles tour).
- depth tiles: real `layers/cog/county/<storm>` read with rasterio (windowed, cached), label pending until WEB_COG_PROGRESS.md says verified.

Not done / not verified:
- no hand-driven session in a real on-screen window (all UI paths ran on the offscreen canvas with scripted calls);
- the county GPU path (tiles + tour + export) was verified by script (1280x720 tour clip, captions show the tile label), not by an automated test;
- the export folder is fixed (Documents\BG_Flood_Clips, BGV_EXPORT_DIR or settings), no folder picker;
- buildings recolour per snapshot hour (statistics are hourly), so building colours step while the water is continuous;
- city has no tiles: depth read-outs there use the flow grid ("preview from depth tiers"); several city facilities read ~0 at a 25 m point
  while `infrastructure.geojson` (footprint maximum) says otherwise;
- fps: county benchmark was taken while other agents ran on the same PC (see the final report).
