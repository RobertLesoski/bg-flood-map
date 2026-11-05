# Cinematic tools: hooks into shared files (for merging)

Everything lives in new files. The shared files are touched in exactly these places, one or two lines each.

## Shared files edited

| File | Change |
|---|---|
| `bgv/gpu/layers/registry.py` | one line: `"bgv.gpu.layers.cinematic:CinematicLayer"` (before `panels`). The layer draws nothing and has no sidebar section. |
| `bgv/ui/manager.py` | `from . import ... hooks ...`; in `on_key`: `if hooks.on_key(self, ev): return True` (first, before the built-in hotkeys); in `frame()`: `hooks.draw_overlays(self)` in the hidden-interface branch, and `hooks.draw_panels(self)` + `hooks.draw_overlays(self)` after `dialogs.draw` in the normal branch. |
| `bgv/ui/menus.py` | `from . import ... hooks ...`; `hooks.draw_menus(ui)` just before the Help menu. |

If another branch adds its own extension mechanism, replace `bgv/ui/hooks.py` by an adapter that forwards to it; the cinematic module only
calls `register_panel`, `register_overlay`, `register_menu`, `register_key`, `register_shortcuts` (all in `panels_cinematic.register()`).

## New files

- `bgv/ui/hooks.py` - the registry (panel / overlay / menu / key / help rows; idempotent by name; every callback guarded).
- `bgv/data/depthtiles.py` - depth read-outs from COG tiles (rasterio, read-only) with the flow grid as fallback.
- `bgv/tools_cinematic.py` - pure logic (bookmarks, flights, time-lapse, tour route / path / schedule, shots).
- `bgv/export_cinematic.py` - offscreen export job (PNG sequence, MP4 via ffmpeg or `make_mp4.bat` / `.sh`).
- `bgv/gpu/layers/cinematic.py` - controller layer (state persists under `layers.cinematic.state` in `user_settings.json`).
- `bgv/ui/panels_cinematic.py` - panel (Bookmarks | Tour | Time-lapse | Export), Cinematic menu, tour captions and controls, export progress, hotkeys.
- `tests/test_cinematic.py` (CPU), `tests/test_cinematic_gpu.py` (offscreen GPU, real data).

## Hooks API (`bgv/ui/hooks.py`)

```python
hooks.register_panel(name, fn)       # fn(ui): drawn with the normal interface
hooks.register_overlay(name, fn)     # fn(ui): drawn always, also while the interface is hidden (captions, progress bars)
hooks.register_menu(label, fn)       # fn(ui): draws one top-level menu (begin_menu / end_menu inside)
hooks.register_key(name, fn)         # fn(ui, ev) -> True when it consumed a key_down event
hooks.register_shortcuts(name, rows) # rows appended once to the F1 help table
```

## Behaviour other code should know about

- The export runs one frame per drawn UI frame, from the overlay hook (`CinematicLayer.export_tick`), never nested inside `render_scene`.
  It renders with its own scene targets by swapping `app._targets` / `_target_size` / `_depth_view` / `_msaa_view`, `camera.width/height` and
  `render_scale` for the duration of one `render_scene` call and puts the window's current ones back (taken fresh every frame: the quality
  governor may re-create them). A running viewer carries on unchanged.
- While a tour / export / flight runs, the layer sets `app.t`, the camera view and (tour with "escalate storms") the storm. Any camera input
  (`camera.moving`) ends a tour or a flight.
- Hotkeys: Ctrl+B save bookmark, Ctrl+1..9 jump, Alt+1..9 overwrite, T tour, Space pause (tour, time-lapse), N / P next / previous place,
  Esc stops the tour, L time-lapse, C panel, Ctrl+E export tab. The built-in 1..9 storm keys are unchanged (they ignore Ctrl / Alt).
- Settings: `cinematic.tile_dir` (or env `BGV_TILE_DIR`, default `cog`) names the folder under `layers/` that holds
  `<domain>/<storm>/depth_*s.tif`; `cinematic.export_dir` (or env `BGV_EXPORT_DIR`) is the clip folder (default Documents\BG_Flood_Clips).
- Depth label: "solver depth (tiles, 6.25 m; verification pending)" until `C:\Users\Minec\Documents\BG_3D_Map\WEB_COG_PROGRESS.md` has a line
  `Status: verified` (or `Verification: verified`). Synthetic demo domains never read the real tiles.
