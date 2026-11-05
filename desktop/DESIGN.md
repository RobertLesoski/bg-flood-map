# BG Flood Desktop Viewer ("bgv") - design contract

Standalone GPU application that shows the flow-grid flood data (`FLOW_FORMAT.md` in the repo root) as a 3D fluid with
flow vectors, buildings and roads coloured by flooding, and live statistics. It replaces the browser viewer for
performance. It reads the same data files; it does not touch the web viewer.

## Stack (all installed in `C:\Users\Minec\anaconda3\python.exe`, Python 3.14)
- **wgpu 0.32 (wgpu-py)**, Vulkan backend, `request_adapter_sync(power_preference="high-performance")` -> GTX 1650 (4 GB).
  NOT OpenGL: the OpenGL context lands on the Intel iGPU on this hybrid laptop. Always print the adapter name at startup.
- **rendercanvas 2.7** (`from rendercanvas.auto import RenderCanvas, loop`; glfw backend) for the window; `rendercanvas.offscreen.OffscreenRenderCanvas` for headless.
- **imgui-bundle 1.93** for UI, drawn by wgpu-py's `wgpu.utils.imgui.ImguiRenderer(device, canvas)` (see
  `imgui_bundle/python_backends/examples/example_python_backend_wgpu.py`, wgpu-py `examples/imgui_*`).
- numpy 2.4, numba 0.65, scipy, rasterio, shapely 2.1, `mapbox_earcut`, PIL, requests. No node, no compiler.
- Shaders are **WGSL**. Compute shaders are available (workgroup size up to 1024, storage buffers up to 2 GB).
- Read the installed wgpu / rendercanvas source in `C:\Users\Minec\anaconda3\Lib\site-packages\` when an API is unclear;
  the wgpu 0.32 API differs from older tutorials (e.g. `adapter.info`, `device.queue.write_texture`, `create_render_pipeline`
  argument shapes) - test small snippets before building on a guess.

## Folders
- Source (git, source of truth): `C:\Users\Minec\Documents\BG_3D_Map\desktop\`   package `bgv`, tests in `desktop\tests`
- Data root (read-only for the app): `C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map`
  `layers\flow\<storm>\meta.json + hNN.rgba.gz + flowpaths.rgba.gz`, `layers\buildings.geojson` (20,284 polygons, props incl.
  `height_m`, `ADDRESS`, `sale_price`, `LAND_USE`), `layers\roads.geojson` (516 MultiLineStrings, `highway`), `dem_bg.tif`
  (EPSG:32616, **elevations in FEET**, cells 5.088 m, nodata -9999; read with `scripts/flowgrid.read_dem_m` semantics: x0.3048).
  Resolve with `bgv.config.data_root()` (env `BGV_DATA`, then `--data`, then the path above).
- Cache (generated, safe to delete): `<source>\desktop\cache\` (npz/npy, keyed by input file size+mtime). Never write into the data root.
- Deployed copy (identical files) is made at the end into `<data root>\desktop\`; launchers `Launch Desktop Viewer.bat` there.

## Coordinates: `bgv/geo.py` (already written, do not change semantics)
World = ground metres, +X east, +Y north, +Z up, origin at the grid centre; `GridFrame.from_meta(meta)` converts
mercator / lng-lat / world / uv / pixel. Textures use the flow grid's pixel grid exactly (row 0 north).
Terrain height is metres (feet x 0.3048) un-exaggerated in data; the renderer applies a user exaggeration factor to Z only.

## Data layer (`bgv/data/`)  - pure CPU, no wgpu imports, unit-testable
- `flow.py`  `Storm.open(root, storm)`: parses meta, `.frame`, `.hours`; `load_hour(h) -> uint8[H,W,4]` (gunzip, validate size;
  hours not in meta.hours return one shared read-only zero grid, `isZero` flag); `load_flowpaths()`; thread pool prefetch;
  `decode_depth(R)` (sqrt law), `decode_vel(code)`; `sample(...)` bilinear CPU probe. Same semantics as `fluid/fluid-data.js`.
- `terrain.py` `load_terrain(root, frame) -> float32[H,W]` metres in the flow-grid frame (warp dem_bg.tif UTM16N -> EPSG:3857 grid,
  bilinear, feet->m); NaN where the DEM has no coverage; cached; plus `terrain_stats`.
- `buildings.py`, `roads.py`, `stats.py`: footprints -> world-frame extruded mesh, per-building pixel index (CSR) on the flow grid,
  road segments, and fast numpy/numba flood statistics (rules below). Cached.
- **Data-layer API as built** (full docs in each module's header; the flow grid is the same for every storm, get the frame with
  `bgv.data.prep.default_frame(root)`; everything is cached in `desktop/cache/*.npz`, cold build about 10 s, cached load < 0.1 s):
  `buildings.load_buildings(root, frame) -> Buildings` (arrays `cx cy area_m2 height_m sale_price address land_use ...`, `.mesh`
  = positions f32[V,3] (world X,Y, Z above the building base) / normals f32[V,3] / building_id u32[V] / indices u32 (CCW, no
  bottom), `.index` = CSR `offsets i32[n+1]`, `pixels i32[]`), `buildings.attach_base_z(index, terrain) -> f32[n]`;
  `roads.load_roads(root, frame) -> RoadSet` (segments `seg_x0..seg_y1`, `seg_class`, `seg_width_m`, `seg_sample`, statistics
  samples `sample_pixel`/`sample_weight_m`, `.ribbon` mesh with `across`, `sample_id`);
  `stats.evaluate_buildings(rgba, index, rgba_b=None, mix=0.0) -> BuildingStats` (`depth_m speed_ms tier flooded order ...`),
  `stats.evaluate_roads(rgba, roads, rgba_b, mix) -> RoadStats`, `stats.summarize_grid(rgba, frame) -> GridSummary`,
  `stats.warmup()`; `None` stands for the all-dry grid. `prep.ensure_all(root, frame, progress)` builds/loads all of it
  (worker thread safe). Rasterising rule verified: identical to the exact "pixel square intersects polygon" rule for all
  20,284 buildings, and to rasterio all_touched=True except 5 sliver cases (see `tests/rasterio_crosscheck.py`).
- Statistics rules (identical to `fluid/fluid-stats.js`, verified there against rasterio): a building's footprint pixels =
  ALL-TOUCHED rasterization on the flow grid; building depth = MAX depth over its pixels; flooded = depth >= 0.1 m; tiers
  `shallow [0.1,0.3)`, `moderate [0.3,0.6)`, `deep [0.6,1.2)`, `severe >= 1.2`; pixels with A=0 ignored; road flooded share = fraction of
  10 m samples with depth >= 0.1 m; wet area = cells with depth >= 0.02 m x cell_ground_m^2.
  **Known-good numbers (tiers data, 25yr hour 12): 1309 flooded buildings, tiers {shallow 831, moderate 363, deep 107, severe 8}, wet area 14.06 km2, max depth 5.90 m, roads 2.1 %.
  5yr hour 12: 908 flooded buildings, 6.70 km2, max depth 5.79 m, roads 1.2 %.** Tests assert these.

## GPU layer (`bgv/gpu/`)
- `context.py`: adapter/device selection, limits, timestamp queries when available.
- `camera.py`: orbit camera (target, distance, yaw, pitch), pan/zoom/rotate with mouse (LMB pan, RMB or MMB orbit, wheel zoom to cursor),
  WASD/arrows, Home = reset, perspective, reversed-Z or standard depth, ground ray picking against the heightmap.
- Frame uniforms (one bind group 0 for every layer): view-proj, camera pos, sun dir, time, hour-mix, exaggeration, grid frame constants,
  flow texture-array bindings. Layers add their own groups.
- `layers/`: every visual feature is a class in its own file implementing `Layer` (`layers/__init__.py`):
  `init(app)`, `resize(app,w,h)`, `update(app,dt)`, `draw(app, pass_)` (called in fixed order: terrain, roads, buildings, water,
  particles, arrows, overlay), `ui(app)` (optional imgui widgets in the sidebar), `save_state()/load_state(d)`. Layers register
  in `bgv/gpu/layers/registry.py` with ONE line each so agents do not edit each other's files.
- Time: continuous hour `t` (float). Flow textures live in ONE `rgba8unorm` 2d-array (one layer per hour, only the selected storm
  resident, ~0.9 GB). Shaders sample layers floor(t), floor(t)+1 and mix; depth is decoded from the sqrt code AFTER interpolating codes
  (matches the web renderer). Hours outside `meta.hours` are dry layers.
- Rendering order and look: hillshaded terrain (optional OSM basemap drape) -> roads -> buildings (extruded, tier colour, `height_m`)
  -> water (surface = ground + depth*exag, depth colour ramp, ripples scrolled along the flow, specular from the surface normal,
  foam/brighter at high speed and at shorelines, soft alpha at the wet edge) -> GPU particle streaks (compute advection) ->
  instanced flow arrows -> imgui.
- Performance budget (GTX 1650, 1920x1080, wgpu Vulkan): >= 60 fps with all layers on at default quality, never a frame hitch while
  playing hours or loading a storm (decode in worker threads; upload <= 1 hour layer per frame). Memory: < 3 GB VRAM, < 3 GB RAM.
  Provide quality presets and an auto governor like the web viewer.

## Tools (every agent uses these to check its work; the user's desktop must not be disturbed)
- `python -m bgv --render out.png --storm 25yr --hour 12 --camera "x,y,dist,yaw,pitch" --size 1920x1080` renders ONE frame headless
  (offscreen canvas, no window) and exits. All visual checks are done by rendering PNGs and looking at them with the Read tool.
- `python -m bgv --benchmark [--seconds 10]` renders a scripted camera orbit/zoom offscreen at 1920x1080, prints frame ms
  (mean/p95/p99/max), fps, adapter, VRAM estimate; JSON to stdout with `--json`.
- Never open an on-screen window in automated checks except the single `--selftest` (3 s window, auto close).
- Tests: `python -m pytest desktop/tests -q` from the repo root; GPU tests are marked `@pytest.mark.gpu` and use the offscreen canvas.

## Honesty requirements (same as the web viewer)
- Velocity is DERIVED from the water-surface slope (Manning), not solver output; the UI says so in a visible info line.
- `meta.source == "tiers"` -> show "Preview from depth tiers" badge and show the severe tier as ">= 1.2 m" (no false precision);
  `"rasters"` -> "Full-precision (solver depth rasters)".
- Flooded-building numbers are labelled "as of hour N" using the hour of the snapshot used (round(t)).

## Layer API (stable - written by the CORE agent)
A layer is a subclass of `bgv.gpu.layers.Layer` in its own file under `bgv/gpu/layers/`, listed with ONE line in
`bgv/gpu/layers/registry.py` (`"module:Class"`, in draw order; a line whose module does not exist yet is skipped, so
the planned lines `roads:RoadsLayer`, `buildings:BuildingsLayer`, `particles:ParticlesLayer`, `arrows:ArrowsLayer`,
`panels:PanelsLayer`, `overlay:OverlayLayer` are already in place - just create the file with that class name).
Hooks (all optional except `name`): `init(app)`, `on_storm(app)` (new storm, same grid), `resize(app,w,h)`,
`update(app,dt)`, `compute(app,encoder)` (compute passes before the scene pass), `draw(app,pass_)`, `ui(app)`,
`save_state()/load_state(d)`, `shutdown(app)`. Attributes: `name`, `title`, `enabled`, `default_enabled`, `has_ui`
(True -> a collapsing "<title> options" section in the sidebar calls `ui()`), `draws` (False = UI only).
A layer that raises is disabled and the message goes to the sidebar; it never takes the app down.

```python
# bgv/gpu/layers/example.py  (registered as "bgv.gpu.layers.example:ExampleLayer")
from ..flowtex import COMMON_WGSL            # bind group 0: frame uniforms, flow array, height + slope textures
from . import Layer
class ExampleLayer(Layer):
    name = "example"; has_ui = True
    def init(self, app):
        dev = app.ctx.device
        mod = dev.create_shader_module(code=COMMON_WGSL + MY_WGSL)   # MY_WGSL: vs/fs using frame.view_proj, flow_at(uv), ground_height(xy)
        pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout])
        self.pipe = dev.create_render_pipeline(layout=pl, vertex={"module": mod, "entry_point": "vs"}, fragment={"module": mod,
            "entry_point": "fs", "targets": [{"format": app.colour_format}]}, primitive={"topology": "triangle-list"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": True, "depth_compare": "greater"},  # reversed-Z
            multisample={"count": app.sample_count})
    def draw(self, app, pass_):
        pass_.set_pipeline(self.pipe); pass_.set_bind_group(0, app.frame.bind_group); pass_.draw(3)
```
Shared scene pass facts every layer must respect:
- **Depth is reversed-Z, `depth32float`, infinite far**: clear 0, compare `greater` (`depth = near/distance`). Colour target
  format = `app.colour_format` (an `*-srgb` format: shaders output LINEAR colour), `app.sample_count` = 4 (MSAA, fixed per
  session; build pipelines with it). One pass: terrain, roads, buildings (write depth) -> water (blends, no depth write) ->
  particles / arrows / overlay (blend, depth test, no write).
- **Z scale**: terrain Z in the shaders is `ground_height(xy) * frame.grid.w` (exaggeration E = `app.exag`); water depth is scaled by
  `frame.enc.w` (= E x water boost). Anything standing on the ground (buildings: `height_m`, road drapes) must use the same E:
  `z = (ground_height(p) + extra_metres) * frame.grid.w` so it moves with the slider. Water surface = `ground*E + depth*frame.enc.w`.
- **WGSL helpers** (`COMMON_WGSL`): `frame.*` (see struct comments), `world_to_uv`, `uv_to_world`, `ground_height(xy)` (bilinear metres, un-exaggerated),
  `height_texel(ix,iy)`, `flow_at(uv) -> {depth m, vel m/s (east,north), valid}` at the displayed continuous time (codes interpolated bilinearly,
  each hour decoded, hours blended - identical to the web renderer), `sky_base/sky_color/haze`. `frame.flow = (t, mix, hour0, hour1)`.
- CPU side: `app.t` (hours), `app.flow.chunk_wet`, `app.terrain.sampler.sample(x, y)` / `.raycast`, `app.terrain.frame` (GridFrame),
  `app.storm` (`data.flow.Storm`: `load_hour`, `sample_t(t, X, Y)` for probes), `app.camera.pick(px, py)` (ground point under a window pixel,
  `(x, y, z, t)` or None; z is exaggerated), `app.stats_hook` (callable run once per frame after time/flow update) and `app.stats` (dict shown in the
  sidebar), `app.layers` (ordered), `app.get_layer(name)`, `app.report_error(where, exc)`.
- Flow texture array: layer = hour (0..24), layer 25 = permanent zero layer; `frame.layer_map[h]` says which layer hour `h` reads (dry hours and hours still
  loading read zeros) - always go through `flow_at` / `flow_layer(h)` rather than indexing the array with the hour.

## Implementation notes (CORE)
- Terrain: vertex pulling from the r32float height texture, 128x128-texel chunks, 6 LODs (128..4 quads per chunk), per-frame CPU frustum cull + LOD, one
  instanced indexed draw per LOD, cracks closed in the vertex shader (edge vertices snap to the coarser neighbour's grid). Lighting uses a rgba16float
  slope map (dz/dE, dz/dN, curvature, DEM-coverage) with mips, a 26-step heightfield shadow trace toward the sun, hypsometric tint and world-space detail noise.
- Water: second pass over only the chunks whose wet table (per hour, computed in the decode threads) is non-empty at floor(t) / floor(t)+1.
- Data corrections vs the first draft: `queue.on_submitted_work_done` is broken in wgpu-py 0.32 (cffi callback type error) - `GpuContext.sync()` uses a tiny
  buffer readback instead; the slope map replaces the 8-bit normal map (8-bit normals band on this very flat ground).
- Camera CLI: `--camera "x,y,dist,yaw,pitch"` is WORLD metres from the grid centre (not lng/lat); `--at "lng,lat"` moves the target; yaw = compass heading looked toward.

## Input, window lifecycle, failures (INTERACT)
- **Event order** (`App.attach_canvas`): order -100 releases (`pointer_up`, `key_up`: always reach the camera, even when imgui swallows the
  event, so a drag that ends over a window or a key released while a text box has focus can never stick), -99 wgpu-py's imgui handlers
  (`stop_propagation` while imgui wants the mouse / keyboard), -98 our gates (`pointer_down` / `wheel` over the UI rectangle of the last frame,
  `key_down` while a text box is active: wgpu-py only swallows letters / digits / named keys, space and `[ ] , . -` would leak), 0 camera + hotkeys.
- **Camera** (`gpu/camera.py`): LMB grabs the ground point under the cursor and keeps it there (plane through the grabbed point, 3 fixed-point
  iterations: exact on hilly ground); RMB / MMB orbit around the target (0.25 deg / px, pitch 4..89.5); wheel zoom is eased and keeps the
  picked ground point under the cursor; keys / pan / orbit glide are exponential and integrated analytically over dt (any frame rate gives the
  same distance; an exact stop, no creeping); `release_keys()`, `end_drag()`, `stop_motion()`; the eye always stays >= 3 m above the exaggerated
  terrain (also when the exaggeration slider changes).  Camera width / height are fractional logical pixels (125 / 150 % DPI).
- **Hotkeys** (`ui/manager.py on_key`): F12 / Shift+F12 screenshot (folder: `settings["screenshot_dir"]`, env `BGV_SCREENSHOT_DIR`, else
  Documents\BG_Flood_Screenshots), F11 fullscreen, F3 perf, F1 help, H hide UI, Esc (dialog -> hidden UI -> selection), `[` `]` hours, F fly to
  selection, Home overview, 1..9 pick the n-th storm.  A click is a press + release that moved <= 4 px (`CLICK_SLOP_PX`) in < 0.6 s.
- **Failures**: `App.report_message` = sidebar line + one red toast per 6 s per message + log.  A hour file that cannot be read reads dry and
  is reported once as "Hours 7, 12 and 13 ... could not be read" (`flow.failed`, `progress()` counts it as resolved); missing / unreadable
  buildings or roads switch those two layers off with a message; a layer that raises is disabled; a frame that raises is reported and retried
  (`draw_frame`), 120 failed frames in a row close the window and `run_window` raises `fatal_error`.
- **Start-up errors**: `__main__` -> `applog.fatal` (traceback to `desktop/logs/bgv.log`, 200 kB x 3 rotating; `BGV_LOG_DIR` overrides) + a short
  native message box (`user32.MessageBoxW`) in window mode, plain text on stderr for the CLI tools; `BGV_NO_DIALOG=1` suppresses boxes.
  `--data` that is wrong is an error (no silent fallback); `BGV_DATA` is only a default.  `BGV_CACHE` / unwritable `desktop/cache` -> temp folder.
- **Launchers** (desktop/): `Launch Desktop Viewer.bat` (pythonw, no console; `--console` or `Launch Desktop Viewer (console).bat` for debugging;
  waits only when started from a console), `Run Desktop Checks.bat` (pytest + headless smoke render, prints PASS / FAIL), helper `_bgv_env.bat`.
  Never use `%~dp0` after `shift` in a batch file (it follows `%0`).
- **Tests**: `tests/test_interaction.py` (camera CPU tests; real canvas events through `canvas.submit_event` + `canvas._process_events()`;
  subprocess scenarios via `tests/interaction_driver.py`).

## Domains (city / county)
`bgv/domain.py`: a `Domain` says where flow grids (`layers/flow` | `layers/flow_county`), terrain (`TerrainSource`: city `dem_bg.tif` feet; county any
GeoTIFF, metres, warped lazily by GDAL with area averaging onto the flow grid) and building / road GeoJSON come from. The county's are cut from the
OSM PBF by `data/osmpbf.py` (pure python + numpy: zlib, hand-decoded protobuf, vectorised varints; dense nodes + building/highway ways, no multipolygon
relations) and cached by `data/countyscene.py` in `<cache>/osm/<key>/`. Terrain textures stay on the FLOW grid (2631 x 3052, ~20 m): r32 height 32 MB +
slope map 85 MB, so the county uses LESS VRAM than the city (1130 vs 1253 MB incl. 835 MB flow array); the 5 m DEM is averaged, not shown at 5 m.
`App.set_domain(key)` drops scene / stats hub, shuts layers, closes flow + terrain, rebuilds for the new grid behind the loading screen (preparation in a
`Prefetch` thread); live layer/UI state is carried; per-domain camera/exag/storm in `settings["domain_views"]`. Start-up chooser: `ui/chooser.py`
(`DomainChooser`: builds only the remembered domain; clicking the other restarts the build; waits for the countdown only when both domains have data).
Demo data (`data/demo.py`): meta.json `"synthetic": true` -> DEMO DATA badges, stamped screenshots.

## Solver tiles, palette, theme, extensions (added later)
- `bgv/data/cog.py` COG reader (shared byte-capped LRU, vectorised sampling), `bgv/data/tiles.py` TileService (tile sets per storm / scope, footprint pixel index on the
  tile grid built through `buildings_from_features(index_only=True)` and cached, per-building footprint maximum, `DepthSample`), `bgv/data/depthapi.py` (the stable
  `app.depth_at / building_depth / depth_profile / tile_info / tiles`), `ui/tilepanel.py` (building-card rows, comparison table, start-up tick), `ui/probe.py`
  (tile read inside the probe worker thread, never on the render thread). Tile values are always shown with `TileSet.label`; tile numbers never replace grid numbers.
- `bgv/palette.py` tier palettes ("std", "cvd"); `App.palette` drives `ui.theme.set_palette`, the buildings uniform block and the roads uniform block (the road tier
  colours moved from WGSL constants to the uniform: `R.tier[i]`). `ui.theme.set_mode("dark" | "light")` + `apply_colours`.
- `bgv/ui/registry.py`: extension registry (see `docs/EXTENSION_API.md`); `ui/ext_*.py` modules are imported once at UI start.
- Shaders: `COMMON_WGSL` `sky_base` is sun-aware (warm horizon band toward a low sun, Mie-like bloom that also tints `haze`); the water fragment has a calmer shallow
  specular and a highlight shoulder; building walls have a procedural window grid that fades out under ~4 px per window.
