# Extension API (stable contract for parallel feature branches)

Written by the owner of the main tree (native tiles + visual polish). Other branches (cinematic, analysis, search) build ONLY on
what is listed here and add NEW files (`bgv/ui/ext_<feature>.py`, `bgv/data/<feature>.py`, tests); they do not edit the shared
UI files (`manager.py`, `menus.py`, `panels_*.py`, `app.py`). If you need something that is missing, add it in your own
module and tell the integrator. These signatures will not change.

## 1. Plug-in modules: `bgv/ui/registry.py`

Every `bgv/ui/ext_*.py` is imported once when the interface starts (`registry.load_extensions()`); a module that fails to import is
logged (`desktop/logs/bgv.log`) and skipped. A hook that raises is reported once and skipped; it never stops the frame.
All callbacks receive the `UiManager` (`ui`): `ui.app` (App), `ui.st` (persistent UiState), `ui.L` (layout rectangles, `L.sx0..sy1` is the
3D view), `ui.s` (interface scale: multiply pixel sizes by it), `ui.toasts.push(text, colour, seconds)`, `ui.nav.fly(...)` (camera flights).

```python
from bgv.ui import registry, theme
registry.register_menu_item("Tools", "My thing", lambda ui: ..., shortcut="", checked=lambda ui: bool, enabled=lambda ui: bool, order=100)
registry.register_panel("name", draw(ui), order=100, when=lambda ui: bool)      # floating imgui panel, drawn every frame after the built-ins
registry.register_sidebar_section("key", "Title", draw(ui), order=100, default_open=False)   # collapsible section at the end of the sidebar
registry.register_hotkey("m", fn(ui), mods=(), description="...")                # key_down, not while a text box has focus
registry.register_tool(MyTool())          # registry.Tool subclass, adds a checkable entry to the Tools menu (+ hotkey if `shortcut`)
registry.state(app, "mykey") -> dict      # persistent settings of your feature (saved in user_settings.json under "ext")
```
Menus: `File`, `View`, `Domain`, `Camera`, `Tools` (created automatically when anything is registered), `Help`. Items are appended after the built-in
ones, sorted by `order`.

`registry.Tool` (one tool active at a time, Esc deactivates): `name`, `title`, `shortcut`, `hint`, `active`; hooks `on_activate(ui)`,
`on_deactivate(ui)`, `on_click(ui, x, y, pick) -> bool` (a click that was not a drag; x, y logical pixels; `pick = (X, Y, Z, t)` world ground
point or None; return True to consume it so no building gets selected), `on_move(ui, x, y, pick)`, `draw_overlay(ui)` (every frame while
active), `on_key(ui, key, mods) -> bool`. Activate from code with `ui.activate_tool(name)` / `ui.toggle_tool(name)` / `ui.activate_tool(None)`.

Drawing in the 3D view: project world points yourself with `app.view_proj` (float64 4x4, column-vector convention as in
`gpu/camera.py`) and draw with `imgui.get_foreground_draw_list()` (cheap, no GPU work). The ground height is
`app.terrain.sampler.sample(X, Y)` (metres, un-exaggerated; the scene draws it times `app.exag`). `ui.L.sx0 / sy0 / sx1 / sy1`
is the 3D rectangle in logical pixels.

## 2. Coordinates

`app.terrain.frame` is the `bgv.geo.GridFrame` of the flow grid: `lnglat_to_world(lng, lat)`, `world_to_lnglat(X, Y)`, `world_to_pixel`.
World = metres, +X east, +Y north, origin at the grid centre. `app.camera` is an `OrbitCamera` (`target`, `dist`, `yaw`, `pitch`, `eye()`,
`pick(px, py)`); fly it with `ui.nav.fly(app.camera, X, Y, dist, yaw_deg, pitch_deg, seconds=1.1)`.

## 3. Depth sampling: `bgv/data/depthapi.py` (also as methods on `App`)

```python
app.depth_at(lng, lat, storm=None, hour=None, source="auto") -> DepthSample
app.building_depth(bid, storm=None, hour=None, source="auto") -> DepthSample          # bid = building number (index into Buildings arrays)
app.depth_profile(lngs, lats, storm=None, hour=None, source="auto") -> numpy float64[n]   # metres, NaN = no data; one call for a whole cross-section
app.tile_info(storm=None) -> TileInfo | None      # which native-tile folder serves this storm (None = no tiles)
```
* `storm`: name (`"100yr"`); None = the displayed storm. `hour`: an integer hour 0..24, a float (rounded to the nearest hour file for
  tiles, interpolated like the 3D view for the grid), the string `"peak"` (largest depth over the run) or None = the displayed hour (`round(app.t)`).
* `source`: `"tiles"` native solver tiles (about 5 m, centimetres), `"grid"` the flow-grid depth the 3D view shows (20-25 m cell,
  sqrt-coded, `meta.source == "tiers"` is only approximate), `"auto"` tiles when they exist for this storm and place else grid.
* `DepthSample` (frozen dataclass): `status` one of `"depth"` (water >= 1 cm), `"dry"` (the model says 0), `"nodata"` (outside the
  modelled area), `"unavailable"` (no tiles / no data for that storm or hour); `m`, `cm`, `ft` (None unless depth/dry); `source`
  (`"tiles"` or `"grid"` - what was actually used); `label` (human text: `"solver depth (native tiles)"` or `"flow-grid depth"`);
  `storm`, `hour`, `scope` (`"county"`, `"city"` or `""`), `note` (caveats; show it where you show the number).
* Tiles exist for the COUNTY solver run (`layers/cog/county/<storm>`). The city view also reads them (`scope == "county"`) because the county
  run covers the city: label it, never relabel it as the city run. Domains without tiles return `"unavailable"` for `source="tiles"`.
* Thread safety: these calls are fast when the tile is cached (microseconds to a few ms) and may read a file the first time; the grid path
  can gunzip an hour (about 0.1 s). Call them from a worker thread for bulk work (`depth_profile` is vectorised for the tile source).
* Honesty: always show `label` with a value. The model is not validated; tile and grid numbers differ by design (5 m vs 20 m cells).

## 4. Other shared state you may read (do not mutate)

`app.storm` (name, `meta`, `source`, `sample_t(t, X, Y)`), `app.t`, `app.playing`, `app.domain` (`key`, `label`, `presets`, `demo`),
`app.storms_available`, `app.root`, `app.settings`, `bridge.buildings(app)` (arrays `cx cy area_m2 height_m address land_use sale_price`, world metres),
`bridge.flood(app)` (current statistics snapshot), `bridge.selection(app)` / `bridge.select_building(app, id, fly=False)`,
`app.report_message(text)`. `app.set_storm(name)`, `app.t = hours` and `app.playing` are the supported controls.
Flow depth over time for a building: `bridge.depth_series(app, bid)` (hours 0..24, grid depth).

## 5. Rules for extension code

* No GPU work outside layers; draw overlays with imgui draw lists. If you need a GPU layer, add `bgv/gpu/layers/<name>.py` and ONE line in
  `bgv/gpu/layers/registry.py` (see `DESIGN.md`, Layer API).
* Honesty labels stay: derived velocities, unvalidated model, demo data, tiers preview.
* Settings: only through `registry.state(app, key)`; new keys must tolerate being absent or garbage.
* Tests go in `desktop/tests/test_<feature>.py`; keep them independent of the real data folder where possible (`@pytest.mark.real` otherwise).

## 6. Appearance state you must respect (added with the theme / palette work)

* The interface theme can switch at run time (dark / light, `ui.st.theme_mode`). Read colours from the module at draw time (`theme.TEXT`, `theme.BG2`, `theme.ACCENT`,
  `theme.OK / WARN / ERR`, `theme.LIGHT_ON`) and never cache them across frames; use `theme.col32(...)` for draw-list colours. Panels get the right window colours from the style.
* Tier colours follow the colour palette (`app.palette` = "std" | "cvd"): use `theme.tier_color(tier)` / `theme.TIER_RGB[k]` (re-read each frame), not hard-coded hex.
* `app.tiles` is the `TileService`; `app.tiles.tileset(storm).label` is the text that must accompany a tile depth ("solver depth (tiles, 6.25 m, verification pending)").
* `bgv/gpu/layers/tiledepth.py` (solver depth overlay) exists but is NOT registered (unfinished, see TILES.md); do not rely on `app.get_layer("tiledepth")`.
* `devshot` (`python -m bgv.ui.devshot --out x.png ...`) renders one UI frame headless; new flags `--domain county`, `--theme light`, `--palette cvd`, `--enable <layer>`, `--at lng,lat`.
