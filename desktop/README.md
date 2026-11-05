# BG Flood Viewer (desktop)

A standalone GPU application for exploring the Bowling Green flood simulations: 3D terrain, animated water, buildings
and roads, flood-depth tiers, at-risk building list, statistics and a timeline. It replaces the browser viewer for
day-to-day use and needs no server and no internet connection.

## Start it

Double-click **`Launch Desktop Viewer.bat`** (there is also a shortcut on the desktop). Variants:

| File | Purpose |
|---|---|
| `Launch Desktop Viewer.bat` | Normal start, no console window (pythonw). Shows a message box if Python or a package is missing. |
| `Launch Desktop Viewer (console).bat` | Same, with a console so you can read messages. |
| `Run Desktop Checks.bat` | Runs the tests and a headless smoke render; prints PASS or FAIL. |

Requirements: Windows 10/11, Python 3 (Anaconda is fine), a Vulkan-capable GPU, and
`python -m pip install wgpu rendercanvas glfw imgui-bundle numpy numba scipy rasterio shapely mapbox_earcut pillow requests`.
The app prefers the high-performance (discrete) GPU. It does not change Windows graphics settings.

Data: it reads the same app folder as the web viewer (`layers\flow\<storm>\...`, `layers\*.geojson`). The default is
`C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map`; override with `--data <folder>` or the `BGV_DATA` environment
variable. Caches go to `desktop\cache` (override `BGV_CACHE`); the log is `desktop\logs\bgv.log`.

## Controls

| Input | Action |
|---|---|
| Left-drag | Pan (the ground point under the cursor stays under it) |
| Right- or middle-drag | Orbit and tilt |
| Wheel, `+` `-` | Zoom toward the cursor |
| W A S D / arrows, Q E, R | Move, rotate, tilt; hold Shift for 3x speed; keys glide to a stop; Home resets the view |
| Click a building / row in the list | Select; **F** flies to the selection (tilt otherwise), **Esc** deselects |
| Space, `[` `]` | Play/pause, step an hour back/forward |
| 1 to 9 | Pick storm |
| H | Hide/show the interface |
| F1 / F3 | Help / performance overlay |
| F11 | Fullscreen |
| F12, Shift+F12 | Screenshot (folder: `BGV_SCREENSHOT_DIR`) |

The side panels hold storm and time, layers (terrain, roads, buildings, water, flow arrows, particles, flow paths,
optional map background), water and flow settings, quality, and statistics. The quality governor is **Auto** by default
and adapts render scale and detail to keep the frame rate up; clicking a preset switches to manual.

## Command line

```
python -m bgv                                   open the window
python -m bgv --storm 100yr --msaa 4            start on a storm
python -m bgv --render out.png --storm 25yr --hour 12 --size 1600x900 --ui [--camera "x,y,dist,yaw,pitch" | --at "lng,lat"]
python -m bgv --benchmark --seconds 12 --size 1920x1080 [--json]
python -m bgv --selftest                        open a window, script mouse/keys, measure frames, exit (rc 0 = pass)
python -m pytest desktop/tests -q               306 tests (about 3.5 minutes)
```

Run these from the `desktop` folder (or set `PYTHONPATH` to it).

## What it measured (GTX 1650, 1600x900 window)

About 90 fps with the full UI, about 3.4 s from launch to fully loaded (window visible in about 0.5 s). Details and
the optimisations behind them are in `docs/PERF.md`; architecture and the layer API are in `DESIGN.md`.

## Know what you are looking at

- **Velocity is derived**, not solver output: Manning's equation on the simulated water-surface slope (same as the web
  viewer; see `FLOW_FORMAT.md`).
- The current flow data are the **depth-tier preview** (`"source": "tiers"`), labelled "Preview from depth tiers" in the
  app. The 5-year to 100-year storms are available; county-wide Vista output is not convertible yet.
- Tested only on one laptop (GTX 1650, Vulkan) with scripted input, not a physical mouse.

## Presentation

A 20-slide technical deck with live-render video clips was made from this app (private link: https://claude.ai/artifact/L7J2yCrkuLEzmww9VBDxCL). `docs/make_media.py` re-renders the clips: it loads the 100-year storm, renders three camera paths (flood rising over 24 hours, fly-through, orbit) as JPEG frames in `docs/media/<clip>/`, ready to encode with ffmpeg (`-framerate 24 -i %04d.jpg -c:v libx264 -pix_fmt yuv420p`).

## City and county

The viewer has two **domains**: the city (Bowling Green, default) and Warren County (about 52 x 60 km). Pick one on the loading
screen (two big buttons; the remembered choice continues after a 3 s countdown you can cancel, `--domain` skips it, the Domain menu
has "Ask which view at start-up"), or any time in the **Domain** menu / the "Area" box in Storm & Time. The county needs
`layerslow_county\<storm>` (converted flow grids) and the terrain `layers\dem_warren_5m.tif` (UTM 16N, metres; or `--county-dem <file>`);
buildings and roads come from `kentucky-latest.osm.pbf` (read once, about a minute, then cached). Without them the County button is greyed
out and its tooltip says what is missing. Command line: `--domain city|county`, `--county-dem`, `--county-pbf`, `--county-demo`.

`python -m bgv --make-county-demo <folder>` writes SYNTHETIC county terrain + 2 storms (marked DEMO DATA everywhere, screenshots stamped);
view with `python -m bgv --domain county --county-demo <folder>`. Real data always takes precedence.

## Solver depth tiles (real solver depth, 6.25 m)

When `layers\cog\county\<storm>\` holds the solver's Cloud-Optimized GeoTIFF depth tiles (12 county storms, 25 hourly + peak each), the viewer reads them
natively (`bgv/data/cog.py`, no GDAL needed) and adds, next to the unchanged flow-grid numbers: the cursor probe's **solver depth** (hour + peak, in cm and ft,
"no water" is different from "no data"), the selected building's solver depth (highest tile cell under the footprint), and a sidebar section **Solver depth tiles**
with a comparison of flooded buildings counted by the tiles vs by the flow grid. The city view uses the county run for the same storm (labelled). Until the
tile verification is recorded the label reads "solver depth (tiles, 6.25 m, verification pending)". Switch: View menu or the sidebar; finer tiles later:
`settings["cog_folder"]` / env `BGV_COG_FOLDER`. Details, verification numbers and the differences to the flow grid: `docs/TILES.md`.

## Appearance and extension points

View menu: **Interface theme** (dark / light) and **Colour palette** (Standard tiers / Colour-blind safe, the web viewer's cividis ramp: tier colours of buildings,
roads, legend and lists). Both are remembered. Look: sun-dependent sky and horizon, aerial perspective tinted toward the sun, calmer and less glaring shallow
floodwater, window facades on close buildings (`docs/shots/polish/`). Other features plug in through `bgv/ui/registry.py` (menu items, panels, tools, hotkeys,
sidebar sections, one file `bgv/ui/ext_<name>.py` each) and the depth API `app.depth_at / building_depth / depth_profile`: see `docs/EXTENSION_API.md`.
