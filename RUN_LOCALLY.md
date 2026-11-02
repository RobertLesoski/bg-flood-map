# Running the Bowling Green Flood Map on your computer

This is the short, plain-language guide. You do not need to be a programmer. (It lives in the code folder, next to
`scripts\`; the launcher and the checks refer to it as `RUN_LOCALLY.md`.)

## Quick start

1. Open the app folder: `C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map`
2. Double-click **Launch Map.bat**.
3. A red-and-black window shows the DSOC banner. A second window titled **DSOC Flood Map Server** opens: that is the
   web server, and it must stay open while you use the map.
4. Type **1** and press Enter. Your browser opens the map at an address such as `http://127.0.0.1:8000/viewer.html`.
5. When you are finished, close the **DSOC Flood Map Server** window. That stops the server.

You need: Windows 10 or 11, Python 3.8 or newer (Anaconda counts), a current Chrome, Edge or Firefox, and an internet
connection (see "Troubleshooting" for why).

Everything is served only to your own computer (`127.0.0.1`). Nobody else on your network can open it, and Windows
will not ask about the firewall.

## The desktop app (faster, no browser)

There is also a standalone desktop viewer in the code folder: double-click **`desktop\Launch Desktop Viewer.bat`** (or the
"BG Flood Viewer" desktop shortcut). It reads the data from the app folder below and needs no server. Controls, the
command line and its limits are in `desktop\README.md`.

## Two folders

| Folder | What it is |
|---|---|
| `C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map` (the **app folder**) | What you double-click. Holds the viewer, the report, all data and the server. |
| `C:\Users\Minec\Documents\BG_3D_Map` (the **code folder**, a git repository) | The Python and shell scripts that build the data (`scripts\`), this guide, and `FLOW_FORMAT.md` (the data format). |

Inside the app folder:

| Item | What it is |
|---|---|
| `Launch Map.bat` | Starts the server and opens the map or the report. |
| `Run Checks.bat` | Checks that everything is in place (see below). |
| `serve.py`, `selfcheck.py` | The server and the checker (plain Python, nothing to install). |
| `viewer.html` | The 3D flood risk map. |
| `flood_report.html` | The flood probability report and TXT export. |
| `layers\*.geojson` | Map layers: buildings, parcels, FEMA zones, sinkholes, roads, rivers. |
| `layers\flood_hpc\<run>\flood_hNN.geojson` | Hourly flood layers (depth classes) used by the time slider. |
| `layers\flow\<storm>\` | Fluid view data: one depth-and-flow grid per simulated hour (see below). |
| `fluid\*.js` | The browser code that draws the moving water and the flow vectors. |
| `dem_bg.tif` | Terrain (LiDAR), used when rebuilding flow data. Not needed just to view the map. |
| `BG FLOOD MAP VISTA SUPERCOMPUTER\v2\<storm>\` | Raw output downloaded from the Vista runs (depth classes per hour, UTM coordinates). |

## What the launcher does

Double-click for the menu:

| Key | Opens |
|---|---|
| 1 | 3D Flood Risk Map |
| 2 | Flood Probability Report and TXT Export |
| 3 | Both |
| 4 | Checks this installation and prints a report, then shows the menu again |

Before it starts the server the launcher quickly checks that the data are in place and prints one summary line. If
something is missing it tells you; the map still opens.

It never closes other programs. If port 8000 is already used by something else it quietly uses the next free port
and shows you the address. If the map server for this folder is already running it reuses it.

For scripts and shortcuts, give it options instead of using the menu:

```
Launch Map.bat 1                  open the map
Launch Map.bat 2                  open the report
Launch Map.bat 3                  open both
Launch Map.bat 4                  run the checks only (no server)
Launch Map.bat 1 /nobrowser       start the server, print the address, open nothing
Launch Map.bat 1 /port 8200       prefer port 8200
Launch Map.bat 1 /nopause         never wait for a key (see the next paragraph)
```

After an error (folder incomplete, Python not found, server did not start, unknown option) and after option 4, the
window waits for a key, so a shortcut does not close before you can read the message. A script that calls the
launcher should add `/nopause`; the exit code is the same either way. Asking for help (`/?`) never waits.

To change the default port for every start, set the environment variable `DSOC_PORT` (for example `8200`; it must be
a number from 1024 to 65535, otherwise it is ignored with a message). `/port` wins over `DSOC_PORT`.

Exit codes: `0` fine, `10` bad option, `11` folder incomplete, `12` Python not found, `13` server did not start,
`14` the checks found a problem (option 4).

## Where the flood data come from

- **Solver.** The depths come from our GPU flood solver (local-inertial approximation, Bates et al. 2010) run on the
  TACC Vista supercomputer: a 2 m grid of 10,000 by 10,000 cells (about 20 km by 20 km around Bowling Green), 24 hours of rain with totals
  within a few percent of NOAA Atlas 14 for the 5-, 10-, 25-, 50- and 100-year storms. The v2 runs include soil infiltration (SCS curve
  numbers) and free outflow at the domain edge. The first run (2-year) had neither.
- **What the solver saves.** One water-depth raster per simulated hour, plus the depth classes (shallow, moderate,
  deep, severe) as polygons. The folder `BG FLOOD MAP VISTA SUPERCOMPUTER\v2\` holds the polygons that are already on this
  computer.
- **Flat time-slider layers.** `scripts\reproject_hpc.py` converts those polygons to latitude/longitude and writes
  `layers\flood_hpc\<run>\flood_hNN.geojson`. Example: `python scripts\reproject_hpc.py v2_all`.
- **Fluid view data.** `layers\flow\<storm>\` holds the depth-and-flow grids described next.

## How the fluid view gets its depths and flow vectors (read this before quoting it)

The fluid view draws the water as a moving surface with flow arrows and streaks.

- **Depth is simulated.** It is the solver's water depth for each hour. The full-precision build uses the solver's
  depth rasters. The quick local preview (`"source": "tiers"` in `meta.json`) only has the four depth classes, so depth
  is the class midpoint (0.2, 0.45, 0.9, 1.5 m) smoothed; depth is approximate and the viewer labels it
  **Preview from depth tiers**.
- **Velocity is derived, not simulated.** The solver does not save velocity. We compute it afterwards from the
  slope of the simulated water surface (ground height plus depth) with Manning's equation: for each cell edge,
  `speed = depth^(2/3) * sqrt(slope) / n`, directed from the higher surface to the lower one, with roughness `n` from the
  solver's roughness map where available and a typical 0.035 otherwise. Speed is limited to the physical bound
  `sqrt(g * depth)` and to 4 m/s. A cell's velocity is the depth-weighted mean of its edges. In words: it is the
  steady-state balance of the solver's momentum equation with the inertia term left out.
- **What that means for you.** The arrows are good for "where does the water go and roughly how fast". They are
  not solver output and not a substitute for it: fast changing flow, flow through pipes and culverts, and
  anything between two hourly snapshots is not captured (the viewer blends neighbouring hours smoothly).
  The grid files and the viewer say "derived" for this reason, and any numbers you quote should say so too.

The exact file format is in `FLOW_FORMAT.md` in the code folder.

## Rebuilding the flow data

The flow grids are large and are not stored in git. Each storm gets its own folder, `layers\flow\<storm>\`
(`5yr 10yr 25yr 50yr 100yr`), with `meta.json`, `h01.rgba.gz` to `h24.rgba.gz` (hours that are dry are left out) and
`flowpaths.rgba.gz`. If the checks say `flow data not built yet: see RUN_LOCALLY.md`, build them with this script
(run from the code folder; the options are documented at the top of the script):

| Script | What it does |
|---|---|
| `scripts/build_flow_from_tiers.py` | On this computer: builds **preview** grids from the depth-class polygons already in `BG FLOOD MAP VISTA SUPERCOMPUTER\v2\`. Works offline. Approximate depths. Example: `python scripts\build_flow_from_tiers.py --storms 5yr` |

**Full-precision grids are not built by any script yet.** They would come from the solver's 2 m depth rasters instead
of the four depth classes (`"source": "rasters"` in `meta.json`). The pieces that exist: `scripts/fetch_v2.sh` downloads
the depth rasters from Vista (about 10 GB per storm), and `scripts/flowgrid.py` holds the functions a builder needs
(velocity from the water surface, aggregation to the viewer grid, the pixel format). `python scripts\flowgrid.py validate
layers\flow\<storm>` checks a finished storm folder, whichever way it was built. Until a raster builder is written the
viewer shows the preview data and labels them **Preview from depth tiers**.

The builder needs extra Python packages (`numpy`, `scipy`, `rasterio`, `shapely`; Anaconda has them). The server,
the launcher and the checks need nothing extra.

After rebuilding, run **Run Checks.bat**. The server asks the browser to re-check every file each time, so a rebuilt
storm is picked up when you reload the page (**F5**). If a map tab that was open during the rebuild still shows old
water, reload it; that page keeps the grids it already loaded.

## Checking the installation

Double-click **Run Checks.bat** (or choose 4 in the launcher). It prints a table with `OK`, `WARN` or `FAIL` per item:

- `FAIL` means something the map needs is missing or damaged (for example `viewer.html`, a map layer, an hourly flood
  layer that was cut off, or a flow grid whose size does not match its `meta.json`). The "What to do next" list says
  how to fix it.
- `WARN` means an optional part is missing, such as flow data that has not been built yet, or a storm that is not
  built yet. The map still works.
- `Run Checks.bat --deep` also reads every flow grid completely and verifies its checksum. It takes longer and catches
  files that were damaged in the middle of a download.
- `Run Checks.bat --json /nopause` prints the result for other programs. Add `/nopause` to skip the final keypress.

## Troubleshooting

**"Python 3.8 or newer was not found."** Install Python from https://www.python.org/downloads/ and tick **Add python.exe
to PATH**, or install Anaconda. If Python lives somewhere unusual, set the environment variable `DSOC_PYTHON` to the
full path of `python.exe`.

**The map page is blank or only grey.** The data files are all local, but the page also downloads the MapLibre library
(from unpkg.com) and the base map and terrain tiles (OpenStreetMap and Amazon S3) from the internet. Without a connection
the page cannot draw. Check the connection, then press Ctrl+F5.

**"The map server did not start within 10 seconds."** Read the **DSOC Flood Map Server** window; it stays open after an
error. Try a different port: `Launch Map.bat /port 8200`.

**The address is not 8000.** Another program uses port 8000, so the server took the next free port. Use the address
the launcher prints. Nothing was closed.

**Browser says "connection refused" later.** The server window was closed. Run `Launch Map.bat` again.

**The map is slow or the browser tab crashes.** The computer has 8 GB of memory. The viewer loads large layers
(buildings 12 MB, parcels 27 MB, each hourly flood layer up to roughly 115 MB) and one fluid grid is up to 144 MB unpacked;
the fluid view keeps a few hours in memory, up to roughly 0.7 GB for the largest storms. Close other tabs and programs,
use one browser, and try the 5-year storm first (the smallest flooded area).

**The fluid view says it needs a newer browser.** It uses WebGL 2 and built-in gzip decoding (Chrome or Edge 80+,
Firefox 113+, Safari 16.4+). Update the browser.

**The checks say the flow data are damaged.** A download was cut off or a build did not finish. Rebuild or fetch that
storm again (see above). `--deep` shows which hour file is broken.

**The checks say an hourly flood layer "ends abruptly" or "does not start like JSON".** The file
`layers\flood_hpc\<run>\flood_hNN.geojson` was cut off, or it is a saved error page instead of data; the map cannot read
it. Delete that file and run `python scripts\reproject_hpc.py <run>` again (`v2_all` for the solver-v2 storms, which only
redoes the missing hours). If the downloaded UTM file it is made from is gone too, fetch it again with
`scripts/fetch_geojson_v2.sh` (in the Ubuntu terminal).

**I only want to see what the server is doing.** Open `http://127.0.0.1:8000/__health` (use your port). It lists the
server's folder, Python version, uptime and, per storm, whether the flow data are present.

## For scripts and developers

```
python serve.py [--port N] [--host 127.0.0.1] [--no-browser] [--open PAGE] [-v]
python serve.py --find [--wait SECONDS]     print the URL of the server already running for this folder (exit 1 if none)
python serve.py --spawn                     open the server in its own console window and return immediately
python selfcheck.py [--brief] [--summary] [--deep] [--json] [--strict] [-v] [--root DIR]
```

- The server serves only its own folder, refuses `..`, drive letters, dot-files such as `.git`, and folder listings, and
  only answers requests whose `Host` is localhost.
- `.gz` files are sent as plain `application/gzip` (no `Content-Encoding`) so the browser's `DecompressionStream`
  gets the raw bytes. Range requests, `HEAD`, `ETag` and `304` are supported.
- Cache: every file is sent with `Cache-Control: no-cache`, so the browser revalidates it with its `ETag` each time
  (a `304` when nothing changed). Flow grids are rewritten under the same names by every rebuild, which is why they
  have no `max-age`. The loader still appends `?v=<meta.built>` to grid URLs; the server ignores the query.
- It writes its port and process id to `.serve_port` in the app folder while it runs.
- When you start the launcher from a program that captures its output, use `Launch Map.bat ... /nobrowser`; the server
  window is opened detached, so the caller is not kept waiting.
