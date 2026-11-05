# HANDOFF: Bowling Green / Warren County flood viewer (continue on another computer with Claude Code)

Written 2026-11-05 about 10:20 AM CST by Claude (Sonnet 5.5) for the next Claude Code session. Read this file first,
then `COUNTY_RUN.md`, `WEB_COG_PROGRESS.md`, `DESKTOP_COG_PROGRESS.md`, `STORM_DATA_SOURCES.md`, `docs/COG_VERIFICATION.md`.
The user is Robert Lesoski (WKU). Nothing in this file contains a password or token: never ask for or store them; the
user types their own TACC password and 6-digit token when needed.

## 0. What this project is
Flood modelling for Bowling Green city (2 m solver grid) and all of Warren County KY (5 m solver grid, 125 M cells), 12 storms:
design storms 2yr 5yr 10yr 25yr 50yr 100yr and historical events ev1..ev6. The solver (`flood_solver_v2.py`, rain-on-grid, SCS curve
number infiltration, one GPU) ran on TACC Vista (project ATM25008, ~7,633 SU left on 2026-11-04, expires 2027-08-31).
Two viewers consume the results: a web viewer (`viewer.html`) and a native GPU desktop viewer (`desktop/`, python + wgpu + imgui).

## 1. FIRST: the work is NOT all in git and NOT on the new computer
Nothing from this session was committed or pushed (last commit on master: `36fc3ec`). Remote: `origin https://github.com/RobertLesoski/bg-flood-map.git`.
- 64 changed/untracked paths in the main tree (uncommitted): desktop tile depth probe + polish, web COG reader, tile verification, storm docs.
- Three finished feature branches exist only in LOCAL git worktrees under `.claude/worktrees/` (committed on their branches, not pushed):
  `desktop-cinematic` (tip df121d7), `desktop-analysis` (ad97ee9), `desktop-search` (b48c415). Branch tips may have moved; run `git branch -v`.
- The DATA lives outside the repo: `C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map\` (app folder; `layers\` has flow grids, 12 county COG tile sets
  ~5.5 GB, address index, infrastructure, roads_flooded, county OSM buildings/roads, `dem_warren_5m.tif` 490 MB, storm_info.json). Raw solver rasters
  (~94 GB) are only on Vista: `/scratch/11931/rlesoski/bgflood/runs/county_v2/<storm>/` and `.../runs/output_v2/` (city).
To move to another computer, on THIS computer the user (or you, on their word) must first do one of:
  a) commit everything on a new branch and push (recommended; do NOT commit `scripts/storms_out/` cache files blindly, `desktop/user_settings.json`, `.claude/`,
     `layers/cog/*.tif`), push the three feature branches too (`git push origin desktop-cinematic desktop-analysis desktop-search`), or
  b) copy the whole `BG_3D_Map` folder (including `.claude\worktrees`) and the app folder to the other machine (about 7 GB without the raw rasters).
Large data (COG tiles, DEM, flow grids) should NOT go to git; copy `layers\` separately (zip/USB/cloud). They are reproducible from Vista but slow to fetch.

## 2. State of each piece
### Done and verified (but uncommitted)
- **COG tiles (county, 12 storms).** Local `layers\cog\county\<storm>\` (27 files each: 25 hourly depth_*s.tif, max_depth.tif, cog_meta.json) is byte-identical
  to Vista `county_cog_tol` (Slurm job 1123323, GDAL warp tolerance 0.01 px). Better than Vista `county_cog` (job 1122676: errors up to ~19 cm). Keep the tol set.
  Verification (22,421 points, all 12 storms): tiles match expected bilinear resample to 1-2 cm max; vs the single solver cell under a point mean abs 7-8 cm deep
  (re-gridding scatter, not bias). Details: `docs/COG_VERIFICATION.md`, `layers/cog/verification.json`. Tiles are 6.25 m Mercator (~4.99 m ground) vs solver 5.0 m UTM 16N:
  same resolution, rotated grids. No finer tiles exist; a native-UTM COG would only remove smoothing.
- **Web viewer (`viewer.html`, `sw.js`, `cog/cog-depth.js`).** City/County chooser, PWA shell, phone layout, colour-blind palette, storm scrubber with ev1-ev6,
  address search + risk card, infrastructure overlay, at-risk road colouring, swipe, rule-based query box, real-depth overlay + click query (verified vs rasterio:
  516/516 points), "About this storm" panel (data in `layers/storm_info.json`, built by `scripts/build_storm_info.py` from `STORM_DATA_SOURCES.md` facts).
  Known failures: layer-harness has 6 pre-existing failures (pitch-60 terrain alignment, zoom-18 density), not caused by this work and not diagnosed.
  Not verified: service worker in a real browser, real phone, the swipe divider and county road colouring by eye.
- **Storm documentation:** `STORM_DATA_SOURCES.md` (all 12 storms; rain files vs NOAA Atlas 14 within 0.01 mm; run_summary facts; CN 75 everywhere except CN 98 inside the city).
- **Data builders:** `scripts/build_address_index.py`, `build_infrastructure.py`, `build_county_layers.py`, `make_cogs.py`, `fetch_hifld.py` (untested URLs), `fetch_svi.py` (untested),
  `fetch_census.py` (written by the stopped demographic agent: not run/reviewed). `python -m pytest scripts/tests -q -p no:cacheprovider` was 267 tests
  (server tests may time out when the PC is busy; rerun alone).

### Desktop viewer: built, in 4 places, NOT merged
Main tree (uncommitted, edits `desktop/bgv/...`): native tile reader (`bgv/data/cog.py`, `tiles.py`, `depthapi.py`), cursor probe with solver depth, building card with tile footprint max,
sidebar "Solver depth tiles" tile-vs-grid comparison, extension registry (`bgv/ui/registry.py`, `docs/EXTENSION_API.md`), visual polish (sun-aware sky, water highlight fix,
window facades, cividis palette option, dark/light UI theme), docs `desktop/docs/TILES.md`, perf in `desktop/docs/PERF.md` (city 104 fps, county 106 fps synced, noisy machine).
Experimental overlay layer `gpu/layers/tiledepth.py` is UNREGISTERED because registering it caused 33 test failures (imgui "Missing PopID" assert in the sidebar) and a county hitch.
The full desktop suite was NOT confirmed green after the final edits (last log: `%TEMP%\bgv\suite4.log`, 40% through with no failures).
Branches (each in its own worktree; cut from master 36fc3ec):
- `desktop-cinematic`: bookmarks (Ctrl+B / Ctrl+1..9), tour (T), time-lapse (L), PNG/MP4 export, tile depth readouts. Hooks in `bgv/ui/hooks.py` + 3 shared files. 25 CPU + 12 GPU tests.
- `desktop-analysis`: measure, profile, point/building series, flood timeline, storm compare + difference layer (`bgv/analysis/*`, `bgv/ui/panels_analysis.py`), 5 shared files hooked
  (`registry.py`, `terrain.py`, `manager.py`, `menus.py`, `panels_side.py`). Its markdown docs were never written (harness refused): hook list is in `bgv/analysis/__init__.py`.
- `desktop-search`: address search (city+county), facilities layer, roads-at-risk list, risk report export, help/onboarding/cheat sheet (`bgv/search/*`, `bgv/ui/panels_search.py`, `bgv/ui/help.py`). Last known commit b48c415; final report never received.
**NEXT STEP (the integration, ~1-2 h):** the three branch patches do NOT apply cleanly on the main tree: they all touch `desktop/bgv/ui/manager.py`, `ui/menus.py`, `gpu/layers/registry.py`, `desktop/README.md` (search also `ui/dialogs.py`),
and the main tree has its own extension registry (`bgv/ui/registry.py`) while each branch used its own hooks. Suggested approach: commit the main tree on a branch first, then merge the branches one at a time
(cinematic, analysis, search), reconciling their hooks onto `bgv/ui/registry.py` (read `desktop/docs/EXTENSION_API.md`), run `python -m pytest desktop/tests -q -p no:cacheprovider` from the repo root after each merge
(about 8-17 min; ~400 tests; known flaky tests: `test_perf.py::test_deferred_start_up_builds_the_same_app_in_steps_behind_a_loading_screen`, a window-close thread timing test; rerun alone before calling a regression), keep county >= 60 fps at 1080p.

### Findings you should surface to the user
- Tiles vs the old 20 m flow grid disagree on totals at 100yr hour 12: flooded buildings 1,432 (tiles) vs 1,903 (grid); wet road km 185 vs 273; wet area 375 vs 479 km2; volume agrees ~2%.
  The desktop sidebar headline still uses the grid, labelled "full precision": decide whether to headline the tiles (recommended, with "verification pending" until `Status: verified` is added to `WEB_COG_PROGRESS.md`).
- Facilities: tiles flag 33-34 of 98 facilities >= 10 cm vs 26-29 in the old index (pond cells inside campus outlines averaged down by the 20 m grid).
- County DEM `dem_warren_5m.tif` is USGS 3DEP 1/3 arc-second (~10 m) resampled to 5 m; real terrain detail outside the city is ~10 m. City 2 m DEM covers ~13% of the grid.
- Model is NOT validated against observed flooding; building footprints read 2-3 cm of ponding everywhere (rain-on-grid), even in ev1; flow velocities in the viewers are DERIVED (Manning from water-surface slope), not solver output. Honesty labels must stay.

### Not started / stopped
- **Demographic exposure report** (flood exposure by race, sex, median household income via Census ACS block groups for Warren County FIPS 21-227): the agent was stopped before delivering. `scripts/fetch_census.py` exists (unreviewed). The user approved the Census public downloads (ACS 5-year API + TIGER block groups, ~100-300 MB). Brief: use the 6.25 m tiles, dasymetric allocation of block-group population to residential buildings, MOEs,
  ecological-fallacy caveat, outputs `DEMOGRAPHIC_EXPOSURE_REPORT.md/.html` and `layers/demographics/`. No Census data has been verified or used.
- **City COG tiles:** city solver rasters exist (`runs/output_v2/{5,10,25,50,100}yr`), no city COGs built. The user said no new Vista runs for now.
- **Higher resolution runs:** analysed, not requested. City is already 2 m. County 2 m (~783 M cells) will not fit one GPU; 4 m (~196 M cells) is ~21 h per storm, ~195 node-hours for 12 storms; 3 m ~490 node-hours.
  SU rate per node-hour on Vista is unknown (Slurm bills 72/node-hour): confirm with TACC docs or a one-minute test before any large request.
  Queue waits for the GPU partition `gh` were ~34-41 h for the last big arrays; `gg` (CPU) waited 14-55 min.

## 3. Vista access (TACC) on a new computer
- Login: `ssh rlesoski@vista.tacc.utexas.edu` (user types password + 6-digit token). Scratch: `/scratch/11931/rlesoski/bgflood/` (254 GB used). Python env: `$SCRATCH/miniforge3`.
- The previous session used a multiplexed SSH connection through WSL Ubuntu (`~/.ssh/config` host `vista`, ControlMaster auto, ControlPersist 12h). To recreate in WSL: add the `Host vista` block (HostName vista.tacc.utexas.edu, User rlesoski,
  ControlMaster auto, ControlPath ~/.ssh/cm/%r@%h, ControlPersist 12h, ServerAliveInterval 60), `mkdir -p ~/.ssh/cm`, then the user runs `ssh -fN vista` once and authenticates; afterwards `ssh -o BatchMode=yes vista "<cmd>"` works without credentials until the token window expires.
- PowerShell quoting gotcha: `$` and nested quotes get mangled passing through `wsl -- ssh ...`; keep remote commands simple or put multi-line logic in an LF `.sh` file, scp it, and run it. Windows `scp` once failed with "Corrupted MAC": retry or `-o Ciphers=aes256-ctr -o MACs=hmac-sha2-256`.
- Scripts: `scripts/vista_make_cogs.sh` (fixed: the generated job file now quotes the storm list; first attempt job 1122498 died in 12 s from the unquoted `|`), `make_cogs.py`, `convert_county_all.sh`, `vista_convert_county.sh`, `vista_climate_scenarios.sh` (never run). Never submit jobs without the user's go-ahead.
- Solver history from Slurm: county `warrenv2` job 1114427 (gh, 72 CPU, 3.3-10.8 h per storm), city `bgflood5v2` job 1102881 (gh, 16.3-19.7 h per storm).
- Other recorded Vista artefacts: `county_flow` (converted viewer grids, 12 storms), `county_cog` (job 1122676 set, superseded), `county_cog_tol` (final set), `county_cog_tol_test` (one test file).

## 4. How to run things
- Python with rasterio/pytest on the old PC: `C:\Users\Minec\anaconda3\python.exe`. Scripts tests: `python -m pytest scripts/tests -q -p no:cacheprovider`. Desktop tests: `python -m pytest desktop/tests -q -p no:cacheprovider` from the repo root.
- Desktop viewer: `cd desktop; python -m bgv --domain county` (loading screen offers City/County; first county read of the 154 MB Kentucky OSM file takes ~50 s then caches). Flags: `--domain`, `--county-dem`, `--county-pbf`, `--county-demo` (demo data is stamped DEMO DATA).
  Requires a GPU with wgpu support and the data folder located via `bgv/config.data_root()`; set env or settings if the folder path differs on the new computer. Tile folder: `settings["cog_folder"]` or env `BGV_COG_FOLDER` / `BGV_TILE_DIR`.
- Web viewer: run `Launch Map.bat` (serves the app folder with `serve.py`, which supports HTTP Range). A test server may still be listening on port 8123.
- Memory notes from the old PC live in `C:\Users\Minec\.claude\projects\C--Users-Minec-Documents-BG-3D-Map\memory\` (HPC viewer status, Vista solver runs incl. rain file m/s unit gotcha, desktop viewer). Copy them to the new machine's project memory if wanted.

## 5. Working agreements with the user (carry these over)
- Be honest about what is verified: every number must come from a run executed; list "NOT verified" items. The model is unvalidated; do not call tiles "accurate" beyond "faithful to the solver output".
- Ask before: committing/pushing, any Slurm submission or cancel, deleting data, downloads (state source and size), sending anything outside the machine. Never type or store the user's password/token (they once pasted a password in chat; they were told to reset it).
- The user prefers short status updates, wanted "maximum resolution" and "no shortcuts", and was told 6.25 m tiles are about the solver's native 5 m. Avoid starting many parallel agents at once: the 5-hour usage limit was hit overnight with 5-6 agents running; agents resume from their progress files when relaunched.
- Subagent progress/log files: `WEB_COG_PROGRESS.md`, `DESKTOP_COG_PROGRESS.md`, `DATA_PROGRESS.md`, `WEB_PROGRESS.md`, `desktop/COUNTY_PROGRESS.md`; in the worktrees: `CINEMATIC_PROGRESS.md`, `ANALYSIS_PROGRESS.md` (maybe missing), `SEARCH_PROGRESS.md`.

## 6. Suggested order for the next session
1. Confirm with the user how the work is being moved (push vs copy) and get the repo + data on the new computer; run both test suites to see the baseline.
2. Integrate the three desktop branches (section 2), fix failures, decide the tile-vs-grid headline question with the user, then commit (with the user's OK).
3. Real-window manual test of the desktop tools; re-measure fps for city and county.
4. Demographic exposure report (new agent), using the 6.25 m tiles.
5. Optional: web viewer extras (real phone test, service worker, layer-harness failures), city COGs, a higher-resolution rerun after the user decides on SU use.
