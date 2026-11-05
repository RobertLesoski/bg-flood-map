# HANDOFF: Bowling Green / Warren County flood viewer (continue on another computer with Claude Code)

Updated 2026-11-05 about 1:45 PM CST by Claude (Sonnet 5.5) for the next Claude Code session. Read this file first, then
`COUNTY_RUN.md`, `WEB_COG_PROGRESS.md`, `DESKTOP_COG_PROGRESS.md`, `STORM_DATA_SOURCES.md`, `docs/COG_VERIFICATION.md`,
`docs/VALIDATION_EV4.md`, `desktop/docs/EXTENSION_API.md`.
The user is Robert Lesoski (WKU). Nothing in this file contains a password or token: never ask for or store them; the user
types their own TACC password and 6-digit token when needed (they once pasted a password into chat and were told to reset it).

## 0. What this project is
Flood modelling for Bowling Green city (2 m solver grid) and all of Warren County KY (5 m solver grid, 125 M cells), 12 storms:
design storms 2yr 5yr 10yr 25yr 50yr 100yr and historical events ev1..ev6. The solver (`flood_solver_v2.py`, 2D local-inertial
"LISFLOOD-FP acceleration" scheme, rain-on-grid, SCS curve-number infiltration, one GPU) ran on TACC Vista (project ATM25008,
7,633 SU left on 2026-11-04, expires 2027-08-31). Two viewers consume the results: a web viewer (`viewer.html`) and a native GPU
desktop viewer (`desktop/`, python + wgpu + imgui).

## 1. Where the work is (git) and what to do first on the new computer
Repo: `C:\Users\Minec\Documents\BG_3D_Map`, remote `origin https://github.com/RobertLesoski/bg-flood-map.git` (only `master` exists there, at e79e730).
NOTHING from this session is pushed. Local branches:
- `desktop-integration` (tip 6244ccd) = the desktop viewer with everything merged: main-tree tile work + three feature branches + storm descriptions. This is the branch to continue from.
  Commits: 5b3d956 (desktop tile reader/probe/polish/registry), e07822d (web COG reader, storm info panel, COG verification scripts/data),
  ffa4db3 (progress logs, earlier handoff, storm data docs), 53c011b (merge desktop-cinematic), 41e2954 (merge desktop-analysis),
  039d274 (merge desktop-search), 6244ccd (storm descriptions in picker + Help, tile-vs-grid footnote, integration screenshots, perf notes).
- `desktop-cinematic` df121d7, `desktop-analysis` ad97ee9, `desktop-search` b48c415: already merged into desktop-integration; kept only for reference (their worktrees are under `.claude/worktrees/`, do not copy those).
- `master` is local only, 36fc3ec, "ahead 4, behind 1" versus origin/master e79e730 ("Update README.md" exists only on the remote). Do NOT push master as is; pull/rebase first if you ever want to update it. `worktree-agent-*` branches are leftovers at e79e730 (safe to delete).
UNCOMMITTED in the working tree (about 23 paths, commit them on desktop-integration with the user's OK):
- Validation: `docs/VALIDATION_EV4.md`, `docs/VALIDATION_EV4_map.png`, `VALIDATION_PROGRESS.md`, `scripts/validate_ev4.py`, `scripts/tests/test_validate_ev4.py`; data in the app folder `layers\validation\` (ev4_observations.json, ev4_scores.json, raw\).
- Web storm picker descriptions: `scripts/build_storm_blurbs.py`, `scripts/tests/test_storm_blurbs.py`, `layers/storm_blurbs.json`, edits to `viewer.html`, `sw.js`, `WEB_PROGRESS.md`, and a 3-line edit to `desktop/bgv/ui/registry.py`. A follow-up task (change the storm-card honesty wording to the real validation result with an expandable ev4 section) was STOPPED part-way: check `git diff viewer.html` for half-finished edits and finish or revert. The viewer.html/sw.js copy in the app folder must stay byte-identical to the repo copy (`cmp`).
- Partial documentation draft (NOT finished): `docs/report_src/01_summary_data.md`, `02_method_storms.md`, `03_verify_compare.md` and 7 SVG figures (`docs/figures/fig1..fig7`, `make_figures.py`). No `DOCUMENTATION.md/.html/.pdf` was assembled and no 'comparison with public flood models' chapter was verified to exist.
- Untracked and best left out of git: `.claude/`, `BG_FloodMap_Report.html`, `scripts/storms_out/` cache files, `desktop/user_settings.json` (rewritten every run).
- `INTEGRATION_LOG.md` was never written (the integrator agent was interrupted), so there is no recorded full-test result for the merged tree.
THE DATA IS OUTSIDE THE REPO: `C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map\` (about 7 GB): `layers\` has flow grids, 12 county COG tile sets (5.5 GB), address index, infrastructure, roads_flooded, county OSM buildings/roads, `dem_warren_5m.tif` (490 MB), storm_info.json, storm_blurbs.json, validation\. It is not in git and must be copied by hand (USB, cloud drive, `robocopy`). The raw solver rasters (about 94 GB) exist only on Vista.
To move: (1) commit the loose files above on `desktop-integration` and `git push -u origin desktop-integration` (only with the user's OK; do not push master); (2) copy the app/data folder; (3) clone on the new machine, `git checkout desktop-integration`, put the data folder in place and point `bgv/config.data_root()` (setting or env) at it; (4) install the Python environment (rasterio, numpy, wgpu, imgui bindings, Pillow, pytest; see `desktop/README.md`) and run both test suites; (5) re-create the Vista SSH setup (section 3).

## 2. State of each piece
### Done (desktop viewer, branch desktop-integration)
- Main-tree work: native tile reader (`bgv/data/cog.py`, `tiles.py`, `depthapi.py`), cursor probe with solver depth, building card with tile footprint max, sidebar 'Solver depth tiles' panel (tile-vs-grid flooded-building comparison, with a footnote that tile numbers differ from the headline grid numbers), extension registry (`bgv/ui/registry.py`, `docs/EXTENSION_API.md`), visual polish (sun-aware sky, water highlight fix, window facades, cividis palette option, dark/light UI theme), flooded roads coloured by depth tier.
- Merged features: cinematic (bookmarks Ctrl+B / Ctrl+1..9, tour T, time-lapse L, PNG/MP4 export), analysis (measure, profile, point/building series, flood timeline, storm compare + difference layer; `bgv/analysis/*`, hooks in `bgv/ui/ext_analysis.py`, `docs/ANALYSIS_HOOKS.md`), search (address search, facilities layer, roads-at-risk list, risk report export, help/onboarding; hooks in `bgv/ui/ext_search.py`, `docs/SEARCH_HOOKS.md`). Key change in the merge: the roads-at-risk list moved from `L` to `Ctrl+L` because `L` is the time-lapse.
- Storm descriptions in the desktop storm picker and Help (from the storm blurb/info JSON). Integration screenshots (city and county, 9 views each) are in `desktop/docs/shots/integration/`; perf notes were appended to `desktop/docs/PERF.md` (read it for the numbers; earlier noisy-machine figures were about 104 to 106 fps synced at 1080p for city and county).
- Experimental overlay layer `desktop/bgv/gpu/layers/tiledepth.py` is UNREGISTERED: registering it made 33 UI tests fail (imgui 'Missing PopID' assert in a sidebar section) and caused a county hitch. Root cause not found.
- Test status: before the merges the main tree had about 361 passed; cinematic branch alone 383 passed + 1 timing failure; the merged tree's full result was NOT seen by me. Run `python -m pytest desktop/tests -q -p no:cacheprovider` from the repo root (8 to 17 min). Known flaky: `test_perf.py::test_deferred_start_up_builds_the_same_app_in_steps_behind_a_loading_screen` and a window-close thread-timing test: rerun alone before calling a regression.
- NOT verified: nobody clicked the new tools by hand in a real window (all checks ran on the offscreen canvas); city fps after the merge; light theme beyond one visual check; tile-vs-grid headline question (below).

### Done (tiles and documentation, uncommitted/committed as listed above)
- COG tiles (county, 12 storms): local `layers\cog\county\<storm>\` (27 files each) is byte-identical to Vista `county_cog_tol` (Slurm job 1123323, GDAL warp tolerance 0.01 px); the other Vista set `county_cog` (job 1122676) is less accurate (errors to ~19 cm) and superseded. Verification of 22,421 points: tiles match the expected bilinear resample within 1-2 cm; against the single solver cell under a point mean abs 7-8 cm deep (re-gridding scatter, not bias). See `docs/COG_VERIFICATION.md`, `layers/cog/verification.json`. Tiles are 6.25 m Mercator (~4.99 m ground) vs solver 5.0 m UTM 16N: same resolution, rotated grids. No finer tiles exist.
- Web viewer (`viewer.html`, `sw.js`, `cog/cog-depth.js`): City/County chooser, PWA shell, phone layout, colour-blind palette, storm scrubber ev1-ev6, address search + risk card, infrastructure overlay, at-risk road colouring, swipe, rule-based query box, real-depth overlay + click query (516/516 points identical to rasterio), 'About this storm' panel (`layers/storm_info.json`), new storm-picker descriptions for all 12 storms with real-world context links (uncommitted). Known: layer-harness has 6 pre-existing failures (pitch-60 terrain alignment, zoom-18 density). Not verified: service worker in a real browser, real phone, links and offline mode of the new cards, city view of the new cards.
- Storm documentation: `STORM_DATA_SOURCES.md` (all 12 storms; rain files vs Atlas 14 within 0.01 mm; the design-storm temporal curve is the first 2-year run's curve, NOT the NRCS Type II table; CN 75 everywhere except CN 98 inside the city).
- `python -m pytest scripts/tests -q -p no:cacheprovider` was 272 passed after the storm blurbs; the validation tests add 10 (server tests can time out when the PC is busy: rerun alone).

### Validation against the April 2025 flood (ev4), done 2026-11-05
Details in `docs/VALIDATION_EV4.md`. 63 observation items found (41 road items, 11 property items, 6 rain items, 2 gauges, 1 damage count of 196 structures); only 21 roads could be scored by exact name. Model rain is 4.97 in over 24 h vs about 10.5 in over four days at the airport; no upstream inflow.
- Roads reaching 30 cm anywhere: 16 of 21 reported roads (76%, Wilson 95% CI 55-89%) vs 27 of 100 random city roads; at 10 cm all 21 vs 42 of 100.
- AUC of road maximum depth 0.845 (bootstrap 95% CI 0.77 to 0.91; permutation p about 0.0002 versus chance). Median road max depth 0.62 m vs 0.08 m.
- IMPORTANT CAVEAT found while computing these: road length alone separates the groups as well (AUC 0.85; reported roads median 1,330 m vs 413 m for random roads), so the model's AUC may partly reflect road length. The agent's subset check (roads under 2 km, AUC 0.84) argues against that but was not independently verified.
- This is NOT a validation of depths. Observations give closure/presence only. Next steps proposed: rerun April 2025 with the full four-day real rain (needs a new Vista run, user's OK), compare against more observed points, document how to interpret.

### Not started / stopped
- Demographic exposure report (flood exposure by race, sex, median household income via Census ACS block groups for Warren County FIPS 21-227): the agent was stopped before delivering. `scripts/fetch_census.py` exists, unreviewed and never run. The user approved the public Census downloads (ACS 5-year API + TIGER block groups, about 100-300 MB). Use the 6.25 m tiles, dasymetric allocation of block-group population to residential buildings, margins of error, ecological-fallacy caveat; outputs `DEMOGRAPHIC_EXPOSURE_REPORT.md/.html`, `layers/demographics/`.
- The 20-page documentation report the user asked for (data lineage, method, comparison with public flood models) is a partial draft only (see section 1). Instruction given to the writer: do NOT claim the model is more accurate than other products; the supportable claim is 'a transparent, high-resolution pluvial screening tool that complements, and does not replace, regulatory maps; not yet shown to be more accurate than any of them'.
- City COG tiles: city solver rasters exist (`runs/output_v2/{5,10,25,50,100}yr` on Vista), no city COGs built. The user said no new Vista runs for now.

### Decisions waiting on the user
- Headline numbers: the desktop sidebar headline still uses the 20 m flow grid labelled 'full precision' while tiles give fewer flooded buildings at 100yr hour 12 (1,432 tiles vs 1,903 grid; wet roads 185 vs 273 km; wet area 375 vs 479 km2; volume agrees ~2%; the grid caps depth at 8 m vs 10.27 m on tiles). Recommendation: headline the tiles with 'verification pending' until the user signs off (add `Status: verified` to `WEB_COG_PROGRESS.md`).
- Facilities: tiles flag 33-34 of 98 facilities at >=10 cm vs 26-29 in the old index (pond cells inside campus outlines averaged down by the 20 m grid).
- Whether to spend SU on a four-day April 2025 rerun, city tiles, or a higher-resolution county run (see section 3).

### Facts and caveats to surface (honesty labels must stay in the viewers)
- The model is NOT validated for depth; rain-on-grid ponds 2-3 cm of water everywhere (even in the 2yr and ev1), so only >=10 cm counts as flooding; ponded water never infiltrates later; no storm drains, no upstream river inflow, no advection term, uniform rain, CN 75/98 not calibrated, hourly snapshots only (`max_depth` = max of snapshots).
- Flow velocities and flow paths in the viewers are DERIVED from the water-surface slope with Manning, not solver output.
- County DEM `dem_warren_5m.tif` is USGS 3DEP 1/3 arc-second (~10 m) resampled to 5 m; real terrain detail outside the city is ~10 m. The 2 m city DEM covers ~13% of the grid.
- The solver equations (from the code): local-inertial update `Q_new = (Q - dt*g*h*d(eta)/dx) / (1 + dt*g*n^2*|Q|/h^(7/3))`, depth update from rain minus flux divergence, CFL 0.3 time step capped at 30 s, wet threshold 1 mm, free outflow at the boundary (normal depth, slope 0.005), SCS runoff on cumulative rain with Ia = 0.2 S.

## 3. Vista access (TACC) on a new computer
- Login: `ssh rlesoski@vista.tacc.utexas.edu` (user types password + 6-digit token). Scratch: `/scratch/11931/rlesoski/bgflood/` (254 GB used). Python env: `$SCRATCH/miniforge3`. No jobs were queued at last check.
- The previous session used a multiplexed SSH connection through WSL Ubuntu (`~/.ssh/config` host `vista`: HostName vista.tacc.utexas.edu, User rlesoski, ControlMaster auto, ControlPath ~/.ssh/cm/%r@%h, ControlPersist 12h, ServerAliveInterval 60; `mkdir -p ~/.ssh/cm`). The user runs `ssh -fN vista` once and authenticates; afterwards `ssh -o BatchMode=yes vista "<cmd>"` works without credentials until the token window ends. The old session's login has probably expired; never type credentials for the user.
- PowerShell quoting gotcha: `$` and nested quotes get mangled passing through `wsl -- ssh ...`; keep remote commands simple or write an LF `.sh` file, scp it, run it. Windows `scp` once failed with 'Corrupted MAC': retry or `-o Ciphers=aes256-ctr -o MACs=hmac-sha2-256`.
- Scripts: `scripts/vista_make_cogs.sh` (fixed: the generated job file now quotes the storm list), `make_cogs.py`, `convert_county_all.sh`, `vista_convert_county.sh`, `vista_climate_scenarios.sh` (never run). Never submit jobs without the user's go-ahead.
- Solver history from Slurm: county `warrenv2` job 1114427 (gh, 72 CPUs, 3.3-10.8 h per storm, 125 M cells, about 94 node-hours total), city `bgflood5v2` job 1102881 (gh, 16.3-19.7 h per storm, 80 M cells at 2 m, about 90 node-hours). Queue waits on the GPU partition `gh` were about 34-41 h; the CPU partition `gg` waited 14-55 min.
- Resolution/cost analysis (estimates, scaled from the real runs): city is already 2 m; county 2 m would be about 783 M cells (too big for one GPU as the solver is written); 4 m about 196 M cells, about 21 h per storm, about 195 node-hours for 12 storms; 3 m about 490 node-hours. The SU rate per node-hour on Vista is unknown (Slurm bills 72 per node-hour): confirm with TACC docs or a one-minute test before any large request. GPU jobs probably cap at 48 h wall time (not verified).
- Validation cost estimate given to the user: Tier 1 compare to existing observations (free, 1-3 weeks), Tier 2 calibration plus storm drains/upstream inflow/spatial rain (1-3 months, roughly 150-400 node-hours), Tier 3 field surveys (money and time).
- Other Vista artefacts: `county_flow` (converted viewer grids, 12 storms), `county_cog` (job 1122676, superseded), `county_cog_tol` (final), `county_cog_tol_test` (one test file).

## 4. How to run things
- Python with rasterio/pytest on the old PC: `C:\Users\Minec\anaconda3\python.exe`. Scripts tests: `python -m pytest scripts/tests -q -p no:cacheprovider`. Desktop tests: from the repo root `python -m pytest desktop/tests -q -p no:cacheprovider`.
- Desktop viewer: `cd desktop; python -m bgv --domain county` (loading screen offers City/County; the first county read of the 154 MB Kentucky OSM file takes ~50 s then caches). Flags: `--domain`, `--county-dem`, `--county-pbf`, `--county-demo`. Needs a GPU with wgpu support. Tile folder: `settings["cog_folder"]` or env `BGV_COG_FOLDER` (old names `BGV_TILE_DIR` / `tile_dir` still accepted as aliases per the integration brief; check `bgv/data/cog.py`). Shortcuts: B/T/L cinematic tools, Ctrl+L roads-at-risk list.
- Web viewer: run `Launch Map.bat` in the app folder (serves with `serve.py`, supports HTTP Range). A test server may still be listening on port 8123 on the old PC.
- Memory notes from the old PC are in `C:\Users\Minec\.claude\projects\C--Users-Minec-Documents-BG-3D-Map\memory\` (HPC viewer status, Vista solver runs incl. rain file m/s unit gotcha, desktop viewer); copy them to the new machine's project memory if wanted.

## 5. Working agreements with the user (carry these over)
- Be honest about what is verified: every number must come from a run executed; list 'NOT verified' items. Never call the model 'validated' or 'accurate'; tiles are 'faithful to the solver output', and the April 2025 comparison is 'modest evidence of where water collects, not depth accuracy'.
- Ask before: committing/pushing, any Slurm submission or cancel, deleting data, downloads (state source and size), sending anything outside the machine. Never type or store the user's password/token.
- The user prefers short status updates and wants 'no shortcuts', but the 5-hour usage limit was hit twice with 5-6 agents running at once (resets about every 5 hours; agents resume from their progress files via SendMessage). Prefer 2-3 agents at a time and never two agents editing the same files; use git worktrees on separate branches for parallel feature work.
- Current user instruction (2026-11-05 ~1:30 PM): work ONLY on the desktop viewer for now (web, documentation and validation work paused). Progress/log files: `WEB_COG_PROGRESS.md`, `DESKTOP_COG_PROGRESS.md`, `DATA_PROGRESS.md`, `WEB_PROGRESS.md`, `desktop/COUNTY_PROGRESS.md`, `VALIDATION_PROGRESS.md`.

## 6. Suggested order for the next session
1. Confirm with the user how the work is being moved; get repo + data on the new computer; run both test suites for a baseline (the merged desktop tree has no recorded green run).
2. Commit the loose files on `desktop-integration` (user's OK), finish or revert the interrupted viewer.html wording edit, push the branch.
3. Desktop viewer: real-window manual test of every new tool, fix what breaks, re-measure fps for city and county on the new machine, decide the tile-vs-grid headline question with the user, write `INTEGRATION_LOG.md`.
4. Then, if the user asks: finish the documentation report, the demographic exposure report, the four-day April 2025 rerun, city tiles.
