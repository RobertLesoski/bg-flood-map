## 3. Method

### 3.1 What kind of model this is

The solver is a **two-dimensional, depth-averaged, local-inertial shallow-water model** with rain applied to every cell (rain-on-grid) and an SCS curve-number loss. The README and `CREDITS_AND_COMPUTE.md` cite Bates, Horritt and Fewtrell (2010, Journal of Hydrology 387, 33 to 45) and de Almeida et al. (2012, Water Resources Research 48, W05528) for the scheme, the same family as the "acceleration" solver in LISFLOOD-FP. Those papers were not fetched in this session (ScienceDirect refused the request), so statements about them below come from the citation list in the repository and general knowledge, and the equations come from the code itself (`scripts/flood_solver_v2.py`, 224 lines), which is the authority. The sections that follow describe what the code does, not what the papers say.

In words: the ground is a grid. Each cell holds a water depth. Water moves across each shared edge at a rate set by the slope of the water surface, reduced by friction. Rain, after the infiltration loss, is added to each cell. Water leaves the model across its outer edge.

### 3.2 Grid and symbols

| Symbol | Meaning | Value or source |
|---|---|---|
| Z | ground elevation of a cell (m) | DEM |
| H | water depth in a cell (m) | solved |
| η | water-surface elevation, Z + H (m) | |
| dx, dy | cell size (m) | 5 m county, 2 m city |
| Qx, Qy | unit discharge across a cell face (m²/s), stored on faces | solved |
| n | Manning roughness (s/m^(1/3)); face value = mean of the two cells | raster |
| g | gravitational acceleration | 9.81 m/s² |
| h_f | flow depth at a face | defined below |
| dt | time step (s) | defined below |
| H_MIN | wet/dry threshold | 1e-3 m |
| r_eff | effective rain after infiltration (m/s) | defined below |
| α | CFL number | 0.3 (`ALPHA`) |
| S0 | energy slope used at the outflow boundary | 0.005 (`OUTFLOW_SLOPE`) |

![Stencil of the local-inertial scheme](docs/figures/fig2_stencil.svg)

*Figure 2. Grid layout and the face update. Depths sit in cell centres; discharges sit on faces between cells.*

### 3.3 The update equations as coded

At every time step, for each pair of neighbouring cells i and j along x (and likewise along y):

1. Surface elevation: η = Z + H.
2. Flow depth at the face: h_f = max(η_i, η_j) - max(Z_i, Z_j), set to zero if negative. This is the depth of water that is above the higher of the two ground levels, so water cannot flow uphill out of a puddle that is below the rim.
3. Face discharge, only where h_f > H_MIN (otherwise Qx = 0):

   Qx(new) = [ Qx(old) - g · h_f · dt · (η_j - η_i) / dx ] / [ 1 + dt · g · n² · |Qx(old)| / h_f^(7/3) ]

   The numerator is the previous discharge accelerated by the surface-slope term. The denominator is the semi-implicit Manning friction term. The code has **no advection (momentum transport) term**, which is the defining simplification of the local-inertial scheme, and no averaging of discharges from neighbouring faces. Fast, supercritical flow over steep thin sheets is therefore represented less faithfully than in a full shallow-water model.
4. Volume limiter: |Qx| is capped at min(H_i, H_j) · dx / dt, so a face cannot move more water in one step than the shallower neighbour holds.
5. Outflow limiter: for each cell, the total outflow rate through its four faces (sum of positive outgoing Q / dx) is compared with the water it holds plus the rain it receives in the step, H + dt · r_eff. If the outflow would exceed this, all its outgoing fluxes are scaled down. The code comments that without this limiter a cell with several outflows can go negative and be clamped, which creates water.
6. Continuity (mass) update:

   H(new) = max( H + dt · [ r_eff - (Qx_out - Qx_in)/dx - (Qy_out - Qy_in)/dy ], 0 )

   cells outside the domain are forced to zero.
7. Outflow boundary (when `OUTFLOW=1`, the default): on every valid cell that touches an invalid cell or the array edge, with k exposed faces, water leaves at Manning normal depth, q_out = H^(5/3) · sqrt(S0) / n per face. The depth drop is min( q_out · dt / dx · k, H ). The removed volume is added to a running outflow total. There is no inflow from outside.

### 3.4 Time step, wet/dry and run length

dt = min( α · min(dx, dy) / ( sqrt(g · H_max) + 1e-6 ), 30 s, time left ), with H_max the largest depth anywhere in the domain, floored at H_MIN. This is a gravity-wave CFL condition set by the deepest cell, so one deep river cell slows the whole domain. At 5 m with 1 m maximum depth dt is about 0.48 s; with a 10 m deep cell about 0.15 s. The run covers 86,400 s (24 h). Output is written when the simulation time passes each multiple of 3600 s, so file names are not exact hours (`depth_003604s`, `depth_000015s`), plus a final state. `max_depth.tif` is the cell-by-cell maximum over the saved files, not over all steps (chapter 5 confirms this against the data). A face carries flow only when h_f exceeds 1 mm.

### 3.5 Rain, infiltration and mass balance

Each step, cumulative rain P (m, same for every cell) advances by rate(t) · dt, where the rate is interpolated from the rain file. Runoff for the step is the difference of the cumulative SCS curve (section 2.5) before and after, divided by dt, computed once for CN 75 and once for CN 98. The effective rain for a cell is r_eff = q_75 + (q_98 - q_75) · C, where C = 1 for impervious cells and 0 otherwise. The consequence, stated plainly in `STORM_DATA_SOURCES.md`, is that infiltration is decided by the **cumulative rain**, not by the water on each cell. Water already on the surface never infiltrates later, and a cell that receives runoff from uphill gets no extra infiltration loss.

At each output the code computes: rain volume = P · cells · area; SCS runoff volume = (Q(P, 75) · pervious cells + Q(P, 98) · impervious cells) · area; stored volume = sum of H · area; and error = runoff - outflow - stored, expressed as a percentage of runoff. This is **a check of the numerical scheme** (it confirms the solver neither created nor lost water), not of the physics. Final values in the county runs lie between -1.3e-08 % and +2.2e-08 %. The first (no-infiltration, closed-edge) city runs were replaced by solver v2 and are not part of these results.

### 3.6 Runs, hardware and cost

| Item | County | City |
|---|---|---|
| Grid | 5 m, 10404 x 12043, 125,295,372 valid cells, UTM 16N | 2 m, 80,030,000 valid cells |
| Storms | 12 (6 design, 6 events) | 5 design storms (5, 10, 25, 50, 100 yr) |
| Machine | TACC Vista, partition `gh`, one NVIDIA H100 (Grace Hopper node) per storm | same |
| Slurm job | array 1114427 (`warrenv2`), all 12 tasks started 2026-11-02 21:25 to 21:27 CST, last ended 2026-11-03 08:14 | array 1102881 (`bgflood5v2`), started 2026-10-28 about 12:04 to 12:28 |
| Wall time per storm | 197.8 min (ev1) to 647.1 min (ev5) | 975.5 to 1179.6 min |
| Total node-hours | 94.1 (sum of wall minutes / 60) | 90.2 |
| Domain mean runoff | see chapter 4 | 47.2 to 120.3 mm |

![Run times per storm](docs/figures/fig5_runtimes.svg)

*Figure 5. County wall time per storm from `scripts/storms_out/vista_run_facts.json` (Slurm `sacct`, copied 2026-11-04).*

Run time grows with the amount of water, not with rain alone: ev5 (153 mm, 24 hours of rain in 24 hours) took longer than the 100-year storm (193 mm), and the quiet ev1 took 3.3 h. The most likely reason is the CFL step shrinking when deep water appears, but this was not tested. Queue waits were long: the `gh` GPU partition waited about 34 to 41 h for the last large arrays (`HANDOFF.md`). The allocation is `ATM25008` with 8,299 service units available on 2026-10-26 and about 7,633 on 2026-11-04 (`HANDOFF.md`, expiry 2027-08-31). The conversion of node-hours to units was not confirmed (Slurm bills 72 per node-hour according to the handoff, which does not fit the recorded balances), so **the SU cost of these runs is not verified**. A first no-infiltration city set (job 1092366) cost 83.98 node-hours and was superseded.

### 3.7 What the viewers add on top of the solver

The solver only writes depth. Everything else is derived by scripts, and the derivations have their own error.

**Velocity.** `scripts/flowgrid.py: face_velocity` computes a speed on each face from the solver's own depth and surface: h_f as above, slope S = (η_b - η_a)/dx, speed = h_f^(2/3) · sqrt(|S|) / n, limited to min(4 m/s, sqrt(g · h_f)), faces below 5 mm of h_f carry nothing, and the cell value is the depth-weighted mean of its faces, with a dead band for slopes under 1e-5. This is the steady balance of the same momentum equation with the inertia term dropped. It is **derived, not solver output**, and the interface says so (`FLOW_FORMAT.md`).

**Flow grids.** `build_flow_from_rasters.py` computes velocity at the solver's native resolution, averages depth over 4 x 4 cells, rotates vectors from UTM grid axes to true east and north using the grid convergence angle, and warps to a Web Mercator grid of 25 m (about 20 m on the ground). Depth is encoded in 8 bits as `(R/255)^2 x 8 m` (so a depth step is about 3 cm near 0.5 m and 6 cm near 1 m); cells under 2 cm are written as dry.

**COG tiles.** `make_cogs.py` resamples each raster bilinearly from 5 m UTM to 6.25 m Mercator (about 4.99 m on the ground), stores centimetres in uint16 (65535 = outside the model), and builds averaged overviews. The grids are of equal resolution but rotated by the grid convergence of about 0.35 degrees, so each tile pixel is a mix of up to four solver cells.

**Depth tiers.** The apps classify depth with `flowquery.py: TIER_EDGES_M = (0.1, 0.3, 0.6, 1.2)`: below 0.1 m not flooded, then shallow, moderate, deep (0.6 to 1.2 m) and severe (1.2 m and deeper). A building counts as flooded at 0.1 m or more.

![Depth tier legend](docs/figures/fig4_tiers.svg)

*Figure 4. Depth tiers and the colour-blind-safe palette.*

**Address index.** `build_address_index.py` writes one gzip JSON file per domain. For each storm it stores a flat integer array in **stride 5**: for each flooded record, five numbers `[index delta, depth in cm, tier 0 to 4, first hour flooded (0 = never), number of hours flooded]`. Decoding sums the deltas to recover the record index, and any record not listed is dry. A building's depth is the maximum over all grid pixels its footprint touches ("all-touched") and "hours flooded" counts the hourly grids at or above 0.1 m. The index was built from the **20 m flow grids** (hours 4 to 24 in the cross-check), not from the 5 m tiles. Street names are normalised to upper case with abbreviations (STREET to ST and so on); the typeahead binary-searches the street table (`ADDRESS_INDEX.md`).

**Infrastructure and roads.** `build_infrastructure.py` assigns each facility the maximum grid depth over its outline or point, per storm, and flags it as flooded when 0.1 m is reached ("flooded in 25yr" means `first_flooded == "25yr"`). Roads are sampled every 10 m as described in section 2.7.

## 4. The 12 storms

All values come from `STORM_DATA_SOURCES.md` and `scripts/storms_out/vista_run_facts.json`. "Peak" is the 5-minute rate for design storms and the wettest hour for events. "Runoff" is the domain-mean SCS runoff depth reported by the run (all cells at CN 75 except 250,641 impervious cells in the city). Wall time is the solver's own timer; Slurm elapsed times agree to the minute.

| Storm | Type and period | Rain (mm / in) | Peak (mm/h) | Runoff (mm) | Runoff / rain | Wall (min) | Slurm elapsed | Mass error (%) |
|---|---|---|---|---|---|---|---|---|
| 2yr | design, NOAA 24 h 3.5 in | 88.9 / 3.50 | 7.41 | 33.2 | 37 % | 422.9 | 07:03:06 | 2.2e-09 |
| 5yr | design, 4.2 in | 106.7 / 4.20 | 8.89 | 46.3 | 43 % | 454.8 | 07:35:03 | 1.2e-09 |
| 10yr | design, 4.9 in | 124.5 / 4.90 | 10.37 | 60.3 | 48 % | 481.7 | 08:01:51 | 1.1e-08 |
| 25yr | design, 5.9 in | 149.9 / 5.90 | 12.49 | 81.3 | 54 % | 520.3 | 08:40:33 | 7.2e-09 |
| 50yr | design, 6.7 in | 170.2 / 6.70 | 14.18 | 98.8 | 58 % | 556.2 | 09:16:22 | 9.5e-09 |
| 100yr | design, 7.6 in | 193.0 / 7.60 | 16.09 | 119.1 | 62 % | 596.2 | 09:56:24 | 6.3e-09 |
| ev1 | 2013-12-06 00:00 UTC, light all-day rain, 18 of 24 h wet | 27.2 / 1.07 | 4.06 | 1.1 | 4 % | 197.8 | 03:17:59 | 4.0e-10 |
| ev2 | 2015-03-04 00:00 UTC, steady moderate rain | 54.1 / 2.13 | 7.62 | 11.4 | 21 % | 292.9 | 04:53:06 | -1.3e-09 |
| ev3 | 2022-06-06 15:00 UTC, heavy storm (9 h wet) | 86.1 / 3.39 | 24.38 | 31.2 | 36 % | 371.2 | 06:11:22 | -1.3e-08 |
| ev4 | 2025-04-03 13:00 UTC, April 2025 flood, day 1 to 2 | 126.2 / 4.97 | 27.18 | 61.7 | 49 % | 506.9 | 08:27:08 | 2.2e-08 |
| ev5 | 2017-09-01 02:00 UTC, Harvey remnants | 153.2 / 6.03 | 29.21 | 84.1 | 55 % | 647.1 | 10:47:19 | -2.5e-09 |
| ev6 | 2010-05-01 18:00 UTC, May 2010 flood | 188.2 / 7.41 | 51.82 | 114.8 | 61 % | 596.4 | 09:56:36 | 1.6e-08 |

![Rain and runoff per storm](docs/figures/fig6_rain_runoff.svg)

*Figure 6. Rain and modelled runoff depth per storm.*

Notes on individual storms.

- **Design storms.** Return period describes the rain depth only, not the flood: antecedent wetness, infiltration and the missing upstream inflow decide how often a given flood depth really occurs. The peak intensity of all design storms is the shallow 2.4 h stretch described in section 2.1.
- **ev1 to ev3.** For ev1 the model produces 1.1 mm of runoff over the county. ev3 compares with Atlas 14 as a little below the 2-year depth (3.54 in) according to `CREDITS_AND_COMPUTE.md`, which also says it is the 16th largest 24-hour total in the gauge file since July 1996, even though the viewer data label it "heavy storm".
- **ev4.** Equals the 10-year depth (4.97 in against 4.97 in) but is a slice of a four-day flood whose airport total, according to `docs/VALIDATION_EV4.md`, was 10.5 in. Most reported closures are dated April 4 to 8, after the window ends.
- **ev5.** A 24 hour window (31 Aug 9 pm to 1 Sep 9 pm Central), not the NWS calendar-day record of 4.93 in for September 1 (`CREDITS_AND_COMPUTE.md`). Its total lies between the 25 and 50-year depths.
- **ev6.** Holds 7.41 in, close to the 100-year depth (7.45 in), with 2 interpolated hours (raw sum 7.18 in). The NWS figure for southern Warren County is just over 10 in over two days.
- **Return-period matching.** No return period has been assigned to the events in this project (**not verified**); the comparisons above are only against the Atlas 14 point depths for a 24-hour duration.
- **City runs.** City runs of the 5 to 100-year storms give domain-mean runoff of 47.2, 61.3, 82.4, 100.0 and 120.3 mm, slightly above the county values because the city has more impervious cells. City wall times: 975.5, 1027.3, 1092.0, 1134.7, 1179.6 min. No city COGs exist yet; the City view of the web viewer uses flow grids built from older depth-tier polygons, which are approximate ("Preview from depth tiers").
