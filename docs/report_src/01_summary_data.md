# Warren County Flood Model: Data, Method and Comparison

*Technical report for the BG Flood Map project (Disaster Science Operations Center and AI Research Lab, Western Kentucky University). Written 2026-11-05 from the repository files, the Vista run records copied into the repository, and web pages fetched on that date. Every number is traceable to a file named in the text. Items that could not be checked are marked **not verified**.*

## 1. Executive summary

**What the model does.** It simulates what happens to rain that falls on every square of the ground in Warren County, Kentucky. For each of 12 storms it computes, hour by hour for 24 hours, how deep the water standing or flowing over the ground is. The county run uses 5 m cells (125,295,372 cells). A separate run covers the City of Bowling Green with 2 m cells (80,030,000 cells) for five of the design storms. The method is a two-dimensional "local-inertial" shallow-water scheme of the type used in LISFLOOD-FP, written in Python with CuPy and run on one NVIDIA H100 GPU per storm at the Texas Advanced Computing Center (TACC) on the Vista system. Rain that does not soak in, according to the SCS curve-number method, becomes water on the surface and moves downhill by gravity, slowed by friction.

**The 12 storms.** Six design storms (2, 5, 10, 25, 50 and 100 year, 24 hours, depths from NOAA Atlas 14: 88.9 to 193.0 mm) and six real rain episodes replayed from the Bowling Green airport gauge (ev1 to ev6, 27.2 to 188.2 mm), including the April 2025 flood (ev4) and the May 2010 flood (ev6).

**What it is good for.**

- Showing where rainwater collects and runs off in a heavy storm, street by street and building by building, for a whole county, with a clear picture in two viewers (a web viewer and a desktop viewer).
- Comparing storms: how a 10-year rain differs from a 100-year rain, or how a real event compares with a design storm.
- A per-address look-up of modelled peak water depth and a depth tier (shallow, moderate, deep, severe).
- Teaching, planning conversations, and finding places that deserve a closer look with proper survey and engineering.
- Re-running quickly with new rain: one county storm takes 3.3 to 10.8 hours of GPU time.

**What it is not good for.**

- It is not a regulatory or insurance product. It has no FEMA acceptance and must not be used to decide a flood zone, an insurance purchase or a building permit.
- It has no river or creek inflow and no storm drains. Flooding from the Barren River or from creeks that carry water from outside the model is not represented.
- Outside the city the real terrain detail is about 10 m, so building-scale values there are indicative only.
- Its parameters (curve number 75, Manning roughness) are typical values, not calibrated.

**Validation status.** The model is **not validated**. What has been verified is the software chain: mass is conserved inside the solver (final mass-balance errors between -2.5e-08 % and +2.2e-08 % of the SCS runoff volume), the map tiles agree with the solver output to 1 cm after allowing for re-gridding, and the address index agrees with the flow grids it was built from. The one attempt to compare the model with observed flooding (April 2025, `docs/VALIDATION_EV4.md`) found that the model puts clearly more water on roads reported flooded than on typical roads (AUC 0.84, 21 observed against 100 control roads), which the validation notes call a weak sign of skill in where water collects and not a measure of accuracy. That test used 21 roads, 7 coarse point reports and 2 river gauges. It cannot validate depths.

**The claim the evidence supports.** This is a transparent, high-resolution pluvial (rainfall-driven) screening tool that complements, and does not replace, regulatory flood maps. It has not been shown to be more accurate than any of the products it is compared with in chapter 6.

**Three numbers to remember.** (1) 12 storms, 94 node-hours of county GPU time, 90 node-hours for the five city runs. (2) For the 100-year storm at hour 12 the web tiles and the older 25 m grid disagree about flooded buildings: 1,432 against 1,903. This is a resolution effect (chapter 5). (3) The model shows 2 to 3 cm of water on almost every building footprint in every storm because it rains on every cell. Only depths of 10 cm or more are treated as flooding.

## 2. Data lineage

This chapter lists every input in the order water takes through the system: rain, terrain, roughness, infiltration, buildings and parcels, roads, facilities, then derived layers. For each one: what it is, who publishes it, the exact version where the repository records it, resolution, file name and size, processing, and limitations. Where the repository does not record a detail (a download date, a tile name, a licence) the text says so.

![Data flow from raw inputs to viewers](docs/figures/fig1_dataflow.svg)

*Figure 1. Data flow. Yellow boxes are model inputs prepared on Vista; the tan box is the solver. Source: `scripts/*.py`, `scripts/*.sh`, `COUNTY_RUN.md`.*

### 2.1 Rain: design storms

| Item | Value |
|---|---|
| What | 24-hour rainfall depth for 2, 5, 10, 25, 50 and 100-year return periods, spread over 24 h |
| Publisher | NOAA National Weather Service, Hydrometeorological Design Studies Center: Precipitation Frequency Data Server (PFDS), Atlas 14 Volume 2 (Ohio River Basin and surrounding states) |
| Point used for the cross-check | 36.9685 N, 86.4808 W (downtown Bowling Green), partial duration series, depths in inches |
| Retrieved | 2026-10-28 (README.md); the original depths used in the runs were typed into `scripts/run_5_storms.sh` and `run_county_all.sh` earlier, with no recorded retrieval date |
| Depths used | 3.5, 4.2, 4.9, 5.9, 6.7, 7.6 in (88.9, 106.7, 124.5, 149.9, 170.2, 193.0 mm) |
| NOAA values at the point | 3.54, 4.32, 4.97, 5.89, 6.65, 7.45 in |
| Difference | -1.1 %, -2.8 %, -1.4 %, +0.2 %, +0.8 %, +2.0 % (README.md table) |
| Licence | NOAA PFDS data are US Government products; the licence terms were not looked up here (**not verified**) |
| Files | `runs/storm_<name>.rain` on Vista; the generator `make_rain.py` is embedded in `scripts/run_5_storms.sh`; the county version `make_rain_county.py` is written by `scripts/run_county_all.sh` (neither exists as a separate file in the repository) |

Two corrections to common assumptions. First, the depths in the runs are rounded values, up to 2.8 % lower or 2.0 % higher than the NOAA point estimates. Second, the label "Atlas 14 Volume 2 Version 3" is used in the repository notes. The NWS page fetched on 2026-11-05 (https://www.weather.gov/owp/hdsc_currentpf) lists Volume 2 as "Ohio River Basin and Surrounding States", 2004, revised 2006. The "Version 3" label is **not verified** against NOAA.

**Temporal distribution.** All six design storms share one 11-point cumulative curve (fraction of 24 h against fraction of the depth): 0%→0%, 10%→4%, 20%→10%, 30%→19%, 40%→31%, 50%→50%, 60%→70%, 70%→79%, 80%→86%, 90%→92%, 100%→100%. It is linear in time between points, and each 5-minute step gets the rain of that step as a constant rate (577 points per file). This is not the NRCS Type II table. The curve is the one the first 2-year run actually used. A check in `run_5_storms.sh` compared the domain-mean water depth of that run at hours 10 and 11 (30.3 and 37.2 mm) with the curve's prediction (30.4 and 37.4 mm); the official Type II table would have given 16.1 and 20.9 mm. The consequence is that the busiest part of every design storm is the stretch between hours 12 and 14.4 at 0.2 times the depth per 2.4 h, which is a smooth, long burst with no short intense peak. For the 100-year storm the peak 5-minute rate is 16.09 mm/h. A real design storm of this depth normally has a much more intense short burst. Peak flows in small catchments would be higher with a short burst.

**Rain file format.** Line 1 holds the number of points. Each following line is `seconds rate`, with the rate in metres per second. The solver reads the values as m/s. A mistaken file in mm/s once made three storms 1000 times too intense; the solver now aborts if the peak rate exceeds 600 mm/h. The integrated totals of the files agree with the target depths within 0.01 mm (`STORM_DATA_SOURCES.md`).

![Hyetographs of a design storm and a real event](docs/figures/fig3_hyetographs.svg)

*Figure 3. Hourly rain of the 100-year design storm (from the 11-point curve) and of event ev4 (computed from `scripts/storms_out/storm_ev4.rain`). Total of the left panel 193.04 mm; right panel 126.24 mm.*

### 2.2 Rain: historical events

| Item | Value |
|---|---|
| What | Hourly precipitation at station BWG (Bowling Green-Warren County Regional Airport, call sign KBWG), field `p01i` |
| Publisher | Iowa Environmental Mesonet (Iowa State University) ASOS archive, redistributing NOAA/FAA observations |
| Downloader | `scripts/build_historical_storms.py` |
| Cached copy | `scripts/storms_out/bwg_hourly.csv`, 257,703 lines (header plus 257,702 rows), columns `station,valid,p01i`, rows start 1996-01-01 00:00 |
| Download date | not recorded in the repository (`CREDITS_AND_COMPUTE.md` says the events were checked on 2026-10-28) |
| Licence | IEM states its data are free to use; the exact wording was not fetched (**not verified**) |
| Rain files | `scripts/storms_out/storm_ev1.rain` to `storm_ev6.rain`, 49 points each (each hour as t and t+3599 s), rates in m/s |

Processing rules (`STORM_DATA_SOURCES.md`): 253,422 of the 257,702 gauge rows are time-stamped at :53. Each report is assigned to the clock hour of its stamp; when several reports fall in one hour the largest is kept. The rain can therefore be shifted by up to about an hour against real time. Gaps up to 3 h that are flanked by data are interpolated. Only ev6 has gaps (2 of its 24 hours). The meaning of `p01i` as defined by IEM was not re-checked.

Six 24-hour windows were chosen: ev1 starts 2013-12-06 00:00 UTC (light all-day rain), ev2 2015-03-04 00:00 UTC (steady moderate rain), ev3 2022-06-06 15:00 UTC (heavy storm), ev4 2025-04-03 13:00 UTC (first day and a half of the April 2 to 6, 2025 flood), ev5 2017-09-01 02:00 UTC (remnants of Hurricane Harvey), ev6 2010-05-01 18:00 UTC (May 2010 flood). The descriptions of each event in the viewer are quoted from the repository script and from NWS Louisville pages cited in `CREDITS_AND_COMPUTE.md` (weather.gov/lmk); they were not re-checked against NWS records here. Chapter 4 lists totals and peaks.

Limits that matter for interpretation: one gauge only, so rain is the same over the whole county; the 24-hour window cuts off multi-day floods (ev4 contains 4.97 in of a 10.5 in four-day airport total according to `docs/VALIDATION_EV4.md`; ev6 holds 7.41 in against about 10 in over two days in southern Warren County according to NWS as quoted in the repository).

### 2.3 Terrain

**County terrain.** `dem_warren_5m.tif`, 514,040,145 bytes in the app folder (`layers/`), 490 MB. Built by `scripts/build_county_grid.sh` on Vista:

1. Download two USGS 3DEP 1/3 arc-second tiles, `USGS_13_n37w087.tif` and `USGS_13_n38w087.tif`, from the USGS "The National Map" staged products bucket (`prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/`). The script takes the "current" version at the time of download. The download date and the tile versions are **not recorded**. USGS 3DEP data are public domain per USGS policy (**not verified** here by fetch).
2. Warp to EPSG:32616 (UTM zone 16N) at 5 m with bilinear resampling, clipped to the box W -86.72, S 36.65, E -86.14, N 37.19 (Warren County plus a margin).
3. The 2 m city DEM `dem_bg_2m.tif` is added as the last input so it overrides 3DEP where it exists.
4. A sanity check stops the build if the city DEM and 3DEP differ by more than 3 m in median (catches feet-versus-metres mistakes).

Resulting grid: 10404 x 12043 cells, 125,295,372 valid cells, about 52 x 60 km. 3DEP 1/3 arc-second pixels measure about 10.3 m north-south and 8.2 m east-west (pixel 9.26e-5 degrees). Checks done on Vista (`STORM_DATA_SOURCES.md`, `docs/COG_VERIFICATION.md`): at 400 random points outside the city the 5 m grid differs from the 3DEP tile by 0.21 m on average (95th percentile 0.77 m), and inside the city it differs from the 2 m DEM by 0.06 m on average. **The real terrain information outside the city is therefore about 10 m, resampled to 5 m. The 5 m cell size does not mean 5 m terrain detail.** Small channels, roadside ditches and embankments are not resolved there.

**City terrain.** `dem_bg_2m.tif` on Vista, 2 m, metres, EPSG:32616, 10000 x 10000 cells (20 x 20 km, about 13 % of the county grid). The repository README says it was made from 1 m lidar tiles merged and resampled to 2 m with GDAL. The user-supplied tile names (USGS 1 m tiles of the form `USGS_1M_16_x54y409...`, KyFromAbove Western Kentucky A22) do **not** appear anywhere in the repository, so the exact tile list, the lidar collection and its acquisition date are **not verified**. `CREDITS_AND_COMPUTE.md` marks the same question "[confirm]". An older local file `dem_bg.tif` (58,728,572 bytes, 5.088 m cells, elevations in feet) comes from the KyFromAbove elevation image service (`kyraster.ky.gov`, services tried in order: Ky_DEM_KYAPED_2FT_Phase3, Phase2, 5FT_WGS84WM; see `step3_data_integration.py`). It feeds the older web layers, sinkhole analysis and the viewer's terrain; the solver does not use it.

**Vertical accuracy.** The legacy methodology text quotes a lidar vertical accuracy of ±10 cm RMSE for a "5 m" KyFromAbove DEM. That text predates the solver runs, describes the old DEM, and the figure was not checked. It is not used here.

### 2.4 Roughness (Manning's n)

| Raster | Value | Notes |
|---|---|---|
| `mannings_n.tif` (city, 2 m, on Vista) | open land 0.035, buildings 0.08, sinkholes 0.12, roads 0.016 | per `CREDITS_AND_COMPUTE.md` ("assumed typical values, not calibrated") and `scripts/build_flow_from_tiers.py` (constants `N_OPEN, N_BUILDINGS, N_ROADS = 0.035, 0.08, 0.016`) |
| `mannings_n_county.tif` (5 m) | 0.035 everywhere, city raster on top where it exists | `scripts/build_county_grid.sh` (gdal_calc: `(A>-9000)*0.035`, then the city raster is warped on top with nearest-neighbour) |

**What is not verified:** the script that produced the city `mannings_n.tif` is not in the repository (it was prepared on Vista before the repository's records begin). The land-cover source of the classes (building and road footprints are presumably OSM, sinkholes presumably the repository's sinkhole layer derived from the lidar) is therefore **not verified**. NLCD land cover was **not used**; `COUNTY_RUN.md` suggests it as an improvement. The solver derives its impervious class from this raster by value: cells with n below 0.02 (roads) or between 0.07 and 0.09 (buildings) are treated as impervious. In the county runs this identifies 250,641 cells (6.3 km²) and they are all inside the city. Roads and buildings outside the city are treated as pervious farmland (CN 75).

### 2.5 Infiltration

SCS (NRCS) curve-number method, USDA NRCS National Engineering Handbook Part 630 Chapter 10 and TR-55 (cited in `CREDITS_AND_COMPUTE.md`; the NRCS documents were not fetched here: **not verified** by fetch). The code (`scs_runoff_m` in `scripts/flood_solver_v2.py`) uses, for cumulative rain P and curve number CN:

- S = (1000 / CN - 10) x 0.0254 (metres)
- Ia = 0.2 S (initial abstraction)
- Q = 0 if P ≤ Ia, else Q = (P - Ia)² / (P - Ia + S)

Values: CN 75 for pervious land (antecedent moisture condition II), CN 98 for roads and buildings. These are typical textbook values chosen without soil or land-use data. Karst terrain around Bowling Green, where water can drain into sinkholes and cave conduits, is not modelled. The value S for CN 75 is 0.0847 m, so Ia = 16.9 mm; for CN 98 S = 0.0052 m and Ia = 1.0 mm. A consequence worth noting: the 100-year runoff depth over the county is 119.1 mm of 193.0 mm of rain, i.e. 62 %.

### 2.6 Buildings, parcels and addresses

| Dataset | Source | Details |
|---|---|---|
| OSM extract | Geofabrik, `kentucky-latest.osm.pbf`, https://download.geofabrik.de/north-america/us/kentucky.html | 154,005,294 bytes, file dated 2026-10-01 03:19 on the local disk; the extract's internal timestamp was not read (**not verified**). OpenStreetMap data are licensed ODbL 1.0; this is stated by OSM generally and was not fetched here. |
| City buildings | OSM building ways, `layers/buildings.geojson` (12,405,697 bytes) | 20,284 structures (README, DESIGN.md); property values and addresses joined from parcels |
| County buildings | `layers/county/buildings.geojson.gz` (1,618,275 bytes) | 24,620 building ways from `scripts/build_county_layers.py` via the desktop reader; multipolygon relations are not read. The number is only 21 % above the city's 20,284 although the county is far larger. Why is **not verified**; it suggests the reader's coverage of rural buildings is incomplete. |
| Microsoft building footprints | not used | |
| City parcels | `layers/all_parcels.geojson` (27,528,814 bytes), 40,987 parcels | Warren County Property Valuation Administrator parcels, via the City of Bowling Green GIS service `https://webgis.bgky.org/server/rest/services/WARCO/Parcel_Reference/MapServer/0` (`step3_data_integration.py`); extent only the city area (lng -86.54..-86.29, lat 36.89..37.06), not the county. Download date: not recorded (files dated 2026-10-01). |
| Address index | `layers/address_index.json.gz` (436,121 bytes) city 30,148 records (11,329 buildings + 18,819 lots); `layers/county/address_index.json.gz` (534,157 bytes) 29,754 records | built by `scripts/build_address_index.py`; format in `ADDRESS_INDEX.md` |

### 2.7 Roads and critical facilities

Roads: OSM driveable highway ways from the same Geofabrik file. City `layers/roads.geojson` (797,322 bytes, 516 multi-line features); county `layers/county/roads.geojson.gz` (1,161,246 bytes, 18,349 ways, simplified with a 1.5 m tolerance). Flooded-road tables in `layers/roads_flooded/<domain>/<storm>.json` are built by `scripts/build_infrastructure.py` by sampling each way every 10 m (midpoints) on the flow grid; a sample is wet at depth ≥ 0.02 m and flooded at ≥ 0.1 m.

Critical facilities: OSM points and outlines with tags for hospital, clinic, fire station, police, school, shelter, assembly point, community centre, town hall and EMS (`scripts/build_infrastructure.py`). Duplicates of one named facility within 200 m are merged. Output: city `layers/infrastructure.geojson` (17,775 bytes, 51 facilities) and county `layers/county/infrastructure.geojson` (40,677 bytes, 98 facilities). HIFLD (Homeland Infrastructure Foundation-Level Data) merging is written (`scripts/fetch_hifld.py`, URLs untested) but **has not been run**. Each facility carries a maximum depth per storm taken from the 25 m flow grid, which is known to average ponds away (chapter 5).

### 2.8 Other inputs mentioned in the repository

- **FEMA NFHL flood zones** (`layers/fema.geojson`, 1,802,470 bytes) fetched from `https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28` by `fetch_fema.py` for a city bounding box. Used for display and for an older map-comparison script; not an input to the solver.
- **Census (ACS block groups, TIGER) and CDC SVI**: scripts `fetch_census.py` and `fetch_svi.py` exist; they were **not run and no Census data has been used**. The planned demographic exposure report was stopped before delivery (`HANDOFF.md`).
- **Legacy DEM-threshold model** (`flood_methodology.txt`, `flood_sim.geojson`, `min_rain.tif`): an earlier approach that mapped terrain percentiles to rainfall thresholds. It is not the solver model and none of its numbers appear in this report.

### 2.9 Derived products and their sizes

| Product | Location | Resolution and format | Size |
|---|---|---|---|
| Solver rasters, county | Vista `runs/county_v2/<storm>/` | 5.0 m, EPSG:32616, float32 m, LZW, 10404 x 12043, 25 hourly files + `max_depth.tif` + `run_summary.json` | about 94 GB for all storms plus the city runs (`HANDOFF.md`) |
| COG tiles | `layers/cog/county/<storm>/` | 6.25 m (about 4.99 m on the ground) EPSG:3857, uint16 centimetres, 65535 = no data, 512 px tiles, DEFLATE predictor 2, overviews 2 to 32 | 324 files, 5.48 GB |
| Flow grids | `layers/flow_county/<storm>/` | 25 m Mercator cell (about 20 m on the ground), 2631 x 3052 cells, 8-bit RGBA hourly grids | tens of MB per storm |
| Storm info | `layers/storm_info.json` | text for the "About this storm" panel | 83,960 bytes |

COG creation: `scripts/make_cogs.py` through `scripts/vista_make_cogs.sh` (Slurm job 1123323 with GDAL warp tolerance 0.01 pixel; the earlier set from job 1122676 with the default 0.125 pixel tolerance had errors up to 15 cm and was replaced). Flow grids: `scripts/build_flow_from_rasters.py` with `scripts/flowgrid.py`: depth is block-averaged 4 x 4 from 5 m to 20 m, velocity is derived (section 3.7), depth is stored in 8 bits as `depth = (R/255)^2 x 8 m`, so resolution is coarse for deep water.
