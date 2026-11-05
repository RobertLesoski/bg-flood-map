## 5. Verification and accuracy

The word "verified" in this project has two different meanings, and mixing them is the main risk when presenting. **Verification** asks whether a product is a faithful copy of the one before it in the chain (solver rasters to tiles to index). **Validation** asks whether the model agrees with what really happened. Section 5.1 to 5.4 are verification. Section 5.5 is the one validation attempt.

### 5.1 Solver internal checks

- **Mass balance.** Every run logs a balance at each output. County final errors range from -1.3e-08 % to +2.2e-08 % of the SCS runoff volume; city errors from -3.1e-09 % to -6.9e-09 % (`scripts/storms_out/vista_run_facts.json`). It confirms conservation of water inside the scheme.
- **Analytic checks.** `CREDITS_AND_COMPUTE.md` lists tests against known answers: rain volume equals the storm depth, SCS runoff matches the textbook formula, water balance closes, rain on an inclined plane matches Manning's equation within 0.3 %, still water stays flat. The test scripts are not part of the repository tree read for this report, so these results are quoted, not re-run (**not re-verified here**).
- **Rain file checks.** The rain files integrate to the target depth within 0.01 mm; the rain file for each event was recomputed from `bwg_hourly.csv` and agrees (`STORM_DATA_SOURCES.md`).
- **Terrain check.** The county DEM build compares the city DEM and 3DEP and aborts if the median differs by more than 3 m.

### 5.2 Tiles against solver rasters

From `docs/COG_VERIFICATION.md` (checks run on 2026-11-05 on Vista and locally; 22,421 random points over all 12 storms and 25 hourly files plus the maximum file):

| Test | Result |
|---|---|
| Structure of 12 x 26 rasters | all pass: EPSG:3857, uint16 cm, nodata 65535, 512 px tiles, overviews 2 to 32, 10475 x 12156 px |
| Local tiles identical to Vista set `county_cog_tol` | sha256, 324 of 324 files; 0 matches with the earlier `county_cog` |
| Tile value against the exact expected bilinear resample of the four surrounding solver cells | maximum error 1 cm, 99th percentile 1 cm (0 cm for ev1), mean 0.004 to 0.023 cm per storm |
| The earlier set `county_cog` (default warp tolerance) | maximum 15 cm, 99th percentile up to 6 cm |
| Tile value inside the range of its four source cells (property of any bilinear resample) | 0 violations of 18,806 values |
| Tile against the single solver cell under the point | mean absolute 1.0 to 2.2 cm per storm; 99th percentile 29 to 51 cm; largest single difference 132 cm |
| `max_depth.tif` against the maximum of the 25 hourly rasters (300 points per storm) | equal to the cm: confirms max depth is the maximum of saved snapshots |
| Web viewer reader against rasterio | 516 of 516 and 480 of 480 points identical |

Interpretation: the tiles faithfully reproduce the solver output as a bilinear resample. The difference against the single cell under a point comes from re-gridding, not from bias. The deep stratum (at least 1.2 m) shows a mean absolute difference of 7.3 cm (hourly files) and a signed mean near zero (-0.23 cm). The tile is lower than the solver cell in 35 % of deep points and higher in 31 %. Roughly: a tile pixel says what the model has "here, within the local roughness", and cell-scale features of one solver cell can be blurred (one documented example: a solver cell of 171 cm reads 82 cm in the tile). The solver rasters are 5.0 m UTM; the tiles are 4.99 m on the ground in Web Mercator; no information is lost by coarsening and none is invented.

### 5.3 Address index and flow grids against tiles

Sampled 300 random buildings per storm (half chosen among those the index lists as wet, so these are not county-wide rates) and all facilities (`docs/COG_VERIFICATION.md` section 7):

- Median difference between the tile footprint maximum and the index depth is 1 to 4 cm. The index and the tile disagree on the 10 cm flood flag for 12 to 18 % of sampled buildings, in both directions. The 20 m grid averages a pit beside a house into the house's cells, and the 5 m tile sees only the footprint pixels. For rare ev1 the tiles flag more buildings (41 of 300 are at or above 10 cm in the tile but not in the index).
- Facilities: the tile maximum minus index maximum has median 0 to 1 cm but a mean of +15 to +18 cm, because a few campus outlines contain a pond or ditch whose 5 m cells reach 2 m while the 20 m grid averages it down. The tiles find 33 to 34 of 98 facilities at 10 cm or more in design storms, where the older index finds 26 to 29. The facility file has **not** been rebuilt from the tiles.
- Flow grids at hours 12 and 24: the tile mean over a grid cell minus the grid depth has signed mean between -0.4 and +0.7 cm, mean absolute 0.7 to 3.5 cm, and 99th percentile 4.9 to 53 cm.

### 5.4 The tile versus grid difference in totals

At hour 12 of the 100-year storm, county-wide totals computed from the two products differ (`HANDOFF.md`):

| Quantity | Tiles (5 m) | Old flow grid (20 m) |
|---|---|---|
| Flooded buildings (at or above 0.1 m) | 1,432 | 1,903 |
| Wet road length | 185 km | 273 km |
| Wet area | 375 km² | 479 km² |
| Water volume | agrees to about 2 % | |

Water volume agrees, so the two describe the same water. The grid reports more flooded buildings, more wet road and more wet area, consistent with the 20 m averaging of thin sheet water into neighbouring cells and the 8-bit depth code with a 2 cm dry cutoff. The tiles resolve the same water into thinner, shallower films and deeper pits. Neither is correct in an absolute sense, because neither has been compared with observations. The desktop viewer's headline still uses the grid ("full precision" label); the handoff recommends headlining the tiles once `WEB_COG_PROGRESS.md` carries `Status: verified`. **Which count to quote in a presentation:** state both, say they are one run at two resolutions, and give the 0.1 m threshold.

### 5.5 Comparison with observed flooding (ev4, April 2025)

Source: `docs/VALIDATION_EV4.md` (analysis dated 2026-11-05, reproducible with `scripts/validate_ev4.py`). The model's ev4 uses 24 hours of rain from 2025-04-03 13:00 UTC (4.97 in); the real flood ran April 2 to 6 with 10.5 in at the same airport gauge and 13.1 in at WKU, with most closures dated April 4 to 8.

| Test | Result |
|---|---|
| Roads reported flooded or closed, scorable by exact name (21 of 41) | model max depth 0.62 m median vs 0.08 m for 100 randomly picked control roads; AUC 0.84 |
| Flag at 0.10 m, any sample on the road | 21 of 21 observed (100 %) vs 42 of 100 control: almost no discrimination |
| Flag at 0.30 m, any sample | 16 of 21 (76 %) vs 27 of 100 (27 %) |
| Flag at 0.30 m with at least 20 % of the length wet | 2 of 21 (10 %) vs 8 of 100 (8 %): no difference |
| 7 NWS storm reports (points) | model depth at the point about 0 in all seven; maximum within 100 m is above 0.1 m for 79 % of random points, so this test has no power |
| Gauges | Barren River peak (35.76 ft, 33,100 cfs, 2025-04-04 22:30 UTC) came about 9.5 h after the model window; Drakes Creek peaked before the heaviest model rain; the model has no river output so only qualitative |
| 196 damaged structures (City, April 10) | not comparable: damage zones, unmapped addresses, river water |

The validation note's own conclusion applies: the model is **not validated**; the road result is a weak sign of skill in where water collects. The control roads are of unknown truth (a road not named in the news is unknown, not dry). The model's 2 to 3 cm film everywhere and 5 m single deep cells make the 0.1 m test nearly free. Expected misses are roads flooded by the Barren River or creeks (Boatlanding Road, Three Springs Road), which the model cannot produce. A proper test needs the full April 2 to 6 rain, upstream inflow, observed depths such as high-water marks, and a list of dry roads.

### 5.6 Test suites

`python -m pytest scripts/tests -q -p no:cacheprovider`: 267 tests passed in the final recorded run (`WEB_COG_PROGRESS.md`), with two timing-sensitive classes (server start-up, one 4000 x 4000 flow-grid timing test) that fail when the machine is busy and pass alone. The desktop suite has about 400 tests; the last full run was not confirmed green after the final edits (`HANDOFF.md`). The web viewer's layer harness has 6 known failures (pitch-60 terrain alignment, zoom-18 density) not diagnosed. The service worker, a real phone and the swipe divider were **not verified**. None of these tests validate the flood physics against observations.

## 6. Comparison with publicly available flood models and products

**Rules for this chapter.** A statement about another product appears only if the page was fetched on 2026-11-05 and the URL is given. Many publishers' pages refused automated access (HTTP 403 or 404). Where that happened the item is marked **not verified**, and the chapter says what is missing instead of filling it from memory. The comparison is of purpose and method, not of accuracy. This project has no accuracy measure to set beside anyone else's.

### 6.1 Products, one by one

**FEMA National Flood Hazard Layer and Flood Insurance Rate Maps.** The FEMA pages for the NFHL, its glossary and the National Risk Index returned HTTP 403 (https://www.fema.gov/flood-maps/national-flood-hazard-layer, fetched 2026-11-05), so FEMA's own wording, the effective date of the Warren County and Bowling Green maps, and the existence of a "Future Risk Index" are **not verified**. A web search (not fetched) suggested that the Bowling Green floodplain ordinance adopts a Warren County Flood Insurance Study dated 2 May 2007 (the ordinance page returned 403 on fetch), so the effective date is **not verified**. A secondary source (Wikipedia, National Flood Insurance Program, fetched 2026-11-05, https://en.wikipedia.org/wiki/National_Flood_Insurance_Program) describes the maps as covering riverine and coastal base floods, does not address pluvial or local drainage flooding, and notes that maps can go out of date after development. This project does hold the NFHL polygons for a city bounding box (`layers/fema.geojson`, fetched from FEMA's ArcGIS service) but has **not** computed a quantitative overlap with the model.

**FEMA National Risk Index.** Page returned 301 then 403 (https://hazards.fema.gov/nri/, fetched 2026-11-05). Not verified.

**First Street (Flood Factor).** The methodology page is behind a terms-acceptance gate (https://firststreet.org/methodology/flood, fetched 2026-11-05). Resolution, pluvial coverage and validation are **not verified** from a first-party source. A web search (not fetched) reported that the model includes rainfall flooding and may overstate risk in small municipalities by missing local protections such as pumps and basins. That is hearsay and is not relied on.

**NOAA National Water Model (NWM) and NWS inundation mapping.** The NOAA page (https://water.noaa.gov/about/nwm, fetched 2026-11-05) states: land-surface grid 1 km; ponded water and soil saturation grids 100 and 250 m; channel and overland routing 250 m; more than 2.7 million CONUS river reaches; an 18-hour short-range deterministic forecast, a 10-day medium range and a 30-day ensemble; it complements river forecasts at about 4,000 points; fast, short-distance rainfall changes are hard to capture. It is a forecast system for rivers, not a building-scale hazard map. The NWPS gauge page for the Barren River at Bowling Green (BWGK2, https://water.noaa.gov/gauges/BWGK2, fetched 2026-11-05) carries only a generic line that inundation services cover almost all of the US. Gauge-specific inundation mapping for BWGK2 is **not verified**.

**USGS Flood Inundation Mapper.** The site refused access (https://fim.wim.usgs.gov/fim/, 403). Whether a map library exists for the Barren River at Bowling Green or Drakes Creek is **not verified**. A web search returned other Kentucky studies and none for these streams, but absence from a search is not proof. Note that the USGS gauges 03314500 (Barren River at Bowling Green) and 03314000 (Drakes Creek near Alvaton) exist and were used for timing in the ev4 analysis.

**HEC-RAS 2D (US Army Corps of Engineers).** The user manual page (https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/latest/running-a-model-with-2d-flow-areas/shallow-water-or-diffusion-wave-equations, fetched 2026-11-05) lists three equation sets: diffusion wave (default), and two full-momentum shallow-water options. It recommends building with the diffusion wave and then running the full set; if the two differ markedly the full-momentum results are the more reliable basis for calibration and event simulation, and are recommended for flash floods, very flat rivers, structures and mixed flow regimes. The product page (https://www.hec.usace.army.mil/software/hec-ras/) lists 1D and 2D unsteady flow, a stormwater pipe network and sediment. The rain-on-grid page returned 404, so the rain-on-grid capability is **not verified from HEC**. HEC-RAS includes pipes and calibrated engineering practice; this project's solver has neither.

**LISFLOOD-FP (University of Bristol).** The version 8 page (https://www.seamlesswave.com/LISFLOOD8.0.html, fetched 2026-11-05) states the code is released under GPL v3 for non-commercial use (Zenodo 10.5281/zenodo.4073011), offers ACC (local-inertial), FV1 and DG2 solvers, added rain-on-grid in v8.0, and says the ACC solver suits fluvial and catchment-scale pluvial flooding but is not recommended for supercritical thin flows at fine resolution. That last point applies directly to this project: a 5 m rain-on-grid model produces thin, fast sheets on slopes. The Bates et al. (2010) paper was not fetched (403).

**Fathom.** The product page (https://www.fathom.global/product/flood-hazard-data/global-flood-map/, fetched 2026-11-05) states the global map has about 30 m resolution downscaled to 10 m where data allow, covers fluvial, pluvial and coastal flooding, offers return periods from 5 to 1,000 years (the US map is about 10 m, inland flooding, 10 return periods), is delivered by API, portal or GeoTIFF under licence (price not stated), and reports its own critical success index of about 0.75 and a mean depth deviation of about 0.6 m. These are the publisher's claims and not independent. A comparison with FEMA is **not verified**.

**Google Flood Hub.** The page (https://sites.research.google/floods/, fetched 2026-11-05) gave only a disclaimer that its information is approximate. Forecast type, US coverage and resolution are **not verified**.

**Kentucky resources.** KyFromAbove (https://kyfromabove.ky.gov/, fetched 2026-11-05) is "Kentucky's Aerial Photography & Elevation Data Program" and is the likely source of the 1 m lidar behind the city DEM. The Kentucky flood-map portal name and the Division of Water floodplain pages were **not verified** (404). This report therefore makes no statement about Kentucky's own flood hazard portal.

**NOAA Atlas 14.** The PFDS (https://hdsc.nws.noaa.gov/pfds/, fetched 2026-11-05) provides point precipitation frequency estimates with confidence intervals, grids and links to temporal distributions. The model uses only the 24-hour depths and not the confidence interval, which would give a range of depths per return period.

### 6.2 Comparison table

The "this project" column is facts from the repository. Entries for other products carry only what is cited above; an empty cell means not verified.

| Aspect | This project | FEMA NFHL / FIRM | NWS National Water Model | Fathom (US/global) | HEC-RAS 2D study | LISFLOOD-FP |
|---|---|---|---|---|---|---|
| Method | 2D local-inertial, rain-on-grid, SCS CN | Engineering studies (not verified here) | Land surface and routing model | Hydrodynamic model (publisher) | 2D diffusion wave or full momentum | Local-inertial (ACC), FV1, DG2 |
| Resolution | 5 m county (10 m real terrain), 2 m city | | 1 km land, 250 m routing | 30 m global, 10 m US | study dependent | user choice |
| Pluvial / rain-on-grid | yes | not verified | partly (ponded water 100/250 m) | yes (global) | not verified at HEC | yes since v8.0 |
| River flooding | no inflow, no channel | yes (riverine) | yes (reaches) | yes | yes | yes |
| Storm drains | no | | no | | pipe network option | no |
| Calibration / validation | none; one weak check (AUC 0.84) | | not stated on the page | publisher claims CSI 0.75 | calibration advised | user dependent |
| Open data and code | yes (code and lineage in repo; OSM, 3DEP inputs public) | data free (not verified) | data via NOAA (not verified) | no, licensed | software free (not verified) | GPL v3 non-commercial |
| Update frequency | when rerun (3 to 11 h per county storm) | | forecast cycles (18 h, 10 d, 30 d) | | | |
| Access | local viewers, repository | | public site | API / portal / GeoTIFF | desktop software | download |
| Cost | none (allocation `ATM25008`) | | | licence, price not stated | | free non-commercial |
| Uncertainty reported | depth tiers, 0.1 m flood flag, honesty labels | | | CSI and depth deviation claimed | | |
| Regulatory use | none | the regulatory standard for NFIP (not verified here) | no | not verified | yes when accepted | no |

![Capability matrix](docs/figures/fig7_matrix.svg)

*Figure 7. Capability matrix drawn from the table. Cells reading "not verified" are shown tan.*

### 6.3 Where this project is stronger

- **Scope in rain-driven flooding.** It treats rain falling on every cell of the county, which riverine regulatory maps do not (by the secondary source above). The result shows ponding in depressions and runoff along streets that a river-only map cannot.
- **Scenario breadth.** Twelve scenarios in one framework, including six real rain events (April 2025, May 2010, Harvey remnants).
- **Resolution in the city.** 2 m cells and a 2 m terrain model give building-scale detail in Bowling Green. County grids are 5 m (but see 10 m terrain, below).
- **Transparency and reproducibility.** Every input has a script, a source URL and a record; the run summaries, Slurm times and verification numbers are in the repository. A reader can see exactly what the model assumes. Commercial products state their method in general terms.
- **Rerun speed.** A county storm takes 3.3 to 10.8 hours on one GPU. A new rain file can be run overnight.
- **Per-address results with honest labels.** Depth, tier, first hour and hours flooded per address, with "derived", "not validated" and 0.1 m threshold labels in the apps.
- **Viewers.** Interactive 3D web and desktop viewers, usable offline in part, with a documented data format.

### 6.4 Where it is weaker or not comparable

- **Not validated.** One weak road-level test (section 5.5). Regulatory and engineering studies are calibrated against gauges and high-water marks. This model uses CN 75 and n 0.035 without calibration.
- **No river or creek flooding.** No inflow from upstream; the Barren River arriving from other counties is absent. In the April 2025 flood much damage was river and creek water. A river-based product covers what this one cannot.
- **No storm drains or sinkhole drainage.** All rain that does not infiltrate stays on the surface and runs to low spots, so ponding may be overestimated in places that drain.
- **Uniform curve number.** CN 75 outside the city and CN 98 only inside it; no soil or land cover.
- **Terrain.** 10 m real terrain outside the city. Small streams and ditches are not resolved.
- **Not regulatory or insurance grade, no FEMA acceptance.** It cannot change a flood zone and must not be shown as an official map.
- **Design storm shape.** The shared smooth curve is not Type II and has no short intense burst.
- **Single gauge.** Rain is the same everywhere; real storms are patchy.
- **Fidelity of other products is not assessed.** This report does not claim that any listed product is more or less accurate than this one.

### 6.5 The claim the evidence supports

> This is a transparent, high-resolution pluvial screening tool that complements, and does not replace, regulatory maps. It has not been shown to be more accurate than any of them.

"Better" can honestly mean "more transparent, faster to rerun with new rain, higher resolution inside the city, and covers rain-driven flooding that river-based maps omit". It cannot honestly mean "more accurate". To claim accuracy the project needs the work listed in chapter 7 (calibration against gauges and high-water marks, a river inflow, drains, and a validation with the full multi-day rain).

## 7. Limitations and risks

### 7.1 Numerical and modelling limits

1. **No advection term.** The scheme drops the momentum-transport term (section 3.3). Fast, steep, supercritical flow is represented less faithfully; LISFLOOD-FP's own documentation advises against the local-inertial solver for supercritical thin flows at fine resolution (section 6.1).
2. **Rain-on-grid ponding.** Rain falls on all cells at once. Sheet flow of 2 to 3 cm is on almost every building footprint: in a sample of 144 buildings the median footprint maximum was 3 cm in the 2-year storm and also in the smallest event (ev1) (`STORM_DATA_SOURCES.md`). This is a feature of the method, not a measurement, and only depths of 10 cm or more are called flooding. The 10 cm threshold is itself a judgement.
3. **Ponded water never infiltrates later.** Infiltration depends on cumulative rain, not on the water on each cell. Water on porous ground stays on it.
4. **Hourly snapshots only.** Rasters are saved hourly. `max_depth` is the maximum of those snapshots, so a peak between two saves is not captured; this has been confirmed against the data. Flood duration is counted in hours, not minutes.
5. **Resampling.** Tiles are 6.25 m Mercator from 5 m UTM; grids are 20 m block averages. Depth values differ between products (section 5.4); 8-bit depth coding in the grids; a 2 cm dry cutoff.
6. **10 m terrain.** Outside the city the effective resolution is about 10 m, the mean difference from 3DEP is 0.21 m and the 95th percentile 0.77 m (400 points). Terrain error is comparable to or larger than the depths that matter for buildings.
7. **Building footprints.** OSM footprints are used as given, with no floor-level data. A building footprint depth is the maximum over touched pixels, which is a conservative choice for a flag but is not the water at a door. The ground-floor height of buildings is unknown, so "flooded" means water on the footprint, not inside.
8. **Boundary effects.** Water leaves freely at the domain edge (normal depth, slope 0.005), nothing enters. Cells near the edge are affected. The slope 0.005 is an assumption.
9. **Derived velocities.** Velocity and flow paths are derived from the water-surface slope with Manning's equation, not output by the solver. Where the surface is nearly flat, the dead band (slope under 1e-5) suppresses motion; in thin sheets it can be overestimated. The viewers say so.
10. **Impervious class.** Only 250,641 cells in the city are impervious in the county runs; buildings and roads elsewhere are treated as farmland.
11. **Time step control by the deepest cell.** One deep cell sets dt for the whole domain, which affects run time but not results beyond the scheme's stability.
12. **Mass balance reference.** The balance is relative to the SCS runoff volume, which assumes the SCS formula is the truth.

### 7.2 Data limits

- Rain: one gauge, uniform in space; a 24 hour window; hourly resolution for events (the real peak minute is higher than the hourly mean); a timestamp shift of up to an hour; return periods of events not assigned.
- Atlas 14 depths are rounded to within 3 % of the point estimates, and the confidence interval is not used.
- The city DEM tiles, the Manning raster build and the OSM extract timestamp are not recorded. County building count looks low relative to the city.
- Parcels cover the city only; county values come from OSM tags where parcels do not match.
- HIFLD and Census data were not used. No social vulnerability analysis exists.

### 7.3 Responsible use

- Present results as "modelled water depth for this storm", never as "your house will flood" or "your house is safe".
- Do not use for flood insurance, mortgage or permit decisions. For these, use FEMA maps and a licensed engineer.
- State that the model is not validated, has no rivers or drains, and that 10 cm of modelled depth is a screening threshold.
- Do not say a storm is "a 100-year flood". The return period applies to the rain depth only.
- Check the address in the official city and county records before reporting any single building.
- Credit the data: OpenStreetMap contributors (ODbL), USGS 3DEP, NOAA, Iowa Environmental Mesonet, TACC for compute.
