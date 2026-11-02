# Credits and Compute Record
**Bowling Green, KY — 3D Flood Risk Model**
Disaster Science Operations Center (DSOC) · AI Research Lab (AIR) · Western Kentucky University

*Prepared 2026-10-26. Items marked **[confirm]** are things the project files do not settle and should be checked before this is used in a paper, poster or report.*

---

## 1. People

| Role | Name | Affiliation |
|---|---|---|
| Developer | Robert G. Lesoski | Undergraduate Researcher, DSOC and AIR, Western Kentucky University (Meteorology) |
| Principal Investigator | Dr. Manmeet Singh | DSOC, Western Kentucky University |
| Research Contributor | Somnath Liutel | DSOC, Western Kentucky University (Graduate Student) |

**AI assistance.** Code, scripts and documentation were developed with Claude Code (Anthropic), using Claude Sonnet 4.6 and later Claude Sonnet 5.5. All results should be reviewed and validated by the research team before being relied on.

---

## 2. Compute resources

| Item | Detail |
|---|---|
| System | TACC Vista, Texas Advanced Computing Center, The University of Texas at Austin |
| Hardware used | NVIDIA H100 / Grace Hopper nodes, partition `gh` (one node and one GPU per storm) |
| Allocation | `ATM25008` (shown in the Vista job-submission checks) **[confirm the full allocation name and the funding program, e.g. TACC or ACCESS, for the acknowledgment]** |
| Solver | Custom GPU solver in Python with CuPy (CUDA 12): 2D local-inertial shallow-water scheme, 2 m grid, about 100 million cells (80.03 million valid) |
| Job manager | SLURM |

### Suggested acknowledgment (standard TACC wording)
> The authors acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for providing computational resources that have contributed to the research results reported within this work. URL: http://www.tacc.utexas.edu

If the allocation came through ACCESS, add the ACCESS allocation number in the form the program requires **[confirm]**.

---

## 3. Compute used

### 3a. Runs recorded so far
These come from the Vista accounting output (`sacct`) pasted during development. One node is allocated per storm.

| Run | Job ID | Storm | Wall time | Node-hours | State |
|---|---|---|---|---|---|
| First set, no infiltration, closed edges | 1092366_0 | 5-year | 15:55:29 | 15.92 | Completed |
| | 1092366_1 | 10-year | 16:21:56 | 16.37 | Completed |
| | 1092366_2 | 25-year | 16:52:14 | 16.87 | Completed |
| | 1092366_3 | 50-year | 17:14:03 | 17.23 | Completed |
| | 1092366_4 | 100-year | 17:35:13 | 17.59 | Completed |
| **Subtotal (first set)** | | | | **83.98** | |
| Solver v2 (infiltration + outflow) | 1099106_[0-4] | all five | 0 | 0 | Cancelled before starting |

### 3b. Other jobs that used time (not recorded in this file)
Earlier work also consumed node-hours, and the exact figures should be taken from Vista (section 5):
- The 2-year run (SCS design storm, about 6 h of wall time).
- The first attempts at the 5- to 100-year storms that were stopped or timed out (the 1000×-too-strong rain files and the debugging reruns).
- The post-processing job that converted the hourly rasters.
- Pending or in-progress: the solver-v2 smoke test (`bgsmoke`, short, on the dev queue) and the five solver-v2 storms.

### 3c. Totals (fill in from Vista)

| Quantity | Value |
|---|---|
| Allocation | ATM25008 |
| **Available SUs (Vista login banner, 2026-10-26)** | **8,299** |
| Allocation expires | 2027-08-31 |
| Allocation total (SU) | ______ (not shown in the banner; see `taccinfo` or the TACC User Portal) |
| Used to date (SU) | ______ |
| `/scratch` usage (2026-10-26) | 95.0 GB (no quota limit shown) |
| Total node-hours since 2026-10-01 | ______ (from the commands in section 7) |

*How node-hours convert to service units depends on the machine's charge factor. Use the balance printed by `taccinfo` for the authoritative figure.*

### 3d. Expected cost of the remaining planned runs
- Solver v2, five storms: about 17 to 22 node-hours each, roughly **85 to 110 node-hours** in total (estimated; v2 does somewhat more work per step than the first solver).
- Smoke test: well under 1 node-hour.

---

## 4. Data sources

| Data | Source | Used for |
|---|---|---|
| Terrain (elevation) | KyFromAbove, Commonwealth of Kentucky (LiDAR-derived DEM; fetched through the state's KyFromAbove image service and/or USGS 3DEP) **[confirm which DEM product and tile set built `dem_bg_2m.tif`]** | Flood simulation terrain, building ground elevation |
| Flood zones | FEMA National Flood Hazard Layer (NFHL), Warren County, KY | Zone overlay and comparison with simulated flooding |
| Parcels | Warren County Property Valuation Administrator (PVA), via the City of Bowling Green GIS service | Parcel values and risk tables |
| Buildings and roads | OpenStreetMap contributors (extract from Geofabrik, read with pyrosm) | Building footprints, road overlay |
| Rainfall depths | NOAA Atlas 14, Volume 2 Version 3, Precipitation Frequency Data Server, point 36.9685 N, 86.4808 W (Bowling Green), partial duration series. NOAA 24-hour depths: 3.54, 4.32, 4.97, 5.89, 6.65, 7.45 in for 2, 5, 10, 25, 50, 100 years. The runs used 3.5, 4.2, 4.9, 5.9, 6.7, 7.6 in, within 3% of NOAA (checked 2026-10-28). | Design storms (2 to 100 year) |
| Historical rainfall | Bowling Green-Warren County Regional Airport (KBWG) hourly observations, Iowa Environmental Mesonet ASOS archive (NOAA / FAA data) | Six historical storm blocks (built, not yet simulated) |
| Event context | NWS Louisville and NWS Paducah event summaries (May 2010, Sept 2017 Harvey remnants, April 2025) | Labelling and checking historical events |
| Base map | OpenStreetMap contributors; AWS Terrain Tiles (Terrarium) for 3D terrain **[confirm tile sources in the viewer]** | Viewer map and terrain |
| Sinkholes | Kentucky Geological Survey karst data (as loaded by the viewer) **[confirm]** | Sinkhole layer and roughness |

---

## 5. Methods and references

- **Flood solver:** Local inertial approximation of the shallow-water equations, the same scheme as the LISFLOOD-FP "acceleration" solver.
  - Bates, P. D., Horritt, M. S., and Fewtrell, T. J. (2010). A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. *Journal of Hydrology*, 387, 33–45.
  - de Almeida, G. A. M., Bates, P., Freer, J. E., and Souvignet, M. (2012). Improving the stability of a simple formulation of the shallow water equations for 2-D flood modeling. *Water Resources Research*, 48, W05528.
- **Friction:** Manning's equation. Roughness: 0.035 open land, 0.08 buildings, 0.12 sinkholes, 0.016 roads **[assumed typical values, not calibrated]**.
- **Infiltration (solver v2):** SCS (NRCS) Curve Number method, USDA NRCS National Engineering Handbook Part 630, Chapter 10 and TR-55. Curve numbers 75 (open land) and 98 (roads and buildings) **[assumed typical values, not calibrated]**.
- **Design-storm rainfall:** NOAA Atlas 14 depths with a 24-hour cumulative curve. The curve was checked against the first (2-year) run's results: domain-mean depth was 30.3 mm and 37.2 mm at hours 10 and 11 against 30.4 mm and 37.4 mm predicted by the curve.

### Historical storm blocks (built, not yet simulated; checked 2026-10-28)
Six 24-hour windows from the KBWG airport gauge (hourly, Iowa Environmental Mesonet), single-gauge, UTC. Compared with NOAA Atlas 14 (24-hr) at 36.9685, -86.4808 and with NWS Louisville pages.

| Event | 24-h total | Peak hour | vs NOAA Atlas 14 |
|---|---|---|---|
| ev1 Dec 2013 | 27.2 mm (1.07 in) | 4.1 mm/hr | below 2-yr |
| ev2 Mar 2015 | 54.1 mm (2.13 in) | 7.6 mm/hr | below 2-yr |
| ev3 Jun 2022 | 86.1 mm (3.39 in) | 24.4 mm/hr | just below 2-yr (3.54 in); 16th largest 24-h total in the gauge file since Jul 1996 |
| ev4 Apr 2025 | 126.2 mm (4.97 in) | 27.2 mm/hr | equals 10-yr (4.97 in) |
| ev5 Sep 2017 | 153.2 mm (6.03 in) | 29.2 mm/hr | between 25-yr (5.89) and 50-yr (6.65) |
| ev6 May 2010 | 188.2 mm (7.41 in) | 51.8 mm/hr | near 100-yr (7.45 in) |

Limits: ev4 is a 24-h slice of a multi-day flood (NWS 4-day Bowling Green record 10.04 in, previous 9.67 in). ev5 is a 24-h window (Aug 31 9 pm to Sep 1 9 pm Central), not the NWS calendar-day record of 4.93 in for Sept 1. ev6 is one 24-h window of a two-day event (NWS: just over 10 in in southern Warren County); 2 of its 24 hours are interpolated across gauge gaps (raw sum 7.18 in). ev3 is labelled "heavy" in the viewer data but is about a 2-year storm. Sources: NWS Louisville event pages (weather.gov/lmk: HistoricRainfallFloodingApril2-62025, september_2017_summary, flood_050210_how_it_happened).

### Verification status (what has and has not been checked)
- **Checked against known answers:** rain volume equals the storm depth; SCS runoff matches the textbook formula; water balance closes (0.000% error in tests); rainfall on an inclined plane matches Manning's equation within 0.3%; still water stays flat.
- **Not yet checked:** agreement with observed flooding (planned: comparison with FEMA zones and the 2010 and 2025 flood records); calibration of the curve numbers and edge-outflow slope; river inflow, channel flow, storm drains and sinkhole drainage are not modelled; rain is applied uniformly from one gauge.

---

## 6. Software

| Software | Role | Notes |
|---|---|---|
| Python | All scripting | |
| NumPy | Array computing | |
| CuPy (CUDA 12) | GPU solver | NVIDIA H100 on Vista |
| rasterio / GDAL | Raster input and output | |
| Shapely (GEOS) | Polygon processing | |
| GeoPandas, pyrosm, OSMnx | Data preparation scripts | Data integration and road and building extraction |
| Miniforge / conda-forge | Python environment on Vista | |
| SLURM | Job scheduling on Vista | |
| MapLibre GL JS 4.7 | Interactive 3D viewer | |
| Git and GitHub | Version control | github.com/RobertLesoski/bg-flood-map |
| Claude Code (Anthropic) | AI-assisted development | Claude Sonnet 4.6 and 5.5 |

---

## 7. How to get the compute numbers from Vista
Log in to Vista and run this one block. It prints the allocation balance and every job since 2026-10-01 with node-hours (and saves the table to `$SCRATCH/bgflood/compute_record.txt`). Paste the results into section 3c.

```bash
echo "=== Allocation balance (TACC) ==="; (/usr/local/etc/taccinfo 2>/dev/null || taccinfo 2>/dev/null || echo "taccinfo not found - check the TACC User Portal, Allocations page"); echo; echo "=== Every job since 1 Oct (also saved to $SCRATCH/bgflood/compute_record.txt) ==="; sacct -u rlesoski -S 2026-10-01 -X -P --format=JobID,JobName,Partition,State,Elapsed,AllocNodes,ElapsedRaw > $SCRATCH/bgflood/compute_record.txt; column -s'|' -t $SCRATCH/bgflood/compute_record.txt; echo; awk -F'|' 'NR>1 && $7 ~ /^[0-9]+$/ {nh += $7/3600*$6; n++} END {printf "Jobs counted: %d\nTotal node-hours since Oct 1: %.1f\n", n, nh}' $SCRATCH/bgflood/compute_record.txt
```

The TACC User Portal (Allocations page) also shows the same balance and usage by date.

---

*Disaster Science Operations Center (DSOC) · AI Research Lab (AIR) · Western Kentucky University*
