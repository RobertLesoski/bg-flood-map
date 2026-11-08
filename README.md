<img width="1087" height="495" alt="image" src="https://github.com/user-attachments/assets/53e21275-aa64-4c94-bd7a-7757b4739293" />
<img width="750" height="238" alt="image" src="https://github.com/user-attachments/assets/ebbc3538-7d03-4981-a0f3-22840b41d8a1" />
<img width="493" height="116" alt="image" src="https://github.com/user-attachments/assets/01f2d983-c0f9-46a6-b5b2-a9d8e2565139" />
<img width="668" height="299" alt="image" src="https://github.com/user-attachments/assets/1f4212cf-8798-438b-8f9b-71cd2ac894f6" />

# Bowling Green, KY — 3D Flood Risk & Property Map

![DSOC](https://img.shields.io/badge/DSOC-AI_Research_Lab-C8102E?style=for-the-badge)
![WKU](https://img.shields.io/badge/Western_Kentucky_University-Meteorology-CC0000?style=for-the-badge)
![TACC](https://img.shields.io/badge/TACC_Vista-H100_GPU-4a90d9?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Research-green?style=for-the-badge)

An interactive 3D flood risk visualization for Bowling Green and all of Warren County, Kentucky, built using real LiDAR terrain data, FEMA flood zones, NOAA Atlas 14 rainfall statistics, historical rain gauge records, and GPU-accelerated hydrodynamic simulation on the TACC Vista supercomputer. It comes as a web viewer (`viewer.html`) and a native GPU desktop viewer ([`desktop/`](desktop/README.md)).

> [!WARNING]
> **Disclaimer: flood depths are not validated.** The depths, extents and risk tiers shown here are output from a research model. They have not been validated against observed flooding, and they should not be treated as truth or used for safety, insurance, property, engineering or emergency decisions. The model does not include storm drains, karst drainage or river inflow from upstream, and its rainfall and infiltration settings are not calibrated. For official flood risk, consult FEMA flood maps and local authorities.

---

## ✦ Features

![MapLibre](https://img.shields.io/badge/MapLibre_GL_JS-4.7-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-CuPy_CUDA_12-yellow?style=flat-square)
![Resolution](https://img.shields.io/badge/Resolution-2m_city_%7C_5m_county-orange?style=flat-square)
![Cells](https://img.shields.io/badge/Grid_Cells-80M_city_%7C_125M_county-red?style=flat-square)

- **12 storm scenarios** — six design storms (2, 5, 10, 25, 50, 100-year, NOAA Atlas 14) and six historical rain events replayed from the Bowling Green gauge (Dec 2013, Mar 2015, Jun 2022, Apr 2025, Sep 2017 Harvey remnants, May 2010)
- **Two domains** — Bowling Green city (2 m grid) and all of Warren County (5 m grid), hours 0–24 of each storm
- **Real solver depth tiles** — Cloud-Optimized GeoTIFFs of the solver output, checked to within 1–2 cm of the solver rasters at 22,421 points ([docs/COG_VERIFICATION.md](docs/COG_VERIFICATION.md))
- **3D buildings and roads** — OpenStreetMap footprints and roads coloured by flood depth tier (10 / 30 / 60 / 120 cm)
- **Address search and risk card** — per-address depth by storm, with source labels and uncertainty
- **Critical facilities and utilities** — hospitals, schools, fire, police, shelters, water towers, treatment plants, substations, power lines (OpenStreetMap; coverage varies)
- **Analysis tools (desktop)** — measure, elevation profile, depth-over-time, flood timeline, storm compare and difference, flood duration, dry-route finder, building exposure filter with CSV/GeoJSON export
- **Presentation tools (desktop)** — camera bookmarks, guided tour, time-lapse, PNG/MP4 export, share card
- **FEMA flood zones and karst sinkholes** — NFHL zones and 3,872 closed depressions mapped from LiDAR
- **Honesty labels** — every view states the source, and that velocities are derived and depths unvalidated

---

## ✦ Data Sources

| Dataset | Source |
|---|---|
| 1m LiDAR DEM (city, resampled to 2 m) | USGS / KyFromAbove Phase 2 |
| County terrain (about 10 m, resampled to 5 m) | USGS 3DEP 1/3 arc-second |
| Historical hourly rain | Bowling Green gauge record (`scripts/storms_out/`) |
| FEMA Flood Zones | FEMA NFHL — Warren County, KY |
| Building footprints | OpenStreetMap |
| Rainfall depths | NOAA Atlas 14 — Warren County, KY |
| Road network | OpenStreetMap via Geofabrik |
| HPC Simulation | TACC Vista (NVIDIA H100) — Local Inertial Approximation |

---

## ✦ HPC Simulation

The flood depth layers are computed using a custom 2D GPU flood solver implementing the **Local Inertial Approximation** (Bates et al., 2010) — the same numerical scheme used by LISFLOOD-FP. The solver runs on TACC Vista's NVIDIA H100 Grace-Hopper nodes via CuPy, processing **80 million grid cells** per timestep at **2-meter resolution** across a 20×20 km domain centered on Bowling Green, and **125 million cells** at **5-meter resolution** across all of Warren County (about 52×60 km). Infiltration uses SCS curve numbers (75 open land, 98 roads and buildings), water leaves freely at the domain edge, and depth is saved every hour for 24 hours. The model has no storm drains, no karst drainage and no river inflow from upstream.

Storm return periods: **2, 5, 10, 25, 50, and 100-year** design events, plus six historical rain events. Full details per storm: [STORM_DATA_SOURCES.md](STORM_DATA_SOURCES.md). The 24-hour depths are close to NOAA Atlas 14 values (table below) and are spread over 24 hours with an 11-point cumulative curve that was checked against the 2-year run. It is **not** the official NRCS Type II distribution. Rain is applied uniformly over the domain.

| Return Period | Depth used in the runs | NOAA Atlas 14, 24-hr, Vol. 2 v3 (36.9685, -86.4808, PDS) | Difference | Annual Probability |
|---|---|---|---|---|
| 2-year | 3.5 in | 3.54 in | -0.04 in (-1.1%) | 50% |
| 5-year | 4.2 in | 4.32 in | -0.12 in (-2.8%) | 20% |
| 10-year | 4.9 in | 4.97 in | -0.07 in (-1.4%) | 10% |
| 25-year | 5.9 in | 5.89 in | +0.01 in (+0.2%) | 4% |
| 50-year | 6.7 in | 6.65 in | +0.05 in (+0.8%) | 2% |
| 100-year | 7.6 in | 7.45 in | +0.15 in (+2.0%) | 1% |

NOAA values were retrieved from the Precipitation Frequency Data Server on 2026-10-28 for downtown Bowling Green (partial duration series, depths in inches). The runs used rounded or earlier values, all within 3% of NOAA.

![Allocation](https://img.shields.io/badge/Allocation-ATM25008-4a90d9?style=flat-square)
![Status](https://img.shields.io/badge/City_runs-5yr--100yr_complete-brightgreen?style=flat-square)
![County](https://img.shields.io/badge/County_runs-12_storms_complete-brightgreen?style=flat-square)
![Validation](https://img.shields.io/badge/Depth_validation-not_done-orange?style=flat-square)

### Validation status

The depths are **not validated**. One first comparison was made against the April 2025 Bowling Green flood ([docs/VALIDATION_EV4.md](docs/VALIDATION_EV4.md)): 16 of 21 roads reported flooded reach 30 cm in the model (76%, 95% CI 55–89%), against 27 of 100 random city roads, with an AUC of about 0.85. Road length alone separates the two groups about as well, the replay used about half of the real four-day rainfall, and the observations record closures, not depths. This is weak evidence that the model finds where water collects, and no evidence about depth accuracy.

---

## ✦ Desktop Viewer (native GPU app)

A standalone Vulkan/wgpu application lives in [`desktop/`](desktop/README.md): city and county domains, all 12 storms,
terrain, animated water, buildings, roads, utilities, statistics, search and the analysis and presentation tools listed
above, with no server or internet needed. Measured at 1080p on a GTX 1650: about 125 fps (city) and 140 fps (county).
It opens on the May 2010 event (ev6) at hour 1. Start it with `desktop/Launch Desktop Viewer.bat`, or build a Windows
package with `desktop/packaging/build.ps1` (see [`desktop/packaging/SHIP_NOTES.md`](desktop/packaging/SHIP_NOTES.md)).
The data folder (about 7 GB) is not in this repo. See [`desktop/README.md`](desktop/README.md) for controls and limits.

## ✦ Running Locally

> **Note:** This repo contains the viewer and scripts. Most data files (GeoJSON layers, DEM, GPKG) are included; HPC simulation outputs are excluded due to size. Follow the steps below to rebuild them.

### Requirements
- Python 3.x with: `rasterio`, `geopandas`, `shapely`, `numpy`, `requests`
- GDAL command-line tools
- A local HTTP server (`python -m http.server 8000`)

### Data Setup

| File needed | Source | How to get it |
|---|---|---|
| `layers/fema.geojson` | FEMA NFHL | Run `fetch_fema.py` |
| `layers/buildings.geojson` | OpenStreetMap | Run `osm_roads.py` after downloading `kentucky-latest.osm.pbf` from [Geofabrik](https://download.geofabrik.de/north-america/us/kentucky.html) |
| `layers/sinkholes.geojson` | Derived from DEM | Run `flood_analysis.py` |
| `layers/roads.geojson` | OpenStreetMap | Run `osm_roads.py` |
| `dem_bg.tif` | USGS / [KyFromAbove](https://kyfromabove.ky.gov/) | Download 1m LiDAR tiles for Warren County, merge and resample to 2m with GDAL |
| `flood_risk.gpkg` | Generated | Run `step3_data_integration.py` after DEM and OSM data are in place |

### HPC Simulation Data Setup

The time-animated flood layers (`layers/flood_hpc/flood_h01.geojson` … `flood_h09.geojson`) are generated from TACC Vista output and are excluded from this repo due to file size (6–79 MB each). To regenerate:

1. Download the raw Vista outputs (`2yr_depth_003600s.geojson` … `2yr_depth_032400s.geojson`) from the simulation run
2. Run the reprojection script (pure Python, no external dependencies):
```bash
python scripts/reproject_hpc.py
```
This converts the files from UTM Zone 16N (EPSG:32616) to WGS84 and writes them to `layers/flood_hpc/`.

### Quick start
```bash
git clone https://github.com/RobertLesoski/bg-flood-map.git
cd bg-flood-map
pip install rasterio geopandas shapely numpy requests
python fetch_fema.py
python osm_roads.py
python step3_data_integration.py
# (Optional) Run HPC reprojection — see above
python -m http.server 8000
# Open http://localhost:8000/viewer.html
```

---

## ✦ Tech Stack

![MapLibre](https://img.shields.io/badge/MapLibre_GL_JS-4.7-blue?style=flat-square)
![CuPy](https://img.shields.io/badge/CuPy-CUDA_12-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.x-yellow?style=flat-square)
![GDAL](https://img.shields.io/badge/rasterio-GDAL-orange?style=flat-square)
![SLURM](https://img.shields.io/badge/SLURM-Array_Jobs-red?style=flat-square)

---

## ✦ Credits

### Developer

![Developer](https://img.shields.io/badge/Developer-Robert_G._Lesoski-C8102E?style=for-the-badge)

**Robert G. Lesoski**

Undergraduate Researcher — Disaster Science Operations Center (DSOC) & AI Research Lab (AIR)

Western Kentucky University · Meteorology, Freshman

*Made with [Claude Code](https://claude.ai/claude-code)*

---

### Principal Investigator

![PI](https://img.shields.io/badge/Principal_Investigator-Dr._Manmeet_Singh-4a90d9?style=for-the-badge)

**Dr. Manmeet Singh**

Disaster Science Operations Center (DSOC)

Western Kentucky University

---

### Research Contributor

![Contributor](https://img.shields.io/badge/Research_Contributor-Somnath_Liutel-555555?style=for-the-badge)

**Somnath Liutel**

Disaster Science Operations Center (DSOC)

Western Kentucky University · Graduate Student

---

*Disaster Science Operations Center (DSOC) · AI Research Lab (AIR) · Western Kentucky University*

> *"Every storm is a dataset waiting to be solved; think global, model local."*

## Running on your own computer

Double-click `Launch Map.bat` (needs Python); `Run Checks.bat` verifies the install. See [RUN_LOCALLY.md](RUN_LOCALLY.md) for the fluid/flow view and [FLOW_FORMAT.md](FLOW_FORMAT.md) for the data format. Flow vectors are derived from the simulated water-surface slope (Manning), not raw solver output.
