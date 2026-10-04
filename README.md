<img width="1087" height="495" alt="image" src="https://github.com/user-attachments/assets/53e21275-aa64-4c94-bd7a-7757b4739293" />

# Bowling Green, KY — 3D Flood Risk & Property Map

![DSOC](https://img.shields.io/badge/DSOC-AI_Research_Lab-C8102E?style=for-the-badge)
![WKU](https://img.shields.io/badge/Western_Kentucky_University-Meteorology-CC0000?style=for-the-badge)
![TACC](https://img.shields.io/badge/TACC_Vista-H100_GPU-4a90d9?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Research-green?style=for-the-badge)

An interactive 3D flood risk visualization for Bowling Green, Kentucky, built using real LiDAR terrain data, FEMA flood zones, NOAA Atlas 14 rainfall statistics, and GPU-accelerated hydrodynamic simulation on the TACC Vista supercomputer.

---

## ✦ Features

![MapLibre](https://img.shields.io/badge/MapLibre_GL_JS-4.7-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-CuPy_CUDA_12-yellow?style=flat-square)
![Resolution](https://img.shields.io/badge/Resolution-2m_Grid-orange?style=flat-square)
![Cells](https://img.shields.io/badge/Grid_Cells-80_Million-red?style=flat-square)

- **3D building extrusion** — 20,284 OSM structures color-coded by flood risk tier
- **Interactive flood overlay** — rainfall slider simulating 0–3" storm events
- **FEMA flood zones** — official NFHL data for Warren County (AE, AO, X zones)
- **Karst sinkhole mapping** — 3,872 closed depressions identified from LiDAR contours
- **HPC time-animated flood simulation** — Hours 1–9 of 2-year storm, toggled with time slider + play/pause
- **3D water volume mode** — flood polygons extruded by depth tier (shallow 3m → severe 50m)
- **Depth tier overlay** — shallow / moderate / deep / severe color-coded at 2m resolution
- **Building flood counter** — live count and % of buildings hit at each simulation hour
- **Elevation analysis** — building-level ground elevation from KyFromAbove Phase 2 LiDAR
- **Road network overlay** — OpenStreetMap road data with flood exposure analysis

---

## ✦ Data Sources

| Dataset | Source |
|---|---|
| 1m LiDAR DEM | USGS / KyFromAbove Phase 2 |
| FEMA Flood Zones | FEMA NFHL — Warren County, KY |
| Building footprints | OpenStreetMap |
| Rainfall depths | NOAA Atlas 14 — Warren County, KY |
| Road network | OpenStreetMap via Geofabrik |
| HPC Simulation | TACC Vista (NVIDIA H100) — Local Inertial Approximation |

---

## ✦ HPC Simulation

The flood depth layers are computed using a custom 2D GPU flood solver implementing the **Local Inertial Approximation** (Bates et al., 2010) — the same numerical scheme used by LISFLOOD-FP. The solver runs on TACC Vista's NVIDIA H100 Grace-Hopper nodes via CuPy, processing **80 million grid cells** per timestep at **2-meter resolution** across a 10×10 km domain centered on Bowling Green.

Planned storm return periods: **2, 5, 10, 25, 50, and 100-year** events using NOAA Atlas 14 rainfall depths distributed via the SCS Type II 24-hour hyetograph.

| Return Period | Rainfall Depth | Annual Probability |
|---|---|---|
| 2-year | 3.5 in | 50% |
| 5-year | 4.2 in | 20% |
| 10-year | 4.9 in | 10% |
| 25-year | 5.9 in | 4% |
| 50-year | 6.7 in | 2% |
| 100-year | 7.6 in | 1% |

![Allocation](https://img.shields.io/badge/Allocation-ATM25008-4a90d9?style=flat-square)
![Status](https://img.shields.io/badge/2yr_Storm-Complete-brightgreen?style=flat-square)
![Pending](https://img.shields.io/badge/5--100yr_Storms-Pending-yellow?style=flat-square)

---

## ✦ Running Locally

> **Note:** This repo contains the viewer and scripts only. The data files (GeoJSON layers, DEM, GPKG) are not included due to size. Follow the steps below to rebuild them.

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
