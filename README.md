![DSOC](https://img.shields.io/badge/DSOC-AI_Research_Lab-C8102E?style=for-the-badge)
![WKU](https://img.shields.io/badge/Western_Kentucky_University-Meteorology-CC0000?style=for-the-badge)
![TACC](https://img.shields.io/badge/TACC_Vista-H100_GPU-4a90d9?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Research-green?style=for-the-badge)
<img width="1087" height="495" alt="image" src="https://github.com/user-attachments/assets/174b89aa-a425-4ff5-b9b4-6fb846461967" />
<img width="750" height="238" alt="image" src="https://github.com/user-attachments/assets/ebbc3538-7d03-4981-a0f3-22840b41d8a1" />
<img width="493" height="116" alt="image" src="https://github.com/user-attachments/assets/01f2d983-c0f9-46a6-b5b2-a9d8e2565139" />
<img width="668" height="299" alt="image" src="https://github.com/user-attachments/assets/1f4212cf-8798-438b-8f9b-71cd2ac894f6" />


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
- **HPC flood simulation** — 2m resolution GPU-computed depth rasters (2-year storm baseline)
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


**CREDITS**

![Developer](https://img.shields.io/badge/Developer-Robert_G._Lesoski-C8102E?style=for-the-badge)
**Robert G. Lesoski**
Disaster Science Operations Center
Western Kentucky University
Meteorology-Freshman

![PI](https://img.shields.io/badge/Principal_Investigator-Dr._Manmeet_Singh-4a90d9?style=for-the-badge)

**Dr. Manmeet Singh**
Disaster Science Operations Center 
Western Kentucky University

---

![Contributor](https://img.shields.io/badge/Research_Contributor-Somnath_Liutel-555555?style=for-the-badge)

**Somnath Liutel**
Disaster Science Operations Center
Western Kentucky University
Graduate Student


*Disaster Science Operations Center (DSOC) · AI Research Lab (AIR) · Western Kentucky University*

> "Every storm is a dataset waiting to be solved; think global, model local."
