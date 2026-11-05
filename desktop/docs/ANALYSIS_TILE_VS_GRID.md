# Tile-derived vs flow-grid-derived depth (Warren County)

Tiles: `C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map\layers\cog\county` (6.25 m, EPSG:3857, uint16 cm), label: *solver depth (tiles, 6.25 m; verification pending)*.
Flow grid: `layers/flow_county/<storm>` (25 m Mercator cells, about 20 m on the ground; the builder block-averages the solver depth).
Generated 2026-11-05 13:05 by `python -m bgv.analysis.verify --points 150 --seed 1`.

## 1. Depth at random points (bilinear on both sides; wet = depth >= 0.02 m)

| storm | hour | points | wet in either | both wet | tiles only | grid only | mean (tile - grid) m | mean abs m | RMS m | p50 abs | p95 abs | max abs | max tile | max grid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100yr | 12 | 150 | 75 | 52 | 8 | 15 | +0.011 | 0.082 | 0.126 | 0.051 | 0.292 | 0.53 | 1.34 | 0.92 |
| 100yr | 24 | 150 | 75 | 51 | 7 | 17 | +0.046 | 0.154 | 0.300 | 0.064 | 0.505 | 1.87 | 6.25 | 6.24 |
| 25yr | 12 | 150 | 75 | 40 | 13 | 22 | +0.038 | 0.088 | 0.160 | 0.041 | 0.268 | 0.90 | 2.21 | 1.31 |
| 2yr | 18 | 150 | 75 | 42 | 6 | 27 | +0.021 | 0.085 | 0.124 | 0.049 | 0.267 | 0.40 | 1.99 | 1.61 |
| ev4 | 18 | 150 | 75 | 39 | 12 | 24 | +0.029 | 0.092 | 0.185 | 0.037 | 0.386 | 0.91 | 1.69 | 0.88 |
| ev6 | 12 | 150 | 75 | 42 | 7 | 26 | +0.030 | 0.074 | 0.137 | 0.035 | 0.223 | 0.66 | 2.06 | 1.40 |

900 points in total; the statistics are over the wet-in-either points of each row.

## 2. Domain-wide numbers of the same hour

Scene: 24,620 OSM buildings, 7,340 km of roads.

| storm | hour | flooded buildings tiles / grid | wet roads km tiles / grid | wet area km2 tiles / grid | max depth m tiles / grid | volume million m3 tiles / grid |
|---|---|---|---|---|---|---|
| 100yr | 6 | 142 / 37 | 4.3 / 1.1 | 9.4 / 1.5 | 1.42 / 0.43 | 0.3 / 0.1 |
| 100yr | 12 | 1,432 / 1,903 | 185.4 / 272.9 | 375.4 / 479.0 | 10.27 / 8.00 | 112.8 / 110.0 |
| 100yr | 24 | 2,154 / 3,019 | 311.2 / 482.3 | 440.6 / 571.7 | 19.20 / 8.00 | 323.8 / 321.6 |
| 25yr | 12 | 1,116 / 1,419 | 129.5 / 179.2 | 306.0 / 394.9 | 9.28 / 6.28 | 67.8 / 65.2 |
| 2yr | 18 | 936 / 1,295 | 106.5 / 153.5 | 232.9 / 324.9 | 9.32 / 6.45 | 64.8 / 62.9 |

Tile rule: flooded building = depth >= 0.1 m at any footprint corner or the centroid (nearest 6.25 m cell); wet road = 10 m road samples with depth >= 0.1 m; area = cells >= 0.02 m. Flow-grid rule: data/stats.py (all-touched pixels, same thresholds).
