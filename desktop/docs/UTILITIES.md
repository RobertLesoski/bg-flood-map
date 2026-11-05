# Utilities and infrastructure layer

Water and power infrastructure from OpenStreetMap, with flood exposure from the county solver tiles.  Off by default: View menu
> "Utilities layer (water, power)" or the "Utilities" section of the sidebar (also per-type toggles = the legend).

## Build

`python scripts/build_utilities.py` (about 1 minute) reads `kentucky-latest.osm.pbf` with the desktop's pure-python reader and writes
`layers/utilities.geojson.gz` (point + LineString features; format in the script docstring).  Tests: `scripts/tests/test_utilities.py`
(synthetic PBF), `desktop/tests/test_utilities_ui.py`.  Exposure: `layers/cog/county/<storm>/max_depth.tif` (uint16 cm, 6.25 m,
65535 = no data counts as dry), the 12 storms, per storm `d` = maximum depth in cm (points: their pixel or footprint pixels; lines:
a sample every 3 m along them), lines also `f10` / `f30` = % of their length at >= 10 / >= 30 cm; `first` = first design storm
(2yr..100yr) with d >= 10 cm.  Source label everywhere: "solver depth (tiles, 6.25 m; verification pending)".

## Tags used

| Type | OSM tags |
|---|---|
| water_tower | man_made=water_tower |
| water_tank | man_made=storage_tank + content=water; man_made=reservoir_covered |
| water_works / wastewater_plant / pumping_station | man_made=water_works / wastewater_plant / pumping_station |
| dam / weir | waterway=dam / weir |
| hydro_plant | power=plant or generator with plant:source / generator:source containing hydro |
| power_plant | any other power=plant or generator |
| substation, transformer | power=substation, power=transformer (left out when more than 400) |
| power_line / minor_line / power_cable | power=line / minor_line / cable |
| pipeline | man_made=pipeline |

## Counts (county box, built 2026-11-05)

599 features: water_tower 19, water_works 1, wastewater_plant 6, dam 2, weir 1, power_plant 244 (mostly small generators, e.g. solar),
substation 44, transformer 50, power_line 120, minor_line 55, pipeline 57.  Flooded >= 10 cm in at least one design storm:
water_tower 3, water_works 1, wastewater_plant 5, dam 2, weir 1, power_plant 48, substation 13, transformer 3, power_line 78,
minor_line 14, pipeline 19.  Not found in the data: hydroelectric plants, water tanks / reservoirs, pumping stations.

## Limits (honest)

* OpenStreetMap completeness varies.  Relations (multipolygons) are NOT read: a plant, reservoir or dam mapped only as a relation is
  missing (this is probably why there is no hydro plant or reservoir).  Names are shown only where OSM has them.
* Towers / poles are not features; lines are as mapped (no capacity or ownership beyond the operator / voltage tags).
* Depth is the tile maximum over the whole storm, not the hour on the slider; verification of the tiles is pending.
* Pipelines and power lines are sampled at 3 m; a line outside the tile grid counts as dry.
* Drawing: markers and lines use the imgui background draw list (no GPU work); at county overview lines are decimated to about 1500
  vertices per frame.

## In the viewer

Markers: water tower triangle, tank circle, treatment / works square, dam / weir diamond, hydro hexagon, power plant pentagon, substation
cross, transformer dot (Okabe-Ito colours).  A white/red ring marks >= 10 cm for the storm shown.  Lines run blue-grey (dry) to thick red
(more of the length flooded).  Click a marker for a card; the address search also finds them ("water tower", "substation", a name).
