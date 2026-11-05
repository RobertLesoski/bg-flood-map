# Validation of model event ev4 against the April 2025 Bowling Green flood

Date of analysis: 2026-11-05. First attempt to compare this model with observed flooding. Everything here is reproducible with
`python scripts/validate_ev4.py` from `layers/validation/ev4_observations.json`; scores are written to `layers/validation/ev4_scores.json`
and the figure to `docs/VALIDATION_EV4_map.png`. Unit tests: `scripts/tests/test_validate_ev4.py`.

## 1. Bottom line

* The model is **not validated** by this exercise, and the sample cannot validate it. What the evidence supports: on the roads that
  officials named as flooded or closed, the model's maximum depth is clearly higher than on randomly picked named roads in the city
  (rank statistic AUC 0.84, 21 observed vs 100 control roads; median of the road maximum 0.62 m vs 0.08 m). That is a
  weak sign of skill in *where* water collects. It is not a measurement of accuracy.
* Two separate reasons the comparison is loose: (1) **time window** - the model replays 24 h of rain starting 2025-04-03 13:00 UTC
  (4.97 in at the airport gauge); the real flood ran April 2-6 with 10.5 in at the same airport gauge (13.1 in at WKU), so roughly
  twice the rain over four days, and most reported closures are dated April 4-8, after the model window; (2) the model has **no upstream
  river inflow**, but much of the observed flooding was river or creek water.
* Because observed roads are matched by name, a whole named road is scored even though only part of it flooded. "Flagged" is therefore a
  generous test, and the control set is not a set of dry roads: roads not named in the news are *unknown*, not dry.
  No true false alarm rate can be computed. The control only shows how often the model flags an arbitrary road.

## 2. Method

1. **Observations** (section 5) were collected from public pages and APIs on 2026-11-05 and paraphrased (no long quotes). Each has a
   source, type, date, confidence note. Nothing was invented; where a page gave no location or number, the field is empty.
   The City page bgky.org/announcements/1527 now shows different (2026) content, so the city statements were taken from news articles
   that relay them; the 196-structure count comes from the Daily News article of April 12 (and a WNKY article that says about 200).
2. **Model values** come from `layers/cog/county/ev4/max_depth.tif` (EPSG:3857, uint16 cm, 65535 nodata, 6.25 mercator m, about 5 m on the
   ground). Roads are the OSM named ways in `layers/county/roads.geojson.gz`. A road is matched by exact (case-insensitive) name; I did
   not guess near-matches (Regents Way vs Regents Avenue etc.), except Glen Lilly/Lily, Normal Street/St, and Weather Stone /
   Weatherstone Boulevard, which are marked as assumptions. All ways of the name are pooled; for Nashville, Louisville and Russellville
   Roads only ways inside the city box (lon -86.56..-86.33, lat 36.90..37.07) are used. Each line is sampled every 6.25 m (mercator);
   I report the maximum depth and the fraction of samples (= length) at or above **0.10 m and 0.30 m**.
3. **Flag rules**: "any" = some sample at or above the threshold (very generous for a long road); "20%" = at least 20% of the road length at or above it.
4. **Control**: 100 randomly chosen named roads (seed 42, not motorway/trunk, not an observed name) from the 1591 candidates inside the city box.
   Truth is **unknown** for them. I also drew 300 random points in the city box for a null baseline of point values.
5. **Hit rate** is shown only with the control rate beside it. Intervals are Wilson 95% intervals.

## 3. Scores

### 3a. Roads (observed n = 21 scorable named roads with a decent match, of 41 observed road items; 20 could not be scored)

| Rule | Observed roads flagged | Control roads flagged (unknown truth) |
|---|---|---|
| depth >= 0.10 m, any sample | 21/21 (100%, 95% CI 85-100%) vs control 42/100 (42%, CI 33-52%) |
| depth >= 0.10 m, >= 20% of length | 4/21 (19%, 95% CI 8-40%) vs control 13/100 (13%, CI 8-21%) |
| depth >= 0.30 m, any sample | 16/21 (76%, 95% CI 55-89%) vs control 27/100 (27%, CI 19-36%) |
| depth >= 0.30 m, >= 20% of length | 2/21 (10%, 95% CI 3-29%) vs control 8/100 (8%, CI 4-15%) |

* Reading: at 0.10 m the model "hits" every observed road, but it also flags 42% of arbitrary city roads, so the hit is nearly free (the model has 2-3 cm
  rain-on-grid ponding everywhere and 5 m cells pick up single deep cells). At 0.30 m the hit rate falls to 76% against 27% of arbitrary roads.
  Requiring 20% of length to be wet, the model catches only 2-4 of 21 observed roads (mostly because the reported segment is short, the road
  is long, or the cause was river water the model lacks).
* AUC on road maximum depth: **0.84** (0.5 = no skill, 1 = perfect). Roads shorter than 2000 m only (n=13 observed, 95 control) give 0.84, so the
  result is not only a road-length effect. With n = 21 the uncertainty is large (no confidence interval was computed).
* Largest model values on observed roads: Russellville Road (max 3.52 m, only 5% of its
  length >= 0.3 m, so the 3.5 m is a few cells, not checked on the ground), Brewster Lane (1.93 m, 34% of length >= 0.3 m), Glen Lily Road (1.84 m, 0% of length), Sumpter Avenue (1.34 m, 19%).
* Biggest misses (model max below 0.3 m on a road that was reported flooded): Cal Batsel Road (0.18 m), Boatlanding Road (0.15), Fair Street (0.11),
  Normal Street (0.20), Turkey Run Drive (0.18). Three Springs Road is a near miss (max 0.33 m, none of its length at or above 0.3 m in a second look at the fractions: 0%); the source ties its flooding to the Basil Griffin Park lake, a water body the model may not represent.
  I did not check whether any of these roads lie in a river floodplain, so I do not claim these misses are caused by the missing river inflow.
* The model's own road product (`layers/roads_flooded/county/ev4.json`, built from the 20 m flow grid) agrees in direction: flooded share of length for the observed roads
  ranges 0-40% (see `ev4_scores.json`).

### 3b. Points (NWS Local Storm Reports; positions are approximate NWS/mPING positions)

| Report | Model at point | Max within 100 m |
|---|---|---|
| 1 N Plano (creek overflow, yard/basement flooding) | 0.00 | 2.63 |
| 1 NW Bowling Green (road flooded and closed, stranded vehicles) | 0.01 | 0.08 |
| 1 SW Bowling Green (creek overflow, yard/basement flooding) | 0.00 | 2.76 |
| 2 NNW Rich Pond (water rescues, McLellan neighborhood) | 0.02 | 0.68 |
| 2 NW Rich Pond (car stranded, County House Lane) | 0.00 | 1.39 |
| Woodburn (road flooded and closed) | 0.01 | 0.40 |
| 7 E Richardsville (road flooded, vehicles stranded) | 0.00 | 6.05 |

Baseline: for 300 random points in the city box the maximum within 100 m is >= 0.10 m for 79% and >= 0.30 m for 65% of points, so a "max within 100 m" is a weak
test (nearly any location has some deep cell within 100 m, including the Barren River channel). The model value *at* the point is about 0 in all seven reports. Three of these
reports are creek or river overflow in valleys (Plano, SW Bowling Green, Richardsville): the 2.6, 2.8 and 6.1 m values nearby are most likely the stream channel itself,
not flooded land. No reliable conclusion for points.

Two reports of "not flooded" (Michael O. Buchanon Park at 9222 Nashville Road, Spring Lakes subdivision) are only absences of reports; the park address is not in the
address index and the subdivision is a single short road, so they are **not scored**.

### 3c. Gauges (Barren River and Drakes Creek)

| Gauge | Rise start (CDT) | Peak (CDT) | Peak (UTC) | Peak stage / flow |
|---|---|---|---|---|
| USGS 03314500 Barren River at Bowling Green | 2025-04-03 15:45 | 2025-04-04 17:30 | 2025-04-04 22:30 | 35.76 ft / 33,100 cfs |
| USGS 03314000 Drakes Creek near Alvaton | 2025-04-03 09:30 | 2025-04-04 02:00 | 2025-04-04 07:00 | 32.41 ft / 35,300 cfs |

Model rain window: 2025-04-03 13:00 UTC to 2025-04-04 13:00 UTC (4.97 in; the heaviest hours are 21-22 of the window, 10:00-12:00 UTC April 4, 2.06 in in two hours). Drakes
Creek (a Warren County stream, inside the model domain) peaked at 07:00 UTC, inside the window and **before** the model's heaviest rain, so it responded to the earlier rain
of April 2-3 that the model run does not contain (and to rain upstream). The Barren River peak came about 9.5 h **after** the model window ended, as expected for a large basin with
upstream inflow. The model has no upstream inflow and only 24 h of rain, so it cannot reproduce either hydrograph and is expected to under-predict river flooding. Timing
comparison is qualitative only; the model outputs no stage or flow at the gauge sites and I did not locate them on the grid. NWS lists the Barren River at Bowling Green in
moderate flood (no crest value on the page). USGS values are provisional.

### 3d. Damaged structures (196, city of Bowling Green, as of April 10)

The model has 10660 building/address records inside the city box, of which 623 have max depth >= 0.10 m, 262 >= 0.30 m and 32 >= 1.0 m (depth taken as the maximum of 3x3 pixels at the record
point). These **cannot be compared with 196**: the city reports 'damaged' structures in 10 unmapped zones after 4+ days and mostly river, creek, cave and stormwater-basin
water, not a 24 h rain block; my box is not the city limit; and 196 gives no list of addresses. The numbers are of the same order (hundreds), which is not evidence of agreement.
A fair comparison needs the city's damaged-address list.

## 4. Limitations

* Small sample (21 scorable roads, 7 coarse points, 2 gauges, 1 count). Many roads came from one WBKO list that the article itself calls incomplete.
* Name matching pools all ways of a name; only part of a road flooded; unmatched names (20) were dropped, which may bias the sample either way.
* Controls are unknown-truth; the missed observations in news are not a census.
* Time window and rain total differ from the real event; no river inflow; coarse terrain outside the city (about 10 m); no calibration.
* Observed depths are not available (reports say closed/impassable, not depth), so only presence/absence is tested, never depth error.
* Sources are secondary (news relaying city statements); the Blue Level Road notice and the 196 breakdown were read through summaries.
* Provisional USGS data.

## 5. What this does and does not show

Shows: the model puts clearly more water on the roads that flooded than on a typical road (AUC about 0.8), and it already catches the reported hot spots such as Brewster Lane,
Sumpter Avenue, Red Clover Avenue and Barberry Court. Does not show: that the depths are right, that a flood map for a real storm would be right, that the model's flagged roads
are really flooded (42% of random roads exceed 0.1 m; 27% exceed 0.3 m), or anything about river flooding, which the model cannot produce. A proper test needs the full April 2-6
rain, upstream boundary inflow, point observations with depth (high-water marks, closure polygons), and a list of dry roads.

Figure: `docs/VALIDATION_EV4_map.png` (model max depth in blue, observed roads in red drawn whole, NWS reports orange).

## 6. All observations

Sources accessed 2026-11-05. The `model` column is my scoring output. Full JSON: `layers/validation/ev4_observations.json`, raw downloads in `layers/validation/raw/`
(NWS LSR CSV from the Iowa Environmental Mesonet, USGS 15-minute JSON).

| id | type | name | date | what it says (paraphrase) | confidence | model | link |
|---|---|---|---|---|---|---|---|
| obs01 | road_closed | Russellville Road | 2025-04-04 | Floodwater over the road near the railroad bridge by WKU; barricades placed. | high that flooding occurred; segment only roughly located (near WKU) | model max 3.52 m; >=0.1 m on 12% of length; >=0.3 m on 5% | [source](https://bgdailynews.com/?p=685010) |
| obs02 | road_closed | Interstate 165 exit to Russellville Road | 2025-04-04 | I-165 exit onto Russellville Road closed for flooding (ramp only). | high for closure; ramp not separate in road data, whole I 165 matched (poor match) | not scored | [source](https://bgdailynews.com/?p=685010) |
| obs03 | road_closed | Nashville Road | 2025-04-06 | Closed for flooding (WBKO); named a hardest-hit corridor within city limits (Daily News). | high that parts flooded; segment not identified | model max 0.44 m; >=0.1 m on 6% of length; >=0.3 m on 1% | [source](https://www.wbko.com/2025/04/06/bowling-green-declares-state-emergency-amid-flooding/) |
| obs04 | road_closed | Louisville Road | 2025-04-06 | Named, with Nashville and Russellville Roads, as hardest-hit corridors within city limits. | medium; segment not identified | model max 0.85 m; >=0.1 m on 3% of length; >=0.3 m on 1% | [source](https://bgdailynews.com/?p=685107) |
| obs05 | road_closed | Cave Mill Road | 2025-04-04/08 | Closed for flooding Apr 4 (WBKO list), Apr 6 (vehicle pulled from deep water), impassable Apr 8; homes flooded by Lost River Cave water. | high | model max 0.71 m; >=0.1 m on 3% of length; >=0.3 m on 2% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs06 | road_closed | Brewster Lane | 2025-04-04/06 | On WBKO closure list Apr 4; city called it significantly compromised Apr 6. | high | model max 1.93 m; >=0.1 m on 36% of length; >=0.3 m on 34% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs07 | road_closed | Glen Lilly Road | 2025-04-04 | On WBKO list; published spelling Lilly, OSM spelling Lily (assumed same road). | medium (spelling assumption) | model max 1.84 m; >=0.1 m on 1% of length; >=0.3 m on 0% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs08 | road_closed | Cal Batsel Road | 2025-04-04 | On WBKO list of significantly flooded or closed roads. | high | model max 0.18 m; >=0.1 m on 2% of length; >=0.3 m on 0% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs09 | road_closed | Smallhouse Road | 2025-04-04 | 1700 block flooded or closed (WBKO list). | high for road; block not located | model max 0.62 m; >=0.1 m on 4% of length; >=0.3 m on 1% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs10 | road_closed | Rodeo Lane | 2025-04-04 | On WBKO list. | high | model max 0.73 m; >=0.1 m on 18% of length; >=0.3 m on 9% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs11 | road_closed | Sumpter Avenue | 2025-04-04 | On WBKO list; Daily News: floodwater over driveways at Sumpter Ave and Normal St. | high | model max 1.34 m; >=0.1 m on 21% of length; >=0.3 m on 19% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs12 | road_closed | Boatlanding Road | 2025-04-04 | On WBKO list. | high | model max 0.15 m; >=0.1 m on 2% of length; >=0.3 m on 0% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs13 | road_closed | Crewdson Drive | 2025-04-04 | On WBKO list. | high | model max 0.81 m; >=0.1 m on 11% of length; >=0.3 m on 9% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs14 | road_closed | Fair Street | 2025-04-04 | On WBKO list. | medium (3 ways of that name in data) | model max 0.11 m; >=0.1 m on 1% of length; >=0.3 m on 0% | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs15 | road_closed | Normal Street | 2025-04-04 | Daily News: floodwater at Sumpter Avenue and Normal Street (driveways, car in ditch). | medium (OSM abbreviation Normal St) | model max 0.20 m; >=0.1 m on 3% of length; >=0.3 m on 0% | [source](https://bgdailynews.com/?p=685010) |
| obs16 | road_closed | Blue Level Road | 2025-04-06 | City notice: impassable between Russellville Road and Worth Way; at least 3 homes flooded. | medium: city notice as relayed in a search summary; original notice not read directly | model max 0.55 m; >=0.1 m on 3% of length; >=0.3 m on 1% | [source](https://bgdailynews.com/2025/04/12/historic-floodwaters-force-families-from-homes) |
| obs17 | road_closed | Three Springs Road | 2025-04-07 | Covered by water between Basil Griffin Park and the KOA campground; temporary bypass built. | high | model max 0.33 m; >=0.1 m on 2% of length; >=0.3 m on 0% | [source](https://bgdailynews.com/?p=685146) |
| obs18 | road_closed | Turkey Run Drive | 2025-04-06 | Flooding shown in photos (Daily News). | low-medium (photo caption in summary) | model max 0.18 m; >=0.1 m on 5% of length; >=0.3 m on 0% | [source](https://bgdailynews.com/?p=685107) |
| obs19 | road_closed | Red Clover Avenue | 2025-04-06 | Flooding shown in photos (Daily News). | low-medium | model max 1.06 m; >=0.1 m on 20% of length; >=0.3 m on 14% | [source](https://bgdailynews.com/?p=685107) |
| obs20 | road_closed | Barberry Court | 2025-04-06 | Sporadic flooding (city update). | medium | model max 0.55 m; >=0.1 m on 28% of length; >=0.3 m on 21% | [source](https://bgdailynews.com/?p=685107) |
| obs21 | road_closed | Weather Stone subdivision entrance | 2025-04-04 | Floodwater covered the entryway off Russellville Road; match to Weatherstone Boulevard is my assumption. | low (name assumption) | model max 0.46 m; >=0.1 m on 19% of length; >=0.3 m on 7% | [source](https://bgdailynews.com/?p=685010) |
| obs22 | road_closed | Marty Drive | 2025-04-03T22:48Z | NWS LSR (IEM): 2-3 inches of standing water across the road, 22:48 UTC Apr 3 (inside the model window). LSR position 36.92N 86.54W. | medium (single report, shallow) | model max 0.64 m; >=0.1 m on 25% of length; >=0.3 m on 18% | [source](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo=LMK&sts=2025-04-02T00:00Z&ets=2025-04-07T00:00Z&fmt=csv) |
| obs23 | road_closed | Brighton Way | 2025-04-04/06 | Reported flooded or closed; not scorable: no road of that exact name in data | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs24 | road_closed | Regents Way | 2025-04-04/06 | Reported flooded or closed; not scorable: data has Regents Avenue only, not assumed same | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs25 | road_closed | Creason Drive | 2025-04-04/06 | Reported flooded or closed; not scorable: data has Creason Street only | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs26 | road_closed | Covington | 2025-04-04/06 | Reported flooded or closed; not scorable: no suffix; ambiguous | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs27 | road_closed | Wakefield | 2025-04-04/06 | Reported flooded or closed; not scorable: no suffix; ambiguous | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs28 | road_closed | Logan Street at Chestnut Street | 2025-04-04/06 | Reported flooded or closed; not scorable: no Logan Street in data | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs29 | road_closed | Rockingham Road at Westen Street | 2025-04-04/06 | Reported flooded or closed; not scorable: data has Rockingham Avenue; not assumed | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs30 | road_closed | South Lee Street | 2025-04-04/06 | Reported flooded or closed; not scorable: data has Lee Drive variants only | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs31 | road_closed | North Lee Street | 2025-04-04/06 | Reported flooded or closed; not scorable: data has Lee Drive variants only | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs32 | road_closed | Brewster Road | 2025-04-04/06 | Reported flooded or closed; not scorable: listed in the same article as Brewster Lane; no such road in data | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs33 | road_closed | WKU railroad underpass | 2025-04-04/06 | Reported flooded or closed; not scorable: not a named road | high that reported; not scorable (no safe name match) | not scored | [source](https://www.wbko.com/2025/04/04/many-roads-impassable-historic-flooding-lingers-bowling-green/) |
| obs34 | road_closed | Brentwood Place | 2025-04-04/06 | Reported flooded or closed; not scorable: no matching road name | high that reported; not scorable (no safe name match) | not scored | [source](https://bgdailynews.com/?p=685107) |
| obs35 | road_closed | Sumter Way | 2025-04-04/06 | Reported flooded or closed; not scorable: no matching road name (Sumpter Avenue is separate) | high that reported; not scorable (no safe name match) | not scored | [source](https://bgdailynews.com/?p=685107) |
| obs36 | road_closed | KY 626 (MP 4.5) | 2025-04-04 | KYTC bulletin 1:33 PM EDT Apr 4: Warren County segment closed for high water. Route numbers are not in the name-based road data. | high that reported; not scorable (no route refs) | not scored | [source](https://content.govdelivery.com/accounts/KYTC/bulletins/3da7209) |
| obs37 | road_closed | KY 2665 (MP 5.5) | 2025-04-04 | KYTC bulletin 1:33 PM EDT Apr 4: Warren County segment closed for high water. Route numbers are not in the name-based road data. | high that reported; not scorable (no route refs) | not scored | [source](https://content.govdelivery.com/accounts/KYTC/bulletins/3da7209) |
| obs38 | road_closed | KY 1083 (MP 3 and 5) | 2025-04-04 | KYTC bulletin 1:33 PM EDT Apr 4: Warren County segment closed for high water. Route numbers are not in the name-based road data. | high that reported; not scorable (no route refs) | not scored | [source](https://content.govdelivery.com/accounts/KYTC/bulletins/3da7209) |
| obs39 | road_closed | KY 1435 (MP 9-11) | 2025-04-04 | KYTC bulletin 1:33 PM EDT Apr 4: Warren County segment closed for high water. Route numbers are not in the name-based road data. | high that reported; not scorable (no route refs) | not scored | [source](https://content.govdelivery.com/accounts/KYTC/bulletins/3da7209) |
| obs40 | road_closed | KY 242 (MP 7) | 2025-04-04 | KYTC bulletin 1:33 PM EDT Apr 4: Warren County segment closed for high water. Route numbers are not in the name-based road data. | high that reported; not scorable (no route refs) | not scored | [source](https://content.govdelivery.com/accounts/KYTC/bulletins/3da7209) |
| obs41 | road_closed | US 68 (MP 27.4, Barren Co line) | 2025-04-04 | KYTC bulletin 1:33 PM EDT Apr 4: Warren County segment closed for high water. Route numbers are not in the name-based road data. | high that reported; not scorable (no route refs) | not scored | [source](https://content.govdelivery.com/accounts/KYTC/bulletins/3da7209) |
| obs42 | flooded_property | 1 N Plano (creek overflow, yard/basement flooding) | 2025-04-03T23:18Z | NWS LSR from mPING crowd report: creek overflowing; yard/basement flooding. | low-medium (crowd report, position approximate) | model 0.00 m at point; max within 100 m 2.63 m | [source](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo=LMK&sts=2025-04-02T00:00Z&ets=2025-04-07T00:00Z&fmt=csv) |
| obs43 | flooded_property | 1 NW Bowling Green (road flooded and closed, stranded vehicles) | 2025-04-04T15:32Z | NWS LSR from mPING: road flooded and closed, vehicles stranded. About 2.5 h after the model window ends. | low-medium | model 0.01 m at point; max within 100 m 0.08 m | [source](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo=LMK&sts=2025-04-02T00:00Z&ets=2025-04-07T00:00Z&fmt=csv) |
| obs44 | flooded_property | 1 SW Bowling Green (creek overflow, yard/basement flooding) | 2025-04-04T16:03Z | NWS LSR from mPING: creek overflowing; yard/basement flooding. | low-medium | model 0.00 m at point; max within 100 m 2.76 m | [source](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo=LMK&sts=2025-04-02T00:00Z&ets=2025-04-07T00:00Z&fmt=csv) |
| obs45 | flooded_property | 2 NNW Rich Pond (water rescues, McLellan neighborhood) | 2025-04-06T03:45Z | NWS LSR: multiple water rescues in the McLellan neighborhood. About 2 days after the model window. | medium | model 0.02 m at point; max within 100 m 0.68 m | [source](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo=LMK&sts=2025-04-02T00:00Z&ets=2025-04-07T00:00Z&fmt=csv) |
| obs46 | flooded_property | 2 NW Rich Pond (car stranded, County House Lane) | 2025-04-06T01:15Z | NWS LSR: dispatch reports a car stranded in high water on County House Lane. | medium | model 0.00 m at point; max within 100 m 1.39 m | [source](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo=LMK&sts=2025-04-02T00:00Z&ets=2025-04-07T00:00Z&fmt=csv) |
| obs47 | flooded_property | Woodburn (road flooded and closed) | 2025-04-06T04:36Z | NWS LSR from mPING: road flooding and closure at Woodburn. | low-medium | model 0.01 m at point; max within 100 m 0.40 m | [source](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo=LMK&sts=2025-04-02T00:00Z&ets=2025-04-07T00:00Z&fmt=csv) |
| obs48 | flooded_property | 7 E Richardsville (road flooded, vehicles stranded) | 2025-04-04T13:16Z | NWS LSR from mPING; Richardsville is in the Barren River valley north-east of the city. | low-medium | model 0.00 m at point; max within 100 m 6.05 m | [source](https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?wfo=LMK&sts=2025-04-02T00:00Z&ets=2025-04-07T00:00Z&fmt=csv) |
| obs49 | flooded_property | Basil Griffin Park | 2025-04-04 | Park flooded Apr 4 (signs submerged); lake rose Apr 5-6. Coordinates not in repo data, so not located. | high that flooded; location unknown |  | [source](https://bgdailynews.com/?p=685010) |
| obs50 | flooded_property | KOA Campground / Spring Lakes access | 2025-04-07 | About 70 RVs trapped by rising water; temporary exit road built. Location not in data. | high that flooded; location unknown |  | [source](https://bgdailynews.com/?p=685146) |
| obs51 | flooded_property | Cave Mill - Hunting Creek homes (Lost River Cave overflow) | 2025-04-05/07 | Several homes flooded by water from the Lost River Cave system; no house numbers. | high that flooded; no address |  | [source](https://bgdailynews.com/2025/04/12/historic-floodwaters-force-families-from-homes) |
| obs52 | flooded_property | Brentwood Place stormwater basin home | 2025-04-05 | Stormwater basin overflowed; about 4 in of water entered one home. No address. | medium; no location |  | [source](https://bgdailynews.com/2025/04/12/historic-floodwaters-force-families-from-homes) |
| obs53 | not_flooded | Michael O. Buchanon Park, 9222 Nashville Road | 2025-04-07 | Not reported as flooded; used as a Red Cross shelter from Apr 7 and recovery centre from Apr 14. | low-medium: 'not reported flooded' is absence of evidence |  | [source](https://bgdailynews.com/?p=685146) |
| obs54 | not_flooded | Spring Lakes subdivision (Spring Lakes Circle) | 2025-04-06 | Source says cut off and used for detours but does not say it flooded itself. | low: ambiguous |  | [source](https://bgdailynews.com/?p=685107) |
| obs55 | rain | Bowling Green WKU COOP, Apr 1-7 | 2025-04-01/07 | NWS Louisville table: 13.11 in over Apr 1-7. | high (official table); window longer than the model's 24 h |  | [source](https://www.weather.gov/lmk/HistoricRainfallFloodingApril2-62025) |
| obs56 | rain | Bowling Green COOP, Apr 1-7 | 2025-04-01/07 | NWS Louisville table: 11.88 in over Apr 1-7. | high (official table); window longer than the model's 24 h |  | [source](https://www.weather.gov/lmk/HistoricRainfallFloodingApril2-62025) |
| obs57 | rain | Bowling Green Warren County Airport, Apr 1-7 | 2025-04-01/07 | NWS Louisville table: 10.48 in over Apr 1-7. | high (official table); window longer than the model's 24 h |  | [source](https://www.weather.gov/lmk/HistoricRainfallFloodingApril2-62025) |
| obs58 | rain | Bowling Green 4.3 S CoCoRaHS, Apr 1-7 | 2025-04-01/07 | NWS Louisville table: 9.30 in over Apr 1-7. | high (official table); window longer than the model's 24 h |  | [source](https://www.weather.gov/lmk/HistoricRainfallFloodingApril2-62025) |
| obs59 | rain | Bowling Green 4-day record | 2025-04-02/05 | NWS: record 4-day total of 10.04 in at Bowling Green (prior 9.67 in, May 2010). | high |  | [source](https://www.weather.gov/lmk/HistoricRainfallFloodingApril2-62025) |
| obs60 | rain | Kentucky Mesonet, Warren County 4-day total | 2025-04-02/05 | Daily News: 10.1 in over the four days. | medium (secondary report) |  | [source](https://bgdailynews.com/?p=685146) |
| obs61 | gauge | USGS 03314500 Barren River at Bowling Green | 2025-04-04T17:30-05:00 | Provisional 15-min data: peak stage 35.76 ft, 33,100 cfs at 17:30 CDT Apr 4 (22:30 UTC). NWS lists the river at moderate flood without a crest value. | high (USGS, provisional) |  | [source](https://waterservices.usgs.gov/nwis/iv/?format=json&sites=03314500&startDT=2025-04-01&endDT=2025-04-10&parameterCd=00060,00065) |
| obs62 | gauge | USGS 03314000 Drakes Creek near Alvaton | 2025-04-04T02:00-05:00 | Provisional 15-min data: peak stage 32.41 ft, 35,300 cfs at 02:00 CDT Apr 4 (07:00 UTC, inside the model window). | high (USGS, provisional) |  | [source](https://waterservices.usgs.gov/nwis/iv/?format=json&sites=03314000&startDT=2025-04-01&endDT=2025-04-10&parameterCd=00060,00065) |
| obs63 | damage_count | City of Bowling Green damaged structures | 2025-04-10 | City counted damage to 196 structures citywide across 10 zones (late Apr 10); 17 dwelling units evacuated or 100% substantially flooded; conservative estimate $5.1M residential + $1.3M commercial. No zone breakdown. Another outlet: about 200 structures, over $6M. | high for the number; definition of damaged and zones not given |  | [source](https://bgdailynews.com/2025/04/12/historic-floodwaters-force-families-from-homes) |

Other sources used: NWS Louisville event page <https://www.weather.gov/lmk/HistoricRainfallFloodingApril2-62025> (rain table; it gives no Barren River crest value),
the KYTC District 3 water-over-road bulletin <https://content.govdelivery.com/accounts/KYTC/bulletins/3da7209>, WNKY <https://www.wnky.com/?p=2378165>
(about 200 structures, over $6 million), IEM LSR service <https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py>. A gauge check on NOAA NWPS returned only recent data, so
USGS was used. Not found: individual addresses of flooded properties, FEMA declaration details (not retrieved), Basil Griffin Park coordinates.
