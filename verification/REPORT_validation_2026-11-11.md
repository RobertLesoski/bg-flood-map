# Validation report: Bowling Green / Warren County flood model

Date: 2026-11-11. Scope: check the repo's storm facts and model inputs against public data, and state exactly what is and is not validated.

## 1. Summary

| Check | Result |
|---|---|
| Storm rainfall totals (hourly data) | Verified: reproduced from the airport hourly gauge. |
| Repo wording of storm records | Corrected (Harvey "daily record"). |
| NWS published rainfall figures | Checked; match or are explained by windows and station. |
| Flood timing (Groundsource news data) | Supported for all four storms. |
| June 2022 rainfall | Total matches NWS; "top 20 since 1996" not found. |
| USGS stage, April 2025 | Downloaded; rise matches the April 2025 flood. |
| Satellite flood extent (OPERA / Sentinel-1) | Not validated. Sentinel-1 window downloaded but not calibrated or georeferenced. OPERA needs your login. |
| Model flood depths | **Not validated.** No observed depth data in the repo or sources checked. Ev4 rasters are on Vista (down until tonight). |

## 2. Rainfall totals (airport hourly gauge, BWG)

Source: `scripts/storms_out/bwg_hourly.csv` (Iowa Environmental Mesonet, BWG ASOS). Each report was assigned to its clock hour (the largest kept), then summed over 24 hours from the window start.

| Storm | Window start (UTC) | Repo total | Recomputed | Missing hours | Status |
|---|---|---|---|---|---|
| Apr 2025 (ev4) | 2025-04-03 13:00 | 4.97 in | 4.97 in | 0 | Reproduced |
| Harvey remnants (ev5) | 2017-09-01 02:00 | 6.03 in | 6.03 in | 0 | Reproduced |
| May 2010 (ev6) | 2010-05-01 18:00 | 7.41 in | 7.41 in (interpolated, 2 h) / 7.18 in (zero-filled) | 2 | Reproduced with interpolation |

Notes:
- The earlier 7.18 in figure came from an unfloored timestamp error. It is corrected here.
- The ev5 and ev6 totals are 24-hour rolling windows, not calendar days.

## 3. NWS published figures

Sources: weather.gov/lmk pages (read 2026-11-11).

- **April 1–7, 2025:** Bowling Green Airport 10.48 in; WKU 13.11 in; COOP 11.88 in; CoCoRaHS 11.20–11.85 in. New 4-day record 10.04 in. No 24-hour figure is published, so the repo's 24-hour 4.97 in cannot be checked against NWS.
- **May 1–2, 2010:** airport 9.67 in for two days (record 2-day total). NWS daily values from the May records (4.75 in May 1, 4.92 in May 2) came from a search snippet, not the page, so they are unconfirmed.
- **Sept 1, 2017 (Harvey):** NWS calendar-day record at Bowling Green is 4.93 in. The September total is 6.47 in.
- **June 2022:** NWS lists 3.39 in on June 6 (Records section). Matches the repo's ev3 total. The NWS June 2022 summary does not state a "top 20 since 1996" ranking, so that claim is **unverified**.

## 4. Flood timing: Groundsource (news-derived flood events)

Source: Zenodo record 18647054, `groundsource_2026.parquet` (667,122,400 bytes; DOI 10.5281/zenodo.18647054). Filtered to a Warren County bounding box (-86.75 to -85.95 W, 36.75 to 37.15 N): 246 events in the box.

| Storm | Events on or near storm dates | Largest event |
|---|---|---|
| May 1–2, 2010 | 4 | ~1,300 km² |
| Sept 1, 2017 (Harvey) | 6 | ~1,444 km² |
| June 6–7, 2022 | 2 | 0.28 km² |
| April 2025 | 75 (Mar 30 – Apr 10) | 52 km² |

Caveat: Groundsource is built from news reports. Google reports 60% of events are accurate in both place and time. This supports that floods were reported near these dates; it does not measure depth.

## 5. USGS gauge: Barren River at Bowling Green (03323500)

Source: USGS NWIS daily values, gage height (ft), downloaded to `verification/usgs/03323500_dv_2025-04.json`.

- Apr 3: 9.13; Apr 4: 9.16; Apr 5: 9.33 (before the event)
- **Apr 6: 15.45**; Apr 7: 16.55; Apr 8: 16.55
- Peak daily mean in the file: 16.94 ft on Apr 9

The stage rises sharply on April 6, consistent with the flood. Discharge (00060) returned no data for this period, and the instantaneous-values request returned an empty file.

## 6. Satellite flood extent

- **OPERA DSWx-S1 (NASA):** two tiles for April 6, 2025 (T16SEF, T16SEG) are identified. Download requires an Earthdata login. The built-in browser pane blocked the archive domain, so the files have **not** been downloaded. Not validated.
- **Sentinel-1 GRD (public):** scene S1A_IW_GRDH_1SDV_20250406T233956 (23:40 UTC, April 6, 2025). Only the Warren County window was read from the public S3 object: `verification/s1/S1_20250406_23z_vv_county_window.tif` and `..._vh_county_window.tif`. The data is uncalibrated and not georeferenced, so the ~32% dark-pixel fraction is **not a usable result**.

## 7. Model outputs

- The April 2025 (ev4) depth rasters exist only on Vista. Vista is down until tonight. **No model-to-observation comparison has been run.**
- Depths are therefore **unvalidated**. Nothing in the sources checked gives observed flood depths for Warren County.

## 8. Changes made

- `layers/storm_info.json`: Harvey period text corrected (rolling 24 h; NWS calendar-day record 4.93 in). May 2010 note added with the reproduced 7.41 in and the zero-fill figure.
- `desktop/bgv/ui/road_names.py`: road labels clipped to the scene rectangle (not under the sidebar). Compiles; not yet re-rendered.
- `verification/` files created: S1 county windows, USGS JSON, this report.

## 9. Test status

- Full desktop test suite: **515 passed, 10 skipped** (run 1, 6 min 44 s). The log contains one `Windows fatal exception 0xc0000139` dump from numba's threading backend during collection.
- Same suite with `NUMBA_THREADING_LAYER=workqueue`: **515 passed, 10 skipped**, no fatal dump (6 min 35 s).
- `desktop/tests/test_roads.py` passes alone.
- Not yet decided: whether to set `workqueue` as the viewer's default. This needs a speed check.

## 10. Open items

1. **Model vs observation (ev4):** compare the model's April 2025 depths with the OPERA or calibrated Sentinel-1 extent. Blocked on Vista (back tonight) and on your OPERA download.
2. **June 2022 "top 20 since 1996":** unverified; no source found.
3. **Storm records and station choice:** decide whether storms are labeled by airport 24-hour windows (current) or NWS calendar days.
4. **Viewer numba crash:** choose a permanent `NUMBA_THREADING_LAYER` setting.
5. **Commit:** the storm text, road label, and verification files are uncommitted.
6. **Security (yours):** the planning-office API key and the password in `TACC LOGIN.txt`.

## 11. Sources

- NWS Louisville: weather.gov/lmk/HistoricRainfallFloodingApril2-62025; weather.gov/lmk/flood_050210_how_it_happened; weather.gov/lmk/september_2017_summary; weather.gov/lmk/june22_summary
- Groundsource: Zenodo 18647054; research.google/blog (Groundsource)
- USGS NWIS: 03323500, Barren River at Bowling Green
- Iowa Environmental Mesonet: BWG hourly ASOS (`scripts/storms_out/bwg_hourly.csv`)
- NASA OPERA: earthdata.nasa.gov (Helene article); PO.DAAC archive (login required)
- Sentinel-1 GRD: AWS Open Data `sentinel-s1-l1c`; Earth Search catalog
- Kentucky River flood replay: manmeet3591.github.io/kentucky-river-flood-replay (partly readable)

## 12. OPERA satellite water extent, April 6, 2025 (added after the downloads)

Source: NASA OPERA DSWx-S1 WTR layer, tiles T16SEF and T16SEG, acquisition 2025-04-06 23:40 UTC (product 2025-04-07 10:20 UTC). Files: `verification/opera/`. CRS EPSG:32616, 30 m. Class codes checked against the OPERA product spec (D-108761 Rev A, WTR table): 1 = open water, 3 = inundated vegetation, 251 = layover/shadow masked, 255 = fill.

Masked to the Warren County boundary (FIPS 21227, from the repo's boundary file). Overlap between the two tiles counted once.

| Quantity | Value |
|---|---|
| County area (boundary polygon) | 1,419 km² |
| County cells covered by a tile | 97.6% |
| Open water (class 1) | 25.2 km² (1.8% of covered county area) |
| Inundated vegetation (class 3) | ~0.5 km² (not counted as open water) |
| Layover/shadow masked (251) | ~0.8 km² (not counted as water) |

What this does and does not show:
- It is the satellite's view of standing open water, about 20 hours after the rain window. Radar misses water under trees and in urban areas, so it is a lower bound on the flood.
- Model comparison: not done. The April 2025 model depths are on Vista. The local copies of the inputs for the November 5 study (`layers/cog/county/ev4`, `layers/validation`) are also missing from this computer, so that study cannot be rerun here.
- The existing study (`docs/VALIDATION_EV4.md`, dated 2026-11-05) reports a weak signal on named roads (AUC 0.84 on 21 roads) and says the model is not validated. Its caveats apply to any comparison: 24 h rain window vs. 4+ days of rain in the real event, no upstream river inflow, and presence/absence only (no observed depths).

## 13. Observed flood depths search (April 2025)

Searched for official observed depths for Bowling Green and Warren County. Result: no official depth or high-water-mark dataset for the city was found. Everything below is secondary (news or local-official quotes) unless noted.

| Observation | Value | Source type | Usable for depth check? |
|---|---|---|---|
| Public Works director, Andy Souza (quoted by Spectrum News, Apr 7 2025) | "about seven feet" in one area; Cave Mill "about 10 feet" | local official, quoted in media | Yes, as approximate local depths, with no location or time stamp beyond Apr 7 |
| Aviation Heritage Park | 10 ft on Apr 7 | media report | Approximate, single location |
| Cave Mill Road | up to 10 ft (resident quote, WKU Talisman); several feet around homes Apr 8 (BG Daily News) | media | Approximate |
| Home interior (BG Daily News, Apr 12) | 4 inches | media, one family | Single point |
| Lost River Cave | "60 feet" | media | No: implausible for a cave-entrance flood, likely a different measure |
| Barren River gauge (USGS 03314500, via NWS bwgk2 page) | crest 35.76 ft on Apr 4 2025 | gauge stage (not depth) | Not a depth; the NWS gauge page itself did not load the values |
| NWS April 1–6 summary | "Moderate Flooding" for Barren River at Bowling Green; hydrograph is an image with no values | NWS | No numbers in text |
| FEMA PDA (FEMA-4864-DR) | not readable in the fetched PDF | FEMA | Not checked |

What this means:
- The highest-quality observations are the Public Works estimates (about 7 ft and 10 ft), which are the only official-sounding depths. They are approximate and not surveyed high-water marks.
- No survey of depths across the county exists in the sources reached, so a full depth-by-depth comparison is not possible. The model can be checked against only a handful of points (Aviation Heritage Park, Cave Mill Road), and the locations are not precise enough for a precise test.
- The USGS high-water-mark database (stn.wim.usgs.gov) was not searched directly; it is the most likely source of surveyed marks and should be checked next.
- The gauge crest is a stage, not depth, and the model has no river stage output, so it cannot be compared with 35.76 ft.
