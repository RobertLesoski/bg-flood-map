# Validation report, ev4 flood model, 2026-11-12

This is a new dated report. The earlier `REPORT_validation_2026-11-11.md` stays as the record of that day, including sections 12 and 13 added on 11-12. Every number here traces to a script and an input file, listed in the table notes or in `verification/derived/run_manifest_*.json`.

## 1. Bottom line

The ev4 model is **unvalidated** for flood depth and for flood extent. Its rainfall input and storm timing are supported by data. The one comparison that could give an accuracy number (model max depth against the satellite water map) is blocked: the ev4 depth raster exists only on Vista, and I could not reach Vista with a key. No accuracy percentage is given.

## 2. Status table

| Claim the model makes | Evidence | Status | Limitation |
|---|---|---|---|
| Rain input: 4.97 in in 24 h for April 2025 (ev4) | Airport hourly gauge, reproduced from `scripts/storms_out/bwg_hourly.csv` | Validated against the airport gauge | NWS totals use other windows and stations (10.48 in at the airport over four days) |
| Storm timing: floods occurred on the storm dates | Groundsource news events (Zenodo 18647054) | Partially validated | News-derived; Groundsource reports 60% accuracy in place and time |
| Max depth at observed points (Cave Mill Road, Aviation Heritage Park, etc.) | Observed depth table, 13 rows (`verification/observed_depths.csv`) | **Unvalidated.** Model depths not available locally | Observed values are anecdotal or official estimates, not surveyed |
| Flood extent matches satellite water (OPERA, 2025-04-06 23:40 UTC) | OPERA WTR tiles, 25.2 km² open water in the county (`scripts/validate_opera_ev4.py`) | **Unvalidated.** Comparison blocked on the ev4 raster | Radar misses canopy and urban water; one overpass; the model covers a 24 h window |
| Roads named as flooded are wetter in the model (AUC 0.84, 21 roads) | `docs/VALIDATION_EV4.md` (Nov 5 study) | **Reported only, not reproduced.** Inputs missing | No bootstrap or permutation possible; the study itself says the model is not validated |
| Barren River gauge stage rose to 35.76 ft (Apr 4) | NWS bwgk2 page (via search) | Stage only | Not a depth; no datum or ground elevation to convert |
| Sensitivity: which inputs change the result | Design only (`verification/SENSITIVITY_DESIGN.md`) | **Not done** (blocked) | Needs a solver run on Vista |

## 3. Results

### 3a. Road validation (Phase 1): blocked
- The inputs for the Nov 5 study (`layers/validation/ev4_observations.json`, `layers/cog/county/ev4/max_depth.tif`, `layers/roads_flooded/county/ev4.json`) are absent from the repo, its worktrees, and git history.
- Bootstrap CI, permutation test, and control-sample sensitivity: **not computed.** Doing so from the reported summary alone would require inventing per-road values, so I did not.
- To unblock: the per-road table (41 observed items, 100 controls, depth per road) and the ev4 depth raster.

### 3b. Model vs. OPERA (Phase 2)
- Method (`scripts/validate_opera_ev4.py`): OPERA cells inside the county polygon (1,574,624 cells; tiles deduplicated in the overlap). The model is sampled at each cell centre after reprojection. Scores at thresholds 0.05, 0.10, 0.20, 0.30, 0.50, and 1.00 m: hit rate, false-alarm ratio, and CSI. Two observed definitions: A = open water (class 1); B = open water plus inundated vegetation (classes 1 and 3).
- Observed water in the county (real data):
  - Open water (class 1): 28,004 cells = **25.2 km²**, 1.82% of the 1,535,733 valid county cells.
  - Inundated vegetation (class 3): 612 cells = **0.55 km²**, 0.04% of valid cells. Definition B adds only this much, so the canopy question moves the observed area by well under 0.1 percentage point.
  - Layover/shadow masked (251): 830 cells, 0.75 km², excluded.
  - Fill (255): 38,061 cells, excluded.
  - County area from cell count: 1,417 km² (the earlier polygon estimate was 1,419 km²).
- Model comparison: **blocked.** Once the raster exists: `python scripts/run_validation.py --model <path to max_depth.tif>`.
- Synthetic end-to-end test on the real tiles (`scripts/tests/test_validate_opera_ev4.py`, class `SyntheticOverlapOnRealTiles`): a model whose wet area is a known UTM rectangle recovers the exact hit count (7,446) and miss count (20,558). False alarms are 554,077 against an independent 554,505, a 0.08% gap from 50 m grid-edge sampling.

### 3c. Observed depths (Phase 3)
- `verification/observed_depths.csv`: 13 rows. Quality ratings: 8 anecdotal, 2 estimated (official verbal estimates), 1 surveyed gauge stage (not depth), 2 road-closure presence records.
- No surveyed high-water marks for Warren County were found. The USGS STN database (stn.wim.usgs.gov) returned web pages rather than data for every API URL tried, so the database was not searched in full.
- FEMA damage assessment (FEMA-4864-DR): text extracted with `pdftotext -layout`. It is a statewide summary with no Warren County depth data, and Warren is not among the individual-assistance counties.
- Stage-to-depth conversion: **not done.** It needs a documented datum and ground elevation at gauge 03314500, which were not found.
- Model depth at each point: **not computed** (no model raster). With 13 rows, 8 of them anecdotal, a skill score would not be meaningful, so none is computed.

### 3d. Sensitivity (Phase 4): blocked
- Design written in `verification/SENSITIVITY_DESIGN.md`: rainfall (1.0x and 2.0x), DEM vertical offset (±0.5 m), Manning n (0.75x and 1.25x), and rolling 24 h versus calendar-day window. Outputs: wet area, depths at observed points, OPERA CSI.
- Not run: the solver and its inputs are on Vista. To unblock: the bgflood solver configuration plus `dem/` and `runs/storm_ev4.rain` on Vista.

## 4. Software changes (this session)

| Change | Reason | File |
|---|---|---|
| New OPERA comparison script: CSI, hit rate, and false-alarm ratio at six thresholds, two observed definitions | Phase 2 requirement | `scripts/validate_opera_ev4.py` (new) |
| Exact centre-in-polygon mask replaces PIL polygon fill | PIL fill includes boundary pixels, a half-cell bias along the county edge; caught in testing before any result was used | same file |
| Removed an unused variable in the contingency count | Cleanup | same file |
| Unit tests with hand-computed values (scores, projection known points, round trip, sampling, units), plus a real-tile synthetic overlap test | Phase 5 requirement | `scripts/tests/test_validate_opera_ev4.py` (new) |
| Single entry point with SHA-256 input hashes, library versions, a fixed seed, and blocked-step records | Phase 5 requirement | `scripts/run_validation.py` (new) |
| Observed-depth table with quality ratings | Phase 3 requirement | `verification/observed_depths.csv` (new) |
| Progress log and work plan | Process | `verification/PLAN_2026-11-12.md`, `verification/progress.log` |

Test results: all 12 unit tests pass (`python -m unittest scripts/tests/test_validate_opera_ev4.py`), including the real-tile synthetic test. The entry point's manifest is in `verification/derived/`; its unit-test step passed, and the other three steps are recorded as blocked.

No model physics or core simulation code was changed. Nothing was committed. Changes from 11-11 remain uncommitted.

## 5. Numba crash (Phase 5)

- Evidence: `scratchpad/pytest_full.txt` (default run) shows `Windows fatal exception: code 0xc0000139` inside numba's `_launch_threads` → `select_known_backend`. That run still reports 515 passed, 10 skipped. `scratchpad/pytest_full2.txt` (with `NUMBA_THREADING_LAYER=workqueue`) reports 515 passed, 10 skipped, with no fatal dump.
- Environment: Anaconda numba 0.65.1, threading priority `['tbb', 'omp', 'workqueue']`. Anaconda's `Library/bin` contains no `tbb12.dll`, only `mkl_tbb_thread.2.dll`. Status 0xc0000139 is a missing DLL entry point, consistent with the tbb backend failing to load.
- Classification: **a workaround, not a root-cause fix.** `workqueue` avoids the tbb backend, but the root cause (the TBB library) is not fixed. Making `NUMBA_THREADING_LAYER=workqueue` the viewer default is a reasonable workaround. That change was not made here; it is your decision.

## 6. Still blocked, and what unblocks each item

| Item | Unblocked by |
|---|---|
| Model vs. OPERA CSI, hit rate, and false-alarm ratio | The ev4 `max_depth.tif` (Vista: `/scratch/11931/rlesoski/bgflood/runs/county_v2/ev4/max_depth.tif`, 542 MB), or a crop of it |
| Model depth at observed points | The same raster, plus the points table |
| Road-validation bootstrap and permutation | The per-road table from the Nov 5 study, plus the ev4 raster |
| Sensitivity sweep | Vista access (bgflood solver, dem, rain file) |
| USGS high-water marks | A working STN query, or a manual download of the Kentucky April 2025 event from the STN web interface |
| Stage-to-depth conversion at the gauge | A documented datum and a ground-elevation survey at USGS 03314500 |

## 7. Physics issues noticed

None recorded in this session. Two model-design limits from the Nov 5 study still apply: no upstream river inflow, and a 24 h rain window against a multi-day event.

## 8. Sources

- OPERA WTR tiles T16SEF and T16SEG, 2025-04-06 23:40 UTC, processed with `scripts/validate_opera_ev4.py`; class codes from the OPERA product spec D-108761 Rev A.
- Observed depths: Bowling Green Daily News, 2025-04-12 (Brentwood Place; Cave Mill Road); Spectrum News, 2025-04-07 (Public Works estimates); WKU Talisman (Cave Mill Road); Bowling Green Daily News (Aviation Heritage Park).
- Road closures: KYTC Water Over the Road bulletins at content.govdelivery.com/accounts/KYTC/bulletins/3da9dfa, 3db2115, and 3dbaa1d.
- FEMA-4864-DR Preliminary Damage Assessment (fema.gov, PDAReport_FEMA4864DR-KY.pdf).
- Barren River gauge 03314500 crest: NWS bwgk2 (search result).
