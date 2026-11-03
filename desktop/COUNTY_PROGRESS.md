# County domain - progress notes (append only)

## Note 1 (about 55% of the work)
Done (code written, imports OK, city render still OK):
- `bgv/data/osmpbf.py` pure-python/numpy PBF reader (real Kentucky file: 50 s once, 24,543 buildings + 26,632 roads in the Warren box;
  OSM simply has few buildings outside Bowling Green: 20,140 inside the city grid, 4,403 elsewhere), `bgv/data/countyscene.py` (cache).
- `bgv/domain.py` (Domain, resolve, DomainInfo, pick_storm), terrain from any GeoTIFF (`data/terrain.py` TerrainSource + warp_dem_lazy),
  flow `subdir` support, buildings/roads/prep `geojson=` overrides, SceneData(domain), Prefetch(domain), App.domain / set_domain().
- UI: Domain menu + "Area" combo in Storm & Time + DEMO DATA badges; CLI --domain --county-dem --county-demo --county-pbf --make-county-demo.
- `bgv/data/demo.py` synthetic demo (full demo written to the scratchpad `county/demo_full`, takes ~4 min).
- Loading-screen chooser: `bgv/ui/chooser.py` + App.step_init gating + Launcher wiring (NOT yet run in a window).
- Tests written: `tests/test_county.py` (+ `tests/pbf_writer.py`); NOT yet run.
In progress / next:
1. Run `python -m pytest desktop/tests/test_county.py -q -p no:cacheprovider` and fix.
2. County render with demo data: `python -m bgv --render out.png --domain county --county-demo <demo> --storm 100yr --ui`.
3. GPU tests (test_county_gpu.py: domain switch VRAM baseline, chooser flow), benchmark county + city, selftest, full suite.
4. Docs: README, DESIGN.md (Domains), PERF.md, COUNTY_RUN.md 'Afterwards'.
Finding: the user's data folder ALREADY contains `layers/flow_county/<12 storms>` (copied 13:33-13:47) but not yet `layers/dem_warren_5m.tif`
-> county reports "terrain is missing" (clear message). Pass `--county-dem` or copy the DEM.
How to test: see above; full suite `python -m pytest desktop/tests -q -p no:cacheprovider` (~4 min).

## Note 2: DONE. All code, 40 county tests pass, real county (flow+DEM+OSM) renders, benchmarks, selftest rc=0, docs written. Remaining: full suite result (scratchpad county/suite.log).
