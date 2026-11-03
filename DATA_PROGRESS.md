# DATA_PROGRESS (data/tools agent)

## Note 1 (start)
- Done: surveyed repo + data. Facts: all_parcels.geojson covers only the city/DEM area (lng -86.54..-86.29, lat 36.89..37.06), NOT the whole county;
  desktop osmpbf reader exists (desktop/bgv/data/osmpbf.py) and is reused read-only; no desktop OSM cache yet.
- In progress: scripts/flowquery.py + scripts/geodata.py (shared libs), then scripts/build_address_index.py (city).
- Next (priority order): 1 address index city, 2 infrastructure + roads_flooded, 3 vista_make_cogs.sh + COUNTY_RUN.md, 4 county layers + county address index, 5 fetch_hifld / fetch_svi / vista_climate_scenarios.
- Commands: none yet.

## Note 2
- DONE: scripts/flowquery.py, scripts/geodata.py, scripts/build_address_index.py. City index built:
  <data>/layers/address_index.json.gz = 30,148 records (11,329 buildings + 18,819 lots), 0.44 MB gz, 23 s. Parity with desktop stats.py verified
  (all 20,284 footprints: identical pixel index, depth codes and tiers on 100yr flowpaths).
- IN PROGRESS: scripts/osm_pbf.py + scripts/build_infrastructure.py (infrastructure.geojson + roads_flooded).
- NEXT: vista_make_cogs.sh/make_cogs.py + COUNTY_RUN.md, county layers + county address index (needs osm_pbf.load_county_buildings), fetch_hifld/fetch_svi/vista_climate_scenarios, ADDRESS_INDEX.md, tests (scripts/tests/test_*.py).
- Commands: python scripts/build_address_index.py --domain city   (county: --domain county, after osm_pbf.py exists)

## Note 3
- DONE: osm_pbf.py (reuses desktop osmpbf + desktop cache; OSM county scene = 24,620 buildings, 18,349 roads), build_infrastructure.py run for city+county
  (layers/infrastructure.geojson 51 facilities, layers/county/infrastructure.geojson 98; layers/roads_flooded/<domain>/<storm>.json + index.json).
- IN PROGRESS: build_county_layers.py, county address index run.  NEXT: make_cogs.py + vista_make_cogs.sh + COUNTY_RUN.md, fetch_hifld/svi, vista_climate_scenarios, ADDRESS_INDEX.md, tests.
- Commands: python scripts/build_infrastructure.py --domain both ; python scripts/build_address_index.py --domain county

## Note 4 (all items written)
- DONE: build_county_layers.py (layers/county/buildings.geojson.gz 24,620 bldgs 1.6 MB gz, roads.geojson.gz 18,349 ways 1.2 MB gz), county address index (29,754 rec, 0.53 MB),
  make_cogs.py, vista_make_cogs.sh, vista_climate_scenarios.sh, fetch_hifld.py, fetch_svi.py, ADDRESS_INDEX.md, COUNTY_RUN.md addendum, tests/test_data_builders.py (11 pass).
- Remaining: final full pytest, report.
