# Address index (`layers/address_index.json.gz`, `layers/county/address_index.json.gz`)

Built by `scripts/build_address_index.py` (rules and statistics = the apps', see `scripts/flowquery.py`). One gzip JSON file, columnar, sorted by
(street, house number). City: 30,148 records, 0.44 MB gz; county: 29,754 records, 0.53 MB gz. If a file would exceed 8 MB gz the builder writes
a manifest at the same path (`"sharded": true`, street table, `shards[]`) plus `address/<first char of street>.json.gz` shards with the record columns.

## Header
`format "bgaddr"`, `version 1`, `domain`, `built`, `n` (records), `origin [west, south]` + `scale 100000` (lng = origin[0] + x/scale), `bbox`,
`use_legend`, `storms[]` (key, label, source `rasters|tiers`, depth_in, hours, grid, records_with_water, records_flooded), `tiers`
(edges 0.1/0.3/0.6/1.2 m), `columns` (text description of every array), `rules`, `stats`.

## Record arrays (all length `n`, same index)
`x`, `y` (1e-5 degree ints), `num` (house number text), `use` (index into `use_legend`), `val` (assessed USD, 0 unknown), `parcel` (PVA id),
`bid` (city: feature index in `layers/buildings.geojson`; county: OSM way id; -1 lot), `kind` (0 building/address, 1 lot), `nb` (footprints merged).
Street of record i: `streets.key[s]` / `streets.disp[s]` with `streets.start[s] <= i < streets.start[s+1]` (binary search `start`). Address text =
`num + " " + streets.disp[s]`; the search key = `num + " " + streets.key[s]` (upper case, abbreviated).

## Flood data
`flood["25yr"]` = flat int array, stride 5: `[index delta, depth cm, tier 0-4, first hour flooded (0 never), hours flooded]`; walk it summing the deltas to get
record indices. A record missing from a storm is dry. Depth = max over the footprint pixels (all-touched), flooded = depth >= 0.1 m, hours counted
over the hourly grids (hour N = end of hour N), tier 0 below 0.1 m / 1 shallow / 2 moderate / 3 deep / 4 severe (>= 1.2 m). Lots (`kind 1`) use the pixel at the lot point.

## Normalisation and typeahead (reference: `normalize_street`, `search` in the builder)
Upper case, `. '` removed, other punctuation -> space, words abbreviated: STREET ST, ROAD RD, DRIVE DR, AVENUE AVE, BOULEVARD BLVD, LANE LN, COURT CT, CIRCLE CIR,
PARKWAY PKWY, HIGHWAY HWY, TERRACE TER, PLACE PL, TRAIL TRL, NORTH N, SOUTH S, EAST E, WEST W ... (`ABBREV`). Query: split a leading number (`^\d+[A-Z]?\s+`), abbreviate the
completed words of the rest (leave the word still being typed alone, also try the fully abbreviated form), binary-search `streets.key` for the prefix, then keep records of those
streets whose `num` starts with the typed number. No hit and 3+ letters: every street where each typed word starts a word (linear over ~2k streets). A number alone: linear scan of `num`.
The name index is therefore the street table itself (no extra structure). Keep clients in sync by loading this document's list, or copy `ABBREV`.
