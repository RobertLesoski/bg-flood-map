# Solver depth tiles (COGs) in the desktop viewer

The viewer shows two kinds of depth. The **flow grid** (`layers/flow*/<storm>`, 20-25 m cells, sqrt-coded, drives the 3D water, the
statistics and the at-risk list) and, when present, the **solver tiles**: Cloud-Optimized GeoTIFFs of the solver output itself
(`layers\cog\county\<storm>\{max_depth.tif, depth_<seconds>s.tif, cog_meta.json}`; EPSG:3857, uint16 centimetres, 6.25 m mercator cells
= about 5 m on the ground, 65535 = outside the model). Nothing existing changes when tiles are present: they add labelled lines.

Label shown with every tile value until the tile agent's accuracy verification is recorded as done (`WEB_COG_PROGRESS.md`):
**"solver depth (tiles, 6.25 m, verification pending)"**. It becomes "solver depth (tiles, 6.25 m)" when `settings["tiles_verified"]` is true
or the storm's `cog_meta.json` contains `"verified": true`, and "solver depth (native tiles, N m)" when the folder's `res_m` is 5 or finer.

## What it does
* **Cursor probe**: "Solver depth" (the hour file nearest the displayed hour, `hour N` = water at the end of simulated hour N) and "peak" (max_depth.tif),
  in cm and ft. "no water (0 cm)" (the model says dry) is different from "no data (outside the modelled area)". The old flow-grid line stays,
  relabelled "Grid depth".
* **Selected building**: "Solver depth" / "Solver peak" = the highest tile cell under the footprint (all-touched cells, the same rule the flow-grid
  statistics use), next to the unchanged "Grid depth". Until the footprint index exists (first use per domain: a few seconds, then cached in
  `desktop/cache/cogindex_*.npz`) the card says so and keeps the grid value.
* **Sidebar "Solver depth tiles"**: switch (also View menu), what serves this storm, and an optional comparison table: flooded buildings by the
  flow grid vs by the tiles (>= 0.1 m on the footprint maximum; tiers 0.1 / 0.3 / 0.6 / 1.2 m), overlap (both / grid only / tiles only),
  max depth, buildings without a tile value. It counts the buildings that have both a grid cell and a tile value. The headline statistics and
  the at-risk list are never replaced by tile numbers.
* **City view**: there are no city COGs yet; the city view reads the COUNTY run of the same storm name (the county run covers the city), always
  labelled "county solver run". Storm names that exist only in one domain simply have no tile lines.
* **API** for other features: `app.depth_at / building_depth / depth_profile / tile_info` (`docs/EXTENSION_API.md`).
* Switching to finer tiles later: set `settings["cog_folder"]` (e.g. `"cog_native"`, a folder under `layers\`) or env `BGV_COG_FOLDER`; or an
  absolute folder with `settings["cog_dir"]` / `BGV_COG_DIR`. No code change.

## Reader
`bgv/data/cog.py`: pure python + numpy + zlib, classic and BigTIFF, deflate with predictor 2, overviews, sparse tiles, one process-wide LRU of decoded
tiles (256 MB default, `settings["tile_cache_mb"]`), vectorised point sampling (a 200-point profile touches a handful of tiles).

## Verification (executed 2026-11-04, real data, `python desktop/tests/tile_verify.py`)
250 random lng/lat points per storm (one third drawn from wet cells), lng/lat to mercator by PROJ (rasterio.warp), compared with rasterio reads of the
same files, peak and hour 12:

| storm | points | wet (peak / hour 12) | mismatches vs rasterio (peak + hour 12) |
|---|---|---|---|
| 2yr | 250 | 98 / 68 | 0 |
| 100yr | 250 | 118 / 84 | 0 |
| ev4 | 250 | 124 / 51 | 0 |

(1500 reads, every value identical.) This verifies the reader, not the solver output or the resampling done when the COGs were made: that is the tile
agent's accuracy check (pending). `tests/test_tiles.py` additionally checks every overview level against GDAL on the fixture and the footprint maximum
against rasterio `all_touched`.

Per-building footprint maximum (`python desktop/tests/tile_verify_buildings.py`, 300 random real county buildings of 24,620, 100yr): peak 299 / 300 identical to rasterio
`all_touched` (the one difference: 2 vs 3 cm), hour 12 298 / 300 identical (differences of 1 and 2 cm). The few differences are cells that the outline only grazes (edge rule
in floating point), the same kind of sliver case documented for the flow-grid index (`tests/rasterio_crosscheck.py`).

## Tile vs flow-grid depth (same points, county flow grid = 5 m solver depth block-averaged to 20 m)
Points where both are wet (>= 0.1 m): mean absolute difference 0.09 to 0.25 m; the grid is lower than the tile at the median (ratio grid / tile 0.64 to 0.92);
largest single difference 1.56 m (100yr, a cell next to a steep bank). Wet in one source and dry in the other: 2 to 12 of 250 points per storm and
hour (the tile sees narrow water the 20 m grid averages away, and the grid smears water over dry neighbours). 100yr hour 12 at the building level (county):
grid 1,903 flooded buildings, tiles 1,698; 1,152 flooded in both, 751 grid only, 546 tiles only.
These differences are expected (cell size and the footprint maximum) and are not errors of either source.

## Experimental overlay (NOT registered: unfinished)
`bgv/gpu/layers/tiledepth.py` is written and was rendered once (devshot `--enable tiledepth`, see below) but is NOT registered in `gpu/layers/registry.py`: adding its line makes the sidebar trip an imgui
ID-stack assert ("Missing PopID", root cause not found) and the county benchmark showed hitches. To try it, add `"bgv.gpu.layers.tiledepth:TileDepthLayer"` after the water line of the registry.
What it does: the tiles draped on the terrain in tier colours (>= 0.1 / 0.3 / 0.6 / 1.2 m), hour or peak. A 1024 x 1024-cell window
around the camera target (level 0 = 6.25 m cells only when closer than about 1.7 km, otherwise the 12.5 m ... overviews), read in a worker thread and uploaded in 256 KB slices into
the idle of two textures. Rendered and visually checked (ponds and channels line up with the flow-grid water); NOT hitch-free: see `PERF.md` (county benchmark p99 46 ms / max 92 ms
with it on). Hour mode draws only terrain chunks that the flow grid considers wet, so water the 20 m grid misses entirely is not drawn there; peak mode draws all visible chunks.
