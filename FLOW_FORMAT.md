# Flow-grid format (v1) — the contract between the Python builders and the browser renderer

Everything that produces or consumes the fluid/flow visualisation follows this document exactly.
If something here is wrong or ambiguous, fix THIS file and say so; do not silently diverge.

## Why this exists
The flood solver saves only water **depth** (one raster per simulated hour). It does not save velocity.
The viewer shows the water as a moving fluid with flow vectors, so we derive a depth-averaged velocity
field from the simulated **water-surface slope** with Manning's equation (see "Velocity method"), and
pack depth + velocity into compact grids on a **Web Mercator** pixel grid that the browser can drape on the map.

Honest labelling is part of the contract: velocities are DERIVED (not solver output). meta.json says so
and the UI must say so.

## Locations
- APP root (the folder the local web server serves): `C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map`
- Python sources (git repo, source of truth for scripts): `C:\Users\Minec\Documents\BG_3D_Map\scripts`
- Output data: `<APP root>\layers\flow\<storm>\`   storm ∈ `5yr 10yr 25yr 50yr 100yr` (optionally `2yr`)
- Browser modules: `<APP root>\fluid\*.js`
- Source tier polygons (already downloaded): `<APP root>\BG FLOOD MAP VISTA SUPERCOMPUTER\v2\<storm>\v2_<storm>_depth_<SSSSSS>s.geojson`
  (EPSG:32616 UTM 16N, FeatureCollection of polygons, props `tier` ∈ shallow|moderate|deep|severe; hours 1..24 → seconds 003600..086400;
  an empty FeatureCollection (42 bytes) means a dry hour; the whole file is ONE line of JSON)
- Local terrain: `<APP root>\dem_bg.tif` (EPSG:32616, float32, 5.088 m cells, 4096×3500, nodata −9999). **Its elevations are in FEET** (mean 541 = Bowling Green at about 165 m; the file has no unit metadata)
  while its cells are metres. `face_velocity` needs metres (eta = z + h): read it with `flowgrid.read_dem_m(path)`, which multiplies by 0.3048 and turns nodata into NaN. Reading it as metres gives slopes 3.28x too steep and speeds about 1.8x too high.
- On Vista the model grid is 2 m, 10000×10000, EPSG:32616; terrain `$SCRATCH/bgflood/dem/dem_bg_2m.tif` (already in METRES, `read_dem_m` takes it unchanged), Manning `$SCRATCH/bgflood/dem/mannings_n.tif`,
  depth rasters `$SCRATCH/bgflood/runs/output_v2/<storm>/depth_<SSSSSS>s.tif` (float32 metres, nodata −9999, 25 files per storm incl. t≈0).

## Grid
- CRS EPSG:3857 (Web Mercator metres), north-up, square cells of `cell_m` mercator metres.
- Pixel (col,row), row 0 = NORTH edge. Pixel centre: `x = xmin + (col+0.5)*cell_m`, `y = ymax − (row+0.5)*cell_m`.
- `width = round((xmax−xmin)/cell_m)`, `height = round((ymax−ymin)/cell_m)`.
- All hours of one storm share one grid. The grid is cropped to the union of wet area over all hours plus a 150 m (mercator) margin,
  snapped to whole cells. It must stay ≤ 6000×6000 (≤ 36 M pixels); pick `cell_m` accordingly (default 5.0 for Vista/rasters, DEM-matched for tier preview).
- The bounds are snapped OUTWARD to multiples of `cell_m` in mercator metres (so storms with the same footprint share pixel alignment); `flowgrid.grid_from_wet_bounds` raises ValueError when a side would exceed 6000 cells.
- NOTE mercator metres are stretched by 1/cos(lat) ≈ 1.25 at Bowling Green; ground size of a cell = `cell_m * cos(lat)` with the latitude of the grid centre (`cell_ground_m` in meta.json, about 4.0 for 5 m).

## Files per storm  (`layers/flow/<storm>/`)
| file | content |
|---|---|
| `meta.json` | description below |
| `hNN.rgba.gz` | hour NN (01..24, zero-padded) — gzip (RFC 1952) of `width*height*4` raw bytes, row-major, RGBA |
| `flowpaths.rgba.gz` | same grid, summary over all hours (encoding below) |

Hours with nothing wet may be omitted; `meta.hours` lists the hour files that exist. A missing hour is dry (all zero). Hour 0 is never written (dry).

Writing: `flowgrid.write_rgba_gz` / `write_meta` write to a uniquely named temp file in the same folder (`<file>.<pid>.<random>.tmp`) and `os.replace` it onto the final name, so a reader sees the old file or the whole new one.
On Windows `os.replace` fails with PermissionError while a reader holds the target open without delete sharing (`serve.py` does while streaming) or while another writer replaces it, so it is retried (10 tries, 0.05 to 0.5 s apart, about 3 s in all) before the error is raised.
A failed write removes its temp file. Only a killed process can leave a `*.tmp` behind; `validate_storm_dir` reports any `*.tmp` in a storm folder as a problem.
`scripts/build_flow_from_tiers.py` goes one step further: it builds the whole storm in a sibling work folder (`layers/.flow.<storm>.<pid>-<id>.building`) and swaps it in as a folder only after `validate_storm_dir` passed (two renames, so for a few milliseconds `layers/flow/<storm>` does not exist and a request in that moment gets a 404). An interrupted or failed build leaves the previous storm folder as it was; work folders of killed builds older than 2 h are removed by the next build of that storm. Its tier and Manning caches (`<temp>/bgflow_tier_cache`) are written the same way and a damaged cache file is rebuilt.

## Pixel encoding (hNN.rgba.gz)
- **R = depth code.** `depth_m = (R/255)^2 * depth_max_m` with `depth_max_m = 8.0`. R=0 → dry (depth < ~0.0001 m); depths above 8 m clip to R=255.
  Cells with depth < `wet_threshold_m` (0.02) are written as 0.
  Encode: `R = round(255*sqrt(clamp(depth/8,0,1)))`.
- **G = u code**, **B = v code**: depth-averaged velocity of the water, **u = toward EAST, v = toward NORTH** (the direction the water MOVES, not where it comes from),
  both true-east/true-north (NOT UTM grid axes — the builder rotates by grid convergence). `u = (G−128)/127 * vmax_ms`, `vmax_ms = 4.0`; code 128 = 0. Clamp |u|,|v| ≤ vmax.
  Encode: `G = 128 + round(clamp(u/4,−1,1)*127)`. Dry cells: G=B=128.
  Image-space note: image y points DOWN, so a positive v moves toward smaller row numbers.
- **A = valid-domain flag**: 255 inside the model domain (including dry cells), 0 outside the model domain / nodata. (The viewer never draws water where A=0; a pixel with A=0 is written R=0, G=B=128.)
- Files are written with gzip mtime 0 and no stored file name, so identical content gives identical bytes.
- All four channels are linearly interpolable (the renderer relies on this); for dry cells G=B=128 so velocity fades smoothly to 0 toward dry cells.

## flowpaths.rgba.gz (summary over all hours, same grid)
- R = **maximum depth over all hours**, same depth encoding.
- G = **mean unit discharge** over the hours present: `q = mean_over_hours(depth*speed)` in m²/s; `G = round(255*clamp(sqrt(q/QMAX),0,1))`, `QMAX = 2.0`. (decode `q = (G/255)^2 * 2.0`)
- B = **hours wet** (count of hours with depth ≥ wet threshold) as `round(255*count/24)`; the divisor is always 24, even when fewer hours were written.
- A = valid flag as above. Pixels never wet at the threshold (R = 0) have G = B = 0.

## meta.json
```json
{
  "format": "bgflow",
  "version": 1,
  "storm": "5yr",
  "label": "5-year storm",
  "source": "rasters",
  "source_note": "depth rasters from solver v2 (2 m), velocity derived from water-surface slope",
  "velocity_method": "face-Manning from simulated water-surface slope (not solver output)",
  "crs": "EPSG:3857",
  "width": 4200, "height": 4100,
  "cell_m": 5.0,
  "cell_ground_m": 4.0,
  "bounds_merc": [xmin, ymin, xmax, ymax],
  "bounds_lnglat": [west, south, east, north],
  "depth_encoding": {"type": "sqrt", "max_m": 8.0},
  "vel_encoding": {"vmax_ms": 4.0, "zero_code": 128},
  "wet_threshold_m": 0.02,
  "qmax_m2s": 2.0,
  "hours": [3, 4, 5],
  "storm_depth_in": 4.2,
  "run_summary": { ... copy of the solver's run_summary.json when available ... },
  "hourly": [ {"hour": 3, "sim_seconds": 10800, "wet_cells": 123, "wet_area_km2": 0.0, "max_depth_m": 0.0, "mean_depth_m": 0.0,
               "max_speed_ms": 0.0, "volume_m3": 0.0, "clipped_cells": 0} ],
  "built": "ISO-8601 time written by the builder"
}
```
`source` ∈ `"rasters"` (full precision, built on Vista from the depth rasters) | `"tiers"` (preview built locally from the depth-TIER polygons: depth is quantised to
tier midpoints (shallow 0.2, moderate 0.45, deep 0.9, severe 1.5 m) then smoothed, so depth is approximate and velocities are indicative only).
`scripts/build_flow_from_tiers.py` smooths by solving for the smoothest water surface inside each tier's depth band (so ponds are flat, depth may leave the midpoint, the severe tier is open ended up to 6 m),
gives the dry cells beside the footprint the surface of their wet neighbours for the velocity only (a wet cell must not read its own depth as a drop into dry ground, which made every pond at rest jet from its rim; `tiers_build.params.halo_max_m`, never written),
treats a surface slope below 0.002 (`tiers_build.params.slope_dead_band`) as unmeasurable and shows that water at rest (speed scaled by `sqrt(max(1 - 0.002/S, 0))`, so ponds, the river and wide flat water read zero velocity: at hour 13 of the 100yr 28 % of the wet pixels move faster than 0.1 m/s, 78 % without the band),
limits the unit discharge `depth*speed` to `qmax_m2s`, and drops tier water outside the local DEM (the solver domain reaches about 1.3 km further west and 3.2 km further north: 17 % of the 100yr hour-13 polygon area).
Its meta.json carries one extra informational object `tiers_build` (method parameters incl. the size and mtime of `dem_bg.tif`, `buildings.geojson` and `roads.geojson`, per-hour `volume_ratio` = smoothed / midpoint volume, per-hour `sources` = size and mtime of the tier file the hour was built from, `dry_hours`); consumers ignore it. A rerun skips an hour only when all of these still match.
The UI must show the source badge: "Full-precision (solver depth rasters)" vs "Preview from depth tiers".

## Velocity method (rasters AND tiers use the same function, `flowgrid.face_velocity`)
Per cell with ground z, depth h, Manning n, spacing dx (ground metres), water surface η = z + h. For each of the four faces between a cell and its neighbour:
- `h_f = max(η_a, η_b) − max(z_a, z_b)`; if `h_f ≤ 0.005 m` → no flow through the face.
- `S = (η_b − η_a)/dx`, flow goes from higher surface to lower; `|S| < 1e-5` → 0.
- `speed_f = h_f^(2/3) * sqrt(|S|) / n_f` with `n_f` the mean of the two cells' n; limited to `min(vcap, sqrt(g*h_f))` (Froude ≤ 1), `vcap = vmax = 4 m/s`, `g = 9.81`.
- Cell velocity = depth-weighted mean of its face velocities per axis (`Σ h_f v_f / Σ h_f` over the two x-faces, likewise y); cells with `h < 0.01` get 0.
- Two refinements (found with analytic tests, implemented in `face_velocity`): (1) the per-axis `sqrt(|S|)` uses the cell's full surface gradient, i.e. each axis velocity is multiplied by
  `sqrt(|Sx|/|S|)` resp. `sqrt(|Sy|/|S|)` with `Sx, Sy` the depth-weighted mean face slopes and `|S| = hypot(Sx, Sy)`; without it a 45 degree slope gives `sqrt(2)*M(s)` instead of
  `M(s*sqrt(2))`, an 18.9 % overshoot, and the flow does not follow the gradient. (2) The final cell vector is limited to `min(vmax, sqrt(g*h))` with the cell's OWN depth `h`
  (the face depth `h_f` can exceed it beside deeper water, which would break the `speed <= sqrt(g*depth)` bound below).
- Invalid cells (NaN, ±inf or the raster nodata −9999 in h or z) are walls: no flow into or out of them, zero velocity. A non-finite or non-positive Manning n falls back to 0.035. Face arithmetic is float64
  (a float32 water surface adds up to 1.5e-5 m of rounding noise at 200 m elevation, which is a spurious slope above the 1e-5 threshold when dx = 1 m), the output is float32. `face_velocity_strips` (row strips with a one-cell halo) is bit-identical to `face_velocity`.
This is the quasi-steady balance of the local-inertial momentum equation the solver uses, with the inertia term neglected.
Rasters path: compute at the solver's native 2 m, then aggregate to the output cell with **depth-weighted** vector averaging (depth: plain mean; `block_reduce` then limits the block speed to `sqrt(g*depth_k)`, because a mean depth that includes dry sub-cells would otherwise break the speed bound), rotate (u,v) from UTM grid axes to true east/north by the
grid convergence angle γ(lon,lat) = atan(tan(lon − lon0)·sin(lat)), lon0 = −87°, then warp to the Mercator grid (bilinear).
Rotation: γ > 0 east of lon0 (grid north is turned clockwise from true north): `u_east = u·cosγ + v·sinγ`, `v_north = −u·sinγ + v·cosγ`. Verified against GDAL: γ(−86.4°, 36.97°) = 0.006298 rad, bearing of UTM grid north measured with
`rasterio.warp.transform` 0.006299 rad. Mercator north is true north, so no rotation is needed after the warp. Bilinear resampling keeps the speed bound (sqrt is concave, so an interpolated speed
cannot exceed sqrt(g * interpolated depth)); only the 8-bit rounding above remains.

## Units and sanity bounds a test can assert
- 0 ≤ depth ≤ 8 (clip), |u|,|v| ≤ 4, speed ≤ min(sqrt(9.81*depth), 4) before quantisation. On the 8-bit pixels this holds only up to the rounding of the codes:
  the true depth may be half a depth code deeper than decoded and each velocity code is off by up to half a step. A test on the files must therefore allow
  `speed ≤ min(sqrt(9.81*depth_hi), 4) + 4/127` with `depth_hi = ((R+0.5)/255)^2 * 8`; `flowgrid.validate_storm_dir` does exactly that. (Measured on real terrain: flow sitting at the cap decodes up to
  0.038 m/s above `sqrt(9.81*decoded depth)`, more than one velocity step, so the literal "+ one step" rule against the decoded depth would reject valid files.)
- Other file-level invariants checked by `validate_storm_dir`: A ∈ {0,255}; A = 0 ⇒ R = 0; R = 0 ⇒ G = B = 128; no depth code between 1 and 12 (below the 0.02 m threshold); A identical in every file of a storm;
  flowpaths: R = max of the hour R's (±1 code), B = hours wet over the hour files (±1 code), R = 0 ⇒ G = B = 0, G not above the largest hourly depth*speed
  (except at pixels whose hour R is 255: G is built from the true depth, which the clipped code does not hold, so those pixels may carry any G up to code 255).
  `validate_storm_dir` always returns its list of problems: a meta.json that is incomplete or has odd types or sizes (width/height must be integers in 1..6000, checked before any grid is allocated) is reported, never a traceback.
- Mass sanity (rasters): Σ(depth*cell_ground_area) per hour in meta.hourly.volume_m3 equals the raster's own volume within 1 % (quantisation + clipping reported in `clipped_cells`).
  `hourly` is computed by `flowgrid.hour_stats` on the final output grid before quantisation: wet = depth ≥ 0.02; `volume_m3` = Σ unclipped depth × `cell_ground_m`² over wet cells; `clipped_cells` = cells deeper than 8 m
  (their pixels hold 8 m, so the pixel volume is then lower than `volume_m3`); `mean_depth_m` over wet cells; `max_speed_ms` over wet cells.
- Flow direction sanity: on an inclined plane the velocity points down-slope; on a lake at rest velocity is 0; water spreading from a mound moves away from the mound.

## Serving (`<APP root>\serve.py`, the only supported local server)
- `*.gz` is sent as `application/gzip` with NO `Content-Encoding` header, so `fetch` returns the raw gzip bytes and `DecompressionStream('gzip')` is the one place they are decoded. A server that adds `Content-Encoding: gzip` would break the decoder (the browser would unpack the file itself).
- Range requests, `HEAD`, `ETag`/`Last-Modified` with `304` work. Every file, the flow grids included, is sent `Cache-Control: no-cache` (revalidated with the `ETag`, a `304` when unchanged): a rebuild rewrites the grids under the same names and `max-age` would keep the old ones in the browser. The loader's `?v=<meta.built>` on grid URLs is kept as a second guard; the server ignores the query string.
- `GET /__health` returns JSON with, per storm in `layers/flow`, `meta` (parses), `hours` (count in meta), `hour_files`, `flowpaths`, `bytes` and `complete`; `python selfcheck.py` checks the files themselves (gzip trailer ISIZE = `width*height*4`).

## JavaScript API every browser module must honour  (`<APP root>\fluid\`)
Plain ES modules (no bundler, no npm), loaded by `viewer.html` via `<script type="module">`; MapLibre GL JS 4.x is the host map (global `maplibregl`).
- `fluid-data.js`: `loadStorm(base, storm) → FlowStorm` (fetch meta.json; hour grids loaded on demand and cached, LRU of 4 decoded hours; decoding via `DecompressionStream('gzip')`),
  `FlowStorm.getHour(h) → Promise<{rgba: Uint8Array, hour}>` (h outside `meta.hours` → one shared all-ZERO-BYTES grid, tagged `rgba.isZeroGrid === true`, treat it as read-only; every consumer reads velocity as 0 wherever R = 0, so its G = B = 0 never means -4 m/s; the renderer uploads a 1x1 texture for it instead of the full grid),
  `decodeDepth(R)`, `decodeVel(G)`, `sample(rgbaA, rgbaB, mix, xMerc, yMerc, meta[, out]) → {depth, u, v, speed, valid}` (bilinear in code space, CPU; `meta` is the storm's meta object, so the method `FlowStorm.sample(rgbaA, rgbaB, mix, x, y[, out])` is the form callers normally use; either grid may be null; `valid` is false outside the grid and where interpolated A < 0.5), `lngLatToMerc/mercToLngLat`.
  Also exported: `FlowStorm.getFlowpaths() → Promise<{rgba}>`, `prefetch(hours)`, `dispose()`; `getHour`/`getFlowpaths` take `{signal}` for abort. A decoded size other than `width*height*4` throws `FlowFormatError`; a failed buffer allocation throws `GridMemoryError`.
  Every grid request is `<file>?v=<encodeURIComponent(meta.built)>` (no query when `meta.built` is not a non-empty string) and `meta.json` is fetched with `cache: 'no-cache'`, so a rebuilt storm is never paired with grids from the browser's or server's cache.
  `prefetch(hours)` warms at most `LRU_HOURS` (4) distinct hours of the list, in order (the cache holds no more, later ones would only evict earlier ones), with at most `PREFETCH_CONCURRENCY` (2) downloads in flight, because every grid in flight is a full `width*height*4` buffer (70-140 MB in production). It resolves to one settled result per hour it tried.
- `fluid-renderer.js`: `class FluidRenderer` — own WebGL2 canvas (`preserveDrawingBuffer: true`), draws a REGION of the grid in mercator space into the canvas: depth colour + animated flowing-water shading + particle streaks + arrow glyphs + flow-paths overlay;
  `setStorm(meta)`, `setHours(rgbaA, rgbaB, mix)`, `setFlowpaths(rgba)`, `setRegion([xmin,ymin,xmax,ymax], pxW, pxH)`, `setOptions({...})`, `frame(nowMs)`, `dispose()`.
  Details: `setHours`/`setFlowpaths` take a `Uint8Array` or the `{rgba}` object `getHour`/`getFlowpaths` return, and upload a texture only when the array identity changes (advance an hour as `setHours(B, C, 0)` and only C is uploaded); call `setStorm` first, then `setRegion`. `setRegion` sizes the canvas (premultiplied-alpha RGBA), clears the trails and keeps the particles, which live in Mercator metres. The canvas may come out SMALLER than requested: sides are limited to the GPU's texture/renderbuffer size, the area to 2^24 pixels (constructor option `maxCanvasPixels`; both sides shrink by the same factor and a warning is logged), and afterwards the size of the drawing buffer the browser really allocated wins. Callers that need the final size read `renderer.canvas.width/height` (the region always maps onto the whole canvas, so draping it by the region's corners stays correct). `setStorm` and `setRegion` may be called in either order after the first pair; the grid placement is recomputed by whichever comes last. `frame(nowMs)` advances time and the particle simulation and draws; `renderOnce()` only redraws. Also: `getOptions()`, `clearTrails()`, `pixelToWorld/worldToPixel`, `readPixels()` (top row first), `FluidRenderer.supported()`, `renderer.gpu`, `renderer.stats`.
  Options and defaults: showDepth/showFlow/showParticles/showArrows true, showFlowpaths false, opacity 1 (water layer only), flowSpeed 8, particleCount 60000 (max 200000), particleSpeed 40, particleLife 6 s, trailFade 0.04, arrowSpacingPx 40, sunAzimuthDeg 225, sunElevationDeg 55, minDepthShown 0.02, plus flowCycleSec 3, rippleStrength 1, rippleScalePx 16, particleWidthPx 1.6, particleAlpha 0.9, particleBudgetMs 6, particleCoverage 0.6, quality 2 (0..2), arrowScale 1, flowpathsOpacity 0.9, viewPitchDeg 0, viewBearingDeg 0, seed 1. Velocities are scaled by 1/cos(lat) so streaks and the ripple drift cover the true ground distance on the Mercator canvas. When one pass over all particles would exceed `particleBudgetMs` the pass is split over 2..4 frames; if even 4 slices do not fit, the active particle count is reduced to what the measured per-particle cost allows (`stats.particleCap`).
  `particleCount` is a maximum: unless `particleCoverage` is 0, the number of ACTIVE particles (`renderer.stats.particleCap`, re-estimated about every 20 frames and on every `setRegion`/`setHours`/`setOptions`) is capped so that the trail ink in the densest moving water stays near `particleCoverage`; without the cap 60000 particles turn a zoomed-in channel into a white haze.
- `fluid-stats.js`: building/road/grid flood statistics from a decoded grid (`buildBuildingIndex`, `evaluateBuildings`, `buildRoadIndex`, `evaluateRoads`, `summarizeGrid`, `tierOfDepth`); footprints use an all-touched pixel rule, flooded means depth >= 0.1 m. Header comment has the details.
- `fluid-layer.js`: `class FluidLayer(map, opts)` - MapLibre integration (renderer canvas as a `canvas` source on a raster layer draped on the terrain, continuous hour animation with temporal interpolation, quality governor, probe, flood statistics). The header comment of the file has the full API and event list. In short:
  `FluidLayer.supported()`; `await setStorm(key, {force})` (rejects with `StormNotFoundError` for a missing storm, `FlowFormatError`/`GridMemoryError`, or `AbortError` when superseded; a failed request leaves the previous storm and its picture running, status back to ready; with no previous storm the layer stays clean, status `error`); `setTime(h)`, `play(msPerHour)`, `pause()`, `togglePlay()`, `setOptions({...})` (renderer options plus `freeze, autoQuality, frameBudgetMs, msPerHour, pixelScale, maxPixels, loopStart, governorOverMs, governorRecoverMs, statsTopN, statsMinDepthM, qualityMode`), `setVisible`, `updateRegion`, `dispose()`; properties `storm, meta, source, status, time, targetTime, timeRange, qualityLevel, qualityMode, qualityInfo, region, canvas, renderer, hasPicture`; events `status, time, hour, storm, region, quality, error, stats`. After `dispose()` every method is a no-op (`probe` is invalid, statistics are null, `attach*` resolve 0, `refreshStats` resolves null) except `setStorm`, which rejects.
  Errors: an hour grid that fails to load keeps the last good picture, emits `error` {phase: 'grid', hour} and sets status `error` while playback waits at that hour (retry after 5 s); after two failures playback skips the hour. Scrubbing to a loadable hour clears the error.
  Quality: levels 0..7 of `QUALITY_LEVELS` (canvas pixel scale first, then particle count and shader quality). `setQualityMode('auto'|'high'|'medium'|'low')` (levels 0 / 3 / 6; a pinned mode turns the governor off), `setQualityLevel(n)`. The governor works on the frame interval: above `frameBudgetMs` for `governorOverMs` (half of it when above twice the budget) it steps down 1-3 levels; below 0.75 of the budget for `governorRecoverMs` times a flap back-off it steps up one level.
  `probe(lngLat, out?)` returns `{valid, depthM, speedMs, u (east), v (north), bearingDeg (heading the water moves TOWARD, 0 = north, clockwise; 0 when still), cardinal ('NE'; '' when still), moving, hour}` at the displayed continuous time with the same blend as `FlowStorm.sample`; `attachProbe(cb)` returns a detach function and calls `cb(result, event)` on mousemove with one reused result object and `cb(null, null)` on mouseout.
  Statistics: `attachBuildings(fc)` / `attachRoads(fc)` (FeatureCollection or feature array, null detaches) build the `fluid-stats` indices for the grid of the current storm in slices of about 25 ms, cached per grid geometry; they resolve to the feature count. Event `stats` fires when the WHOLE hour changes (hour = floor of the displayed time), once a storm has drawn, and when an index becomes ready or `statsMinDepthM` / `statsTopN` change. Payload: `{hour, storm, floodedCount, tierCounts, belowTierCount, buildingCount, ranked [{index, depthM, tier, speedMs}] (first statsTopN), order (Int32Array, deepest first), depthM, speedMs, roads {pctFlooded, floodedLengthM, totalLengthM, floodedFeatures, perFeaturePct} | null, grid {wetCells, wetAreaKm2, maxDepthM, meanDepthM, maxSpeedMs, volumeM3}, wetAreaKm2, maxDepthM, maxSpeedMs, computeMs, cpuMs}`. Nothing is computed without a `stats` listener or attached data; a playback faster than the computation skips hours rather than queueing them. `getBuildingFlood()`, `getRoadFlood()`, `getFloodStats()` return the latest values (null until computed, cleared on a storm change), `refreshStats()` computes now, `buildingLngLat(index, out?)` gives the centroid for fly-to, `getIndexInfo()` the index build cost.
  Lifecycle: `dispose()` removes layer, source, listeners, timers and GL resources (it also frees the texture and buffers that MapLibre 4.7 leaves behind for a removed canvas source). A map `webglcontextlost` stops the loop and `webglcontextrestored` replaces the canvas source (MapLibre 4.7.1 itself leaves the base map black after a restore until new tiles are requested). The renderer's own context loss is handled by the renderer and the layer redraws when it is restored. A hidden page stops the loop and the clock and keeps the playing flag.
