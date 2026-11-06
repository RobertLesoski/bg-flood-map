# bgv performance log

Hardware: Intel Core i5-10300H (4 cores / 8 threads), NVIDIA GTX 1650 4 GB (Vulkan, wgpu 0.32), Intel UHD iGPU (unused),
8 GB RAM, Windows 11, Python 3.14.  1920x1080, MSAA 4x, all layers on, no UI in the benchmark (UI cost: `python -m
bgv.ui.uibench`).

Method: `python -m bgv --benchmark --seconds 12 --size 1920x1080 --json` (scripted orbit / dive / street orbit / pull back,
60 fps steps, every frame synced to the GPU: "synced frame" = CPU recording + GPU pass in series; "pipelined" CPU time and
the free-running fps come from a second loop without the per-frame sync).  Three runs per row, MEDIAN reported.  Layer
breakdowns: small timer wrappers around every layer method (CPU) and disabling one layer at a time with the GPU timestamp
query (GPU).  Start-up: `python -m bgv.startup` (the windowed start-up sequence on an offscreen canvas) and one real
`--selftest` window.

CAVEAT: a second agent ran tests / renders on the same PC while this was measured (and the laptop's cores are slower after
every idle wait, so CPU times in a synced loop are ~1.7x the pipelined ones).  Means are good to ~10 %; p99 / max still
contain 10 - 30 ms interference stalls that also occur with the app idle.  Re-measure on a quiet machine before trusting
max values.

## Results (ms unless stated)

| | baseline (start of this work) | now (median of 3) | target |
|---|---|---|---|
| synced frame mean | 14.8 | 9.7 | |
| synced p95 / p99 / max | 19.0 / 26.8 / 51.9 | 15.6 / 16.7 / 21.5 | p99 <= 20, max <= 30: met |
| CPU command recording, mean, synced loop | 6.3 (p99 12, max 34) | 3.4 (p99 6.2, max 13.6) | <= 3: NOT met in the synced loop |
| CPU command recording, mean, pipelined | - | 1.6 | <= 3: met |
| GPU pass mean | 6.7 | 4.8 | <= 6: met |
| free-running fps | 134 | 189 | |
| storm switch: water back / worst frame / p99 during load | 183 ms / 36.1 / 26.8 | 262 ms / 20.1 / 17.9 (worst of the 3 runs 24.7) | no frame > 25: met |
| start-up, process start to window with loading screen | 6.5 s (to first frame) | 0.54 s (real window, GDI splash) | <= 1.5 s: met |
| start-up, to fully loaded (headless sequence / real window) | 6.5 s | 2.1 s / 3.4 s | |

Intermediate steps (synced mean, p99, CPU synced, GPU): baseline 13.3 / 20.9 / 5.2 / 6.7 (re-measured) -> numba culling,
shared frustum, uniform caching, adaptive shoreline, shadow early-out, nogil statistics 9.7 / 18.0 / 3.3 / 4.9 -> pooled
decode buffers (storm-switch worst frame 36 -> 20 ms) -> final above.

## What was done (profile first)

CPU recording was 6.3 ms, of which: terrain chunk culling 1.5, building / road tile culling + parameter packing 0.8, frame
uniform build 0.3, arrows / particles parameter arrays 0.45, ~90 wgpu calls x ~14 us (wgpu-py proxy overhead) 1.3, rest layer
draws.  Fixes:
- `bgv/gpu/fastcull.py`: numba kernels for the terrain chunk LOD / frustum / snap-word / sorted instance lists and for the
  tile boxes + merged draw ranges (numpy twins kept as fallback and as the test reference).  Terrain cull 1.5 -> ~0.3 ms.
- `bgv/gpu/cull.py` + `app.frustum()`: frustum planes computed once per frame, shared by terrain / roads / buildings.
- Caching: terrain update skipped when camera / params / wet table are unchanged and instance buffers only re-uploaded when
  their bytes change; building uniform block only rebuilt when selection / sliders change; water params and the frame
  uniform's view-projection inverse and hour->layer table cached; one list assignment for the 32 uniform scalars.
- statistics kernels: `nogil=True` (they held the GIL for ms at a time), numba threads limited to 3, decode / scene /
  stats threads at BELOW_NORMAL priority (`bgv/gpu/bgthread.py`).
- GPU: sub-texel shoreline reconstruction only closer than 8 m/px (cross-fade over 6..8; a flow cell is < 0.6 px beyond it):
  overview GPU 4.4 -> 2.7 ms.  Terrain shadow march: exact early exits (fully shadowed / ray above every height) and the
  detail noise only where it is visible (< 2.5 km).  Governor knobs travel in two spare words of the frame uniform.
- Storm switch hitches: decode into 3 reused 40 MB buffers (`gunzip_into`) instead of a fresh 40 MB allocation per hour
  (page-fault stalls of 10 - 30 ms hit unrelated wgpu calls on the render thread); upload goes through a single-layer
  staging texture + one GPU copy per hour and a per-frame time budget (2 ms / 1 MB chunks).
- Start-up: GPU adapter without enumerating every backend (-0.8 s); cffi type parsing 50x faster (`bgv/gpu/cffiwarm.py`
  patches cffi so a type string parses against only its own typedefs, 2.6 s -> 0.06 s, benefits every process incl. tests);
  `bgv/launch.py`: window + GDI splash first, GPU init and storm / terrain preparation in worker threads
  (`bgv/startup.py`), then `App(defer=True)` builds terrain / flow textures / each layer in steps of <= 10 ms behind an imgui
  loading screen (`App.step_init`).  Buildings / roads / statistics caches already built in a worker thread with a progress
  bar (scene_data); terrain preparation now also runs in a worker.
- `bgv/perf.py`: gc.freeze + gc.disable during frames with a young collection every 2 s; automatic quality governor.

## Auto quality governor (default mode, "Auto" button in Quality)

4 levels (High / Medium / Low / Minimum): shadow march steps 26/20/14/8, shoreline range 8/5/3/0 m/px, particle + arrow
presets, terrain LOD x1/.85/.7/.55, render scale 1/1/.85/.7 (scene into an intermediate texture, bilinear blit, interface
stays sharp).  Down after the mean frame time exceeds 1.25 x 16.7 ms for 0.6 s; up one level after 4 s with GPU time < 45 %
of the target; never within 3 s of a change.  Picking a preset button switches to manual (governor stands down, native
resolution).  MSAA is fixed at launch (pipelines are built for it): the adaptive replacement is the render scale; MSAA 4x
-> 1x would only save 0.8 - 1.8 ms GPU anyway.  At 1080p on this GPU it stays at High (GPU 4.8 ms).

## Not done / findings

- Render bundles: not implemented on purpose.  The visible chunk / tile lists change almost every frame while the camera
  moves, so a bundle would be rebuilt as often as it is replayed; when the view is still, culling is cached and the frame
  is GPU-bound.  Remaining recording cost is mostly wgpu-py call overhead (~14 us per call, ~90 calls).
- CPU recording in the synced loop is 3.4 ms (target 3.0); it is 1.6 ms pipelined.  Cold-core effect, see caveat.
- The imgui interface costs ~4.5 - 5 ms CPU per frame (uibench) - more than the whole scene now; its p95/p99 with the UI on
  is 19 - 25 ms because of its half-rate "lite" switching.  That is the UI's code, not changed here.
- Overview shoreline: pixels at > 8 m/px differ slightly from the old render (max diff 107 in 8.5k px of 2 M, mean 0.24 /
  255; compare `docs/shots` style renders: visually identical at 3x zoom); 3 of the 4 reference shots are bit-identical
  (max diff <= 4) or within 28 on 25 px.

## County domain (real flow grids + real DEM + OSM, 1920x1080, same method)
county: synced mean 7.9 ms (127 fps), p99 17.7, GPU 4.0 ms, free-running 230 fps, VRAM ~1130 MB. City measured in the same session: 8.7 ms (115 fps),
p99 19.5, GPU 4.9 ms, 1253 MB (city before this work, noisier machine: 14.0 ms) - city unchanged. First county start: PBF read 50 s once, DEM warp + slope map
once (cached), then cached. OSM has 24.6k buildings in the county; not stress-tested at 100k.

## Native solver tiles + visual polish (2026-11-05, same method: `--benchmark --seconds 12 --size 1920x1080`, median of 3, other agents were using the PC)
| | city | county |
|---|---|---|
| synced frame mean / p99 / max | 9.62 / 32.3 / 55.7 ms | 9.44 / 27.6 / 55.7 ms |
| fps synced / free-running | 104 / 176 | 106 / 182 |
| GPU pass mean | 5.0 ms (was 4.9) | 4.19 ms (was 4.0) |
| CPU recording, mean | 2.6 ms | 3.4 ms |
| VRAM estimate | 1253 MB | 1130 MB |
Unchanged within noise: the sun-aware sky, water shoulder and window facades cost about 0.1 - 0.2 ms GPU; tile reads happen in the probe worker thread, never on a frame
(a cached tile read is microseconds; the footprint index is built once in a worker and cached). County p99 / max are dominated by interference from other processes
on the machine (the previous county log had p99 17.7 on a quiet PC): re-measure on a quiet machine before trusting them.
Experimental solver-depth overlay (`tiledepth` layer, NOT registered in the shipped build - measured with the registry line added by hand, 1024 x 1024-cell clipmap window around the camera target, two textures, worker-thread reads):
city synced mean 9.73 ms, GPU +0.5 ms (5.49 vs 5.0); county 13.4 ms synced mean (about 75 fps, 174 fps free-running) but p99 46 ms / max 92 ms (earlier runs with a 2048 window:
p99 50 - 105 ms, max 300 - 620 ms) while the scripted benchmark dives and the window is re-read. It meets 60 fps on average but is not hitch-free: it stays an opt-in experiment.

## Integrated viewer (desktop-integration: tile reader + cinematic + analysis + search) - 2026-11-05

`python -m bgv --benchmark --domain <d> --seconds 12`, GTX 1650, 1920x1080, MSAA 4, scripted camera path, offscreen. The PC was NOT idle
(other Python processes and agents were running), so treat these as lower bounds:

| Domain | synced fps | mean ms | p95 | p99 | GPU pass mean | CPU record |
|---|---|---|---|---|---|---|
| county | 92.8 | 10.78 | 14.60 | 20.18 | 4.20 ms | 2.88 ms |
| city | 82.7 | 12.09 | 19.60 | 27.27 | 4.98 ms | 3.46 ms |

County is above the 60 fps requirement. The benchmark path draws the scene plus the standard interface; it does not open the optional
tool windows (analysis window, facilities list ...). Screenshots of every tool in the merged app: `docs/shots/integration/`.

## Utilities layer (2026-11-05)

`BGV_UTILITIES=1 python -m bgv --benchmark --domain county --seconds 12 --size 1920x1080`: layer on (366 markers, 232 lines, imgui draw list):
65.6 and 79.4 fps synced mean (p99 50 / 26 ms).  The same command with the layer off measured 44.5 fps in between: the PC was loaded
(other processes), so the layer's cost is below the run-to-run noise there.  Offscreen UI harness with image read-back (orbit, 1080p, county):
overview 18.9 ms off / 22.2 ms on (about +3 ms from the line polylines), close view 14.9 / 14.6 ms (67 fps).  Not a clean >= 60 fps
proof for the county overview under load: marked as measured-but-noisy.

## viewer-perf2 measurements (2026-11-05, `--benchmark --seconds 8 --size 1920x1080`, 3 runs, median [min-max])
Machine was heavily loaded (other agents running pytest / features_perf.py); spreads below show it. VRAM is the app's estimate.

| domain/config | synced fps | p99 ms | GPU ms | VRAM MB | max ms |
|---|---|---|---|---|---|
| city base | 108.3 [106.6-111.1] | 27.0 [25.0-28.2] | 4.9 | 1253 | 32 [31-89] |
| city utilities | 89.3 [77.9-117.6] | 29.6 | 6.4 | 1253 | 43 |
| city tiledepth | 112.6 [34.8-114.6] | 20.2 [19.7-112] | 5.3 | 1257 | 25 [24-212] |
| county base | 70.8 [53.4-88.4] | 66.5 [64.9-72.7] | 5.5 | 1130 | 170 |
| county utilities | 109.4 [55.2-113.7] | 36.4 [19-138] | 4.5 | 1130 | 71 |
| county tiledepth | 36.5 [28.4-111.7] | 72.3 [22-175] | 6.5 | 1134 | 244 |

Tools (probe, timeline, swipe, tour, time-lapse, facilities) have no benchmark switch and were not measured.
Regression tests that count bytes/allocations instead of time: `tests/test_perf_alloc.py`.
