# viewer-perf2 resume notes
- Branch reset onto desktop-integration (80934a1). c4e7177: compare grid float conversion 1 alloc; tests/test_perf_alloc.py (2 pass).
- Shared byte LRU already exists (cog.TILE_CACHE); compare._maps already single-entry; other LRUs bounded.
- Baseline matrix running: scratchpad bench.py (city/county x base/util/tiledepth x3, 8 s, 1080p) -> bench_before.json; summ.py summarizes.
- Tools (probe/timeline/swipe/tour/timelapse/facilities) have no --benchmark flag: not yet measured.
- TODO: summarize, re-run after fixes, startup timing (python -m bgv.startup), PERF.md, full suite once.
