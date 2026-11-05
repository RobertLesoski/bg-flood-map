# Viewer fix: tiledepth overlay (branch viewer-fix-overlay2, based on desktop-integration ca72db2)

## Done
- ROOT CAUSE of imgui "Missing PopID": TileDepthLayer set `self.label = ""` / `ts.label`, shadowing `Layer.label()`.
  `_sec_layers` called `l.label()` between push_id/pop_id -> TypeError -> PopID skipped -> assert at End().
  Fix: renamed to `tiles_label`; `_sec_layers` checkbox now pops in try/finally.
- Registered `tiledepth` (default off); View menu toggle "Solver depth overlay on the map".
- New CLI `--enable-layers NAMES` for benchmarks.
- County bench, overlay on (1080p, 12 s, PC busy): last frame-stat block printed before fps_free_running:
  p95 14.9 / p99 17.6 / max 19.9 ms; free-running 170 fps; storm-switch p99 48 ms. Section not fully verified.

## Resume checklist
1. Re-run test_ui_gpu + test_interaction (fix not yet test-verified).
2. Check full bench JSON; consider caching tile->f16 conversion.
3. Offscreen render alignment check (`--render --domain county --enable-layers tiledepth`), look at PNG.
4. Tests: no registered layer shadows label(); menu toggle.
5. Flaky tests (deferred start-up layer names; window-close timing): NOT started.
6. Update docs/TILES.md + PERF.md "NOT registered" notes. Full suite once at end.
