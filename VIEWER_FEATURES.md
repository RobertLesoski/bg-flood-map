# viewer-features resume notes
- base: desktop-integration 80934a1; all work committed
- modules: ui/keymap.py, ui/ext_features.py (all UI), ui/sharecard.py, ui/cvd.py, data/routing.py, data/exposure.py, data/duration.py, data/session.py, gpu/layers/hoursflooded.py (+1 registry line)
- tests: tests/test_features.py (16), tests/test_features_gpu.py (5); perf harness docs/features_perf.py; PERF.md section added
- known shared-UI overlaps NOT fixed (shared files): 1366x768@200%: menu 'Help' under the source badge, timeline y-axis labels overlap, sidebar layer checkboxes overlap
- skipped: none fully; tiles thumbnails compare uses city in test
