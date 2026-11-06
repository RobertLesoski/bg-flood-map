# Packaging resume notes
- branch viewer-packaging based on desktop-integration 80934a1
- DONE: build (165 s, 500 MB), offscreen --render county --ui OK, --selftest county OK (45.9 fps), docs, portable route
- fixes: conda Library\bin DLLs (ffi) + sys.path for collect_submodules("bgv") in bgv.spec
- portable .bat not trial-run (would create a conda env)
