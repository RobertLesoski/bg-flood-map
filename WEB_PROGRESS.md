# Web viewer progress (agent: web/PWA). Canonical folder: C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map ; repo copy synced at the end.

## Note 1 (14:10)
DONE: backups of originals in session scratchpad/orig; fluid/fluid-renderer.js now takes a `palette` option (4 colours -> uniforms uPal0..3, default = old blue ramp);
fluid/fluid-layer.js `setStorm(key, {base})` + `layer.base` (per-domain flow folder, same key in another base reloads); icons/*.png generated (PIL).
IN PROGRESS: manifest.webmanifest + sw.js, then viewer.html: start screen (City/County), bottom sheet, palette, scrubber, swipe.
NEXT order: start screen -> bottom sheet + PWA -> cividis palette -> scrubber + ev events -> swipe compare -> selfcheck additions -> sync to repo.
TEST: python serve.py --no-browser --port 8123 (canonical folder); open http://127.0.0.1:8123/viewer.html?r=N ; harnesses fluid/test/*.html.

## Note 2
DONE: manifest.webmanifest, sw.js (syntax ok; SW cannot register in the embedded pane: environment refuses any SW script), icons/, start screen City/County (verified: county + city fluid render), domain probing, county lazy buildings/roads loader (files absent -> water only), bottom-sheet CSS+JS, scrubber + events list code, phone defaults.
IN PROGRESS: mobile layout check at 375x812 (pane resized to mobile; reset with preset desktop).
NEXT: cividis palette (renderer `palette` option already there), selfcheck additive rows, harness tests, swipe (not started), sync to repo copy.
TEST: http://127.0.0.1:8123/viewer.html?domain=county&nosw=1

## Note 3 (final)
DONE and synced to repo: start screen, county/city domains, bottom sheet, PWA files, cividis palette + text size, scrubber + events, selfcheck rows.
NOT DONE: before/after swipe (P1-8). NOT RUN: fluid/test harnesses (pane hidden), contrast script, SW in a real browser (embedded pane refuses SW registration).

## Note 4
DONE: pytest scripts/tests = 236 passed. Address search typeahead (city + county index), risk card (tier ranges in ft, modelled depth, template summary, honest 'not validated' note, Share link ?addr=N, Print/PDF via print CSS), infrastructure overlay toggle (white ring = flooded). Verified in pane (city).
NOT DONE: swipe compare, natural-language query box, flooded-road flags, fluid/test harnesses (pane hidden; run layer-harness.html with the pane visible), contrast script.

## Note 5
DONE: swipe compare (hour/storm/none, flat maps, divider), flooded-roads list (+county colouring by osm_id), rule-based query box, contrast fixes, county search verified. Harness: stats 40/41 (fixtures missing), layer 150/156 (6 terrain/zoom-18 alignment fails, cause not diagnosed). Server 8123 stopped.

## Note 6
Layer harness 6 fails PRE-EXIST (same with original fluid js; 177/184 at last poll). At-risk roads by max-depth tier (toggle rf-cb, city+county, joined by OSM id to layers/county/roads.geojson.gz, legend, popup). Swipe not seen visually (pane only 539x279). Servers 8123/8124 stopped.

## Note 7 (real-depth COG)
DONE: cog/cog-depth.js reader, click-to-query anywhere, optional real-depth overlay (own palette/tiers, peak or hour), risk card per-building solver depth, setBeforeId in fluid-layer.js, sw.js shell v2. Tests: scripts/tests/test_cog_reader.py (10), cog/test/cog-harness.html (30+). Details/log: WEB_COG_PROGRESS.md.

## Note 8 (storm picker explains each storm)
DONE: scripts/build_storm_blurbs.py -> layers/storm_blurbs.json (numbers from storms_out/*.json + build_storm_info.DESIGN; wording and public-report context hand-written, flagged 'not verified by this project'); scripts/tests/test_storm_blurbs.py (5). viewer.html: short line + tooltip/aria-label on each of the 12 options, a storm card under the picker (name, short line, expandable 'What is this storm?' with long text, rain, Real-world context + links, model run for city/county, honesty line). sw.js unchanged (layers/*.json already network-first cached).
VERIFIED (pane): county, all 12 cards render with right text/links/run dates; 375 px no horizontal overflow; no console errors. NOT VERIFIED: link clicks opened, city domain visually, offline, cog harness html, docs/VALIDATION_EV4.md does not exist yet (referenced as 'when available').
