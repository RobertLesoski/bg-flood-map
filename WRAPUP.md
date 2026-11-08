# Wrap-up resume checklist (desktop-integration)

- [x] 1. Handoff files committed (2b9a1ac)
- [x] 2. Merged accuracy-2, features, perf2 (PERF.md conflict: kept both sections), packaging (a34adfb)
- [~] 3. Suites. scripts: 286 passed. desktop: deferred start-up test failed in sequence = GPU OOM
      ('Not enough memory left', flow-array) from Apps never shut down -> fixed: App.shutdown idempotent + frees
      layers/terrain; conftest autouse module fixture shuts down every App created. Then 516 passed, 1 timing flake
      (test_stats speed 5.03 ms vs 5 limit; 3.7 ms alone) - diagnosing threads alive.
- [x] 4. UX: badge shortens/hides (menus.py), timeline tick step adapts (panels_time.py), layer checkboxes one column
      when labels do not fit (panels_side.py). Storms 10-12 keys: skipped (0/-/= conflict with zoom; no clean binding).
- [ ] 5. Perf runs -> desktop/docs/PERF.md
- [ ] 6. Real-window check
- [ ] 7. Package build + SHIP_NOTES.md + zip
- [ ] 8. HANDOFF.md 2a/2b update
