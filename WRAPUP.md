# Wrap-up resume checklist (desktop-integration)

- [x] 1. Handoff files committed (2b9a1ac)
- [x] 2. Merged accuracy-2, features, perf2 (PERF.md conflict: kept both sections), packaging (a34adfb)
- [x] 3. Suites: scripts 286 passed; desktop 518 passed (after fixes below)
      - GPU OOM in sequence (deferred start-up test): App.shutdown idempotent + frees layers/terrain; conftest autouse
        module fixture shuts down every App a module created (41bbfa2)
- [x] SEARCH BUG (1870b07): typeahead dropdown window took imgui focus on appearing -> address box inactive after the
      first character -> keys went to hotkeys. Fix: no_focus_on_appearing on cards + panels. Test:
      test_interaction::test_address_box_keeps_the_focus_when_the_typeahead_dropdown_appears (fails without fix).
      Real window: `python docs/realwin_search.py --out x.png` (OS SendInput; guarded; window-only PrintWindow capture): PASS.
- [x] 4. UX: badge shortens/hides, timeline tick step adapts, layer checkboxes one column when labels do not fit.
      Storms 10-12 keys skipped (no conflict-free binding: - is zoom, 0 alone covers one storm).
- [ ] 5. Perf runs -> desktop/docs/PERF.md (scratchpad bench.py)
- [ ] 6. Real-window check (domain switch, UI scale)
- [ ] 7. Package build + SHIP_NOTES.md + zip
- [ ] 8. HANDOFF.md 2a/2b update
