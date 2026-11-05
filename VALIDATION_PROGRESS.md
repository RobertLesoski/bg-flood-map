# Validation progress (ev4 vs April 2025 flood)

Done 2026-11-05:
- Observations collected: layers/validation/ev4_observations.json (63 items) + raw/ (IEM LSR CSV, USGS 03314500 and 03314000 JSON).
- scripts/validate_ev4.py (run: python scripts/validate_ev4.py) -> layers/validation/ev4_scores.json, docs/VALIDATION_EV4_map.png.
- scripts/tests/test_validate_ev4.py: 10 tests pass.
- docs/VALIDATION_EV4.md written (tables partly generated from the scores JSON).
Open / not done: FEMA declaration docs not retrieved; Basil Griffin Park and Buchanon Park not located; city damaged-address list needed for building comparison;
full April 2-6 rain + upstream inflow run would be the proper test. Note: docs/VALIDATION_EV4.md was assembled by a scratch generator; edit it directly now.
