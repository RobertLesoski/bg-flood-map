"""Analysis tools (measure, profile, point / building series, flood timeline, storm compare) plugged into the canonical registry.

The tools themselves live in `bgv/analysis/*` and `bgv/ui/panels_analysis.py` (unchanged); this module is the only wiring
(see docs/ANALYSIS_HOOKS.md).  The shader-side hooks (app.compare_hook / app.compare_wet) stay in gpu/layers/terrain.py.
"""
from __future__ import annotations

from . import panels_analysis, registry

registry.register_top_menu("Analysis", panels_analysis.menu)
registry.register_panel("analysis", panels_analysis.draw, order=40)                  # tools window + map overlays
registry.register_overlay("analysis view rect", panels_analysis.draw_hidden)         # keeps the view rectangle right while the UI is hidden
registry.register_key_hook("analysis keys", lambda ui, ev: panels_analysis.on_key(ui, ev.get("key", "")))
registry.register_click_hook("analysis click", panels_analysis.handle_click)
