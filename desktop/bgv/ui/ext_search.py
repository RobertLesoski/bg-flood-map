"""Search / facilities / roads-at-risk / risk report / help, plugged into the canonical registry (docs/SEARCH_HOOKS.md).

The features live in `bgv/search/*`, `bgv/ui/panels_search.py` and `bgv/ui/help.py` (unchanged).  Direct edits that remain in shared files:
`ui/dialogs.py` (the searchable Help window replaces the two dialogs) and `gpu/layers/overlay.py` (search worker shutdown, queued report).
"""
from __future__ import annotations

from imgui_bundle import imgui

from . import panels_search, registry


def _find_menu(ui):
    if imgui.begin_menu("Find"):
        try:
            panels_search.menu(ui)
        finally:
            imgui.end_menu()


registry.register_top_menu("Find", _find_menu)
registry.register_panel("search", panels_search.draw, order=30)          # markers, search bar, cards, facilities / roads windows, hints
registry.register_key_hook("search keys", panels_search.on_key)          # Ctrl+F, /, M, L, Ctrl+P, Esc
registry.register_click_hook("search click", panels_search.on_click)
