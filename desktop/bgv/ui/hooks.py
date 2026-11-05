"""Compatibility shim: the cinematic branch's hook API, now implemented by the canonical `bgv.ui.registry`.

New code should call `registry.*` directly (docs/EXTENSION_API.md).  Kept so `panels_cinematic.py` (and any other module written
against it) works unchanged.

    register_panel(name, fn)      -> registry.register_panel (floating panel, not drawn while the interface is hidden)
    register_overlay(name, fn)    -> registry.register_overlay (also drawn while hidden)
    register_menu(label, fn)      -> registry.register_top_menu (a complete top-level menu)
    register_key(name, fn)        -> registry.register_key_hook (fn(ui, ev) -> True when consumed)
    register_shortcuts(name, rows)-> registry.register_shortcuts
"""
from __future__ import annotations

from . import registry


def register_panel(name: str, fn):
    registry.register_panel(name, fn, order=90)


def register_overlay(name: str, fn):
    registry.register_overlay(name, fn)


def register_menu(label: str, fn):
    registry.register_top_menu(label, fn)


def register_key(name: str, fn):
    registry.register_key_hook(name, fn)


def register_shortcuts(name: str, rows):
    registry.register_shortcuts(name, rows)


def draw_panels(ui):          # panels are drawn by UiManager._draw_extensions
    pass


draw_overlays = registry.draw_overlays
draw_menus = registry.draw_top_menus
on_key = registry.on_key_hooks
