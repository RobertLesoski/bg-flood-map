"""Tiny extension registry for the interface: a feature module registers a panel, an overlay, a top-menu entry, a hotkey
handler and help rows with ONE call each (idempotent by name); `UiManager`, `menus.draw` call the `draw_*` / `on_key` functions
here.  Every callback runs guarded: an exception is logged once and shown by the manager like any other panel error.

    register_panel(name, fn)      fn(ui) drawn with the normal interface (not while the interface is hidden with H)
    register_overlay(name, fn)    fn(ui) drawn always (captions, progress bars): also while the interface is hidden
    register_menu(label, fn)      fn(ui) draws one top-level menu: `if imgui.begin_menu(label): ...; imgui.end_menu()`
    register_key(name, fn)        fn(ui, ev) -> True when it consumed the key_down event (asked before every built-in hotkey)
    register_shortcuts(rows)      [(key, description) | (section, None)] appended once to the F1 help dialog
"""
from __future__ import annotations

PANELS: dict = {}
OVERLAYS: dict = {}
MENUS: dict = {}
KEYS: dict = {}
_SHORTCUTS_DONE: set = set()


def register_panel(name: str, fn):
    PANELS[name] = fn


def register_overlay(name: str, fn):
    OVERLAYS[name] = fn


def register_menu(label: str, fn):
    MENUS[label] = fn


def register_key(name: str, fn):
    KEYS[name] = fn


def register_shortcuts(name: str, rows):
    """Append rows to the help dialog's table (once per `name`)."""
    if name in _SHORTCUTS_DONE:
        return
    _SHORTCUTS_DONE.add(name)
    try:
        from . import dialogs
        dialogs.SHORTCUTS.extend(rows)
    except Exception:
        pass


def draw_panels(ui):
    for name, fn in list(PANELS.items()):
        ui.safe(name, fn)


def draw_overlays(ui):
    for name, fn in list(OVERLAYS.items()):
        ui.safe(name, fn)


def draw_menus(ui):
    for label, fn in list(MENUS.items()):
        ui.safe("menu " + label, fn)


def on_key(ui, ev) -> bool:
    for name, fn in list(KEYS.items()):
        try:
            if fn(ui, ev):
                return True
        except Exception as e:
            if name not in ui.error_logged:
                ui.error_logged.add(name)
                try:
                    ui.app.report_error(f"key hook {name}", e)
                except Exception:
                    pass
    return False
