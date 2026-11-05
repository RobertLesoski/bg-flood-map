"""Extension registry: how a feature plugs into the interface WITHOUT editing the shared UI files.

Each feature lives in its own module `bgv/ui/ext_<name>.py` that calls the `register_*` functions at import time.  The UI
manager imports every `ext_*.py` once (`load_extensions`), so two branches that each add one file never conflict.
A module that fails to import, or a hook that raises, is logged once and skipped; the viewer keeps running.

    register_menu_item("Tools", "Measure distance", action, shortcut="M", checked=lambda ui: ..., enabled=lambda ui: ...)
    register_panel("mypanel", draw, order=100, when=lambda ui: True)         # drawn every frame after the built-in panels
    register_tool(MyTool())                                                  # click / move / overlay hooks, listed in Tools
    register_sidebar_section("mysec", "My section", draw, order=100)         # collapsible section at the end of the sidebar
    register_hotkey("m", fn, mods=(), description="Measure")                 # key_down, only when no text box has focus
    state(app, "mykey") -> dict                                               # persisted per-feature settings (user_settings.json "ext")

All callbacks get the `UiManager` (`ui`; `ui.app` is the App, `ui.st` the persistent UiState, `ui.L` the layout, `ui.s` the
scale).  Built-in menu names: "File", "View", "Domain", "Camera", "Tools", "Help".  The documented contract is
`desktop/docs/EXTENSION_API.md`; keep these signatures stable.
"""
from __future__ import annotations

import importlib
import pkgutil
import traceback
from dataclasses import dataclass, field
from typing import Callable

MENUS = ("File", "View", "Domain", "Camera", "Tools", "Help")


@dataclass
class MenuItem:
    menu: str
    label: str
    action: Callable
    shortcut: str = ""
    checked: Callable | None = None
    enabled: Callable | None = None
    order: int = 100


@dataclass
class Panel:
    name: str
    draw: Callable
    order: int = 100
    when: Callable | None = None


@dataclass
class SidebarSection:
    key: str
    title: str
    draw: Callable
    order: int = 100
    default_open: bool = False


@dataclass
class Hotkey:
    key: str
    fn: Callable
    mods: tuple = ()
    description: str = ""


class Tool:
    """Base class of an interactive tool (measure, cross-section, ...).  Subclass, set `name` / `title`, register an instance.

    `active` is toggled by the Tools menu (or `ui.activate_tool(name)`); only one tool is active at a time.
    on_click(ui, x, y, pick)   x, y logical pixels of a click that was NOT a drag; pick = (X, Y, Z, t) world ground point
                               under it (Z exaggerated) or None.  Return True to consume the click (no building selection).
    on_move(ui, x, y, pick)    pointer moved over the 3D view while the tool is active.
    draw_overlay(ui)           called every frame while active (draw with imgui's foreground / window draw list).
    on_activate(ui) / on_deactivate(ui) / on_key(ui, key, mods) -> bool (Esc is handled by the manager: deactivates).
    """
    name = "tool"
    title = "Tool"
    shortcut = ""
    hint = ""                  # one line shown in the status area while the tool is active

    def __init__(self):
        self.active = False

    def on_activate(self, ui): ...
    def on_deactivate(self, ui): ...
    def on_click(self, ui, x, y, pick) -> bool:
        return False
    def on_move(self, ui, x, y, pick): ...
    def draw_overlay(self, ui): ...
    def on_key(self, ui, key, mods) -> bool:
        return False


# Whole-menu / always-drawn / key-event hooks (used by feature modules that own a complete top-level menu, an overlay that must
# also show while the interface is hidden with H, or raw key_down events).  Ported from the cinematic branch's hooks.py.
OVERLAYS: dict = {}           # name -> fn(ui), drawn every frame, also while the interface is hidden
TOP_MENUS: dict = {}          # label -> fn(ui) drawing `if imgui.begin_menu(label): ...; imgui.end_menu()`
KEY_HOOKS: dict = {}
CLICK_HOOKS: dict = {}        # name -> fn(ui, x, y) -> True when it consumed a (non-drag) click on the 3D view          # name -> fn(ui, ev) -> True when it consumed the key_down event (asked before built-in hotkeys)
_SHORTCUTS_DONE: set = set()

MENU_ITEMS: list[MenuItem] = []
PANELS: list[Panel] = []
SECTIONS: list[SidebarSection] = []
HOTKEYS: list[Hotkey] = []
TOOLS: dict[str, Tool] = {}
LOADED: list[str] = []
FAILED: dict[str, str] = {}


def register_menu_item(menu, label, action, *, shortcut="", checked=None, enabled=None, order=100):
    MENU_ITEMS.append(MenuItem(menu, label, action, shortcut, checked, enabled, order))


def register_panel(name, draw, *, order=100, when=None):
    PANELS[:] = [p for p in PANELS if p.name != name]
    PANELS.append(Panel(name, draw, order, when))
    PANELS.sort(key=lambda p: p.order)


def register_sidebar_section(key, title, draw, *, order=100, default_open=False):
    SECTIONS[:] = [s for s in SECTIONS if s.key != key]
    SECTIONS.append(SidebarSection(key, title, draw, order, default_open))
    SECTIONS.sort(key=lambda s: s.order)


def register_hotkey(key, fn, *, mods=(), description=""):
    HOTKEYS.append(Hotkey(key, fn, tuple(mods), description))


def register_tool(tool: Tool, *, menu="Tools"):
    TOOLS[tool.name] = tool
    register_menu_item(menu, tool.title, lambda ui, t=tool: ui.toggle_tool(t.name), shortcut=tool.shortcut,
                       checked=lambda ui, t=tool: t.active, order=50)
    if tool.shortcut:
        register_hotkey(tool.shortcut.lower(), lambda ui, t=tool: ui.toggle_tool(t.name), description=tool.title)


def menu_items(menu: str) -> list[MenuItem]:
    return sorted((m for m in MENU_ITEMS if m.menu == menu), key=lambda m: m.order)


def state(app, key: str) -> dict:
    """Persistent per-feature settings dict (saved with the other settings when the window closes)."""
    ext = app.settings.setdefault("ext", {})
    d = ext.get(key)
    if not isinstance(d, dict):
        d = ext[key] = {}
    return d


def load_extensions(log=None) -> list[str]:
    """Import every bgv.ui.ext_*.py once (idempotent).  Returns the names that loaded."""
    if LOADED or FAILED:
        return list(LOADED)
    pkg = __package__
    ui_pkg = importlib.import_module(pkg)
    for mod in sorted(m.name for m in pkgutil.iter_modules(ui_pkg.__path__) if m.name.startswith("ext_")):
        full = f"{pkg}.{mod}"
        try:
            importlib.import_module(full)
            LOADED.append(mod)
        except Exception as e:                                   # a broken extension never stops the viewer
            FAILED[mod] = f"{type(e).__name__}: {e}"
            try:
                from .. import applog
                applog.log().error("extension %s failed to load: %s\n%s", mod, e, traceback.format_exc())
            except Exception:
                pass
            if log:
                log(f"extension {mod} failed to load: {e}")
    return list(LOADED)


def call(ui, fn, *args, default=None, what=""):
    """Run an extension callback; log the first failure of each `what`, return `default` on error."""
    try:
        return fn(ui, *args)
    except Exception as e:
        key = what or getattr(fn, "__name__", "ext")
        seen = getattr(ui, "_ext_errors", None)
        if seen is None:
            seen = ui._ext_errors = set()
        if key not in seen:
            seen.add(key)
            try:
                ui.app.report_error(f"ext.{key}", e)
            except Exception:
                pass
        return default


def register_overlay(name, fn):
    OVERLAYS[name] = fn


def register_top_menu(label, fn):
    TOP_MENUS[label] = fn


def register_key_hook(name, fn):
    KEY_HOOKS[name] = fn


def register_shortcuts(name, rows):
    """Append rows [(key, description) | (section, None)] to the F1 help dialog's table (once per `name`)."""
    if name in _SHORTCUTS_DONE:
        return
    _SHORTCUTS_DONE.add(name)
    try:
        from . import dialogs
        dialogs.SHORTCUTS.extend(rows)
    except Exception:
        pass


def draw_overlays(ui):
    for name, fn in list(OVERLAYS.items()):
        ui.safe(name, fn)


def draw_top_menus(ui):
    for label, fn in list(TOP_MENUS.items()):
        ui.safe("menu " + label, fn)


def on_key_hooks(ui, ev) -> bool:
    for name, fn in list(KEY_HOOKS.items()):
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


def register_click_hook(name, fn):
    CLICK_HOOKS[name] = fn


def on_click_hooks(ui, x, y) -> bool:
    for name, fn in list(CLICK_HOOKS.items()):
        try:
            if fn(ui, x, y):
                return True
        except Exception as e:
            if name not in ui.error_logged:
                ui.error_logged.add(name)
                try:
                    ui.app.report_error(f"click hook {name}", e)
                except Exception:
                    pass
    return False
