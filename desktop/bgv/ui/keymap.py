"""One table of every keyboard binding of the viewer, the conflict check over it, and the F1 cheat sheet generated from it.

Built-in handlers (manager.on_key, App._on_key, the camera, the search / cinematic / analysis key hooks) are plain code, so
their keys are DECLARED here (BUILTIN, one row per key and context); tests/test_ux_audit.py drives the real handlers with
these keys so the declarations cannot drift from the code.  Everything registered through `bgv.ui.registry`
(register_hotkey, register_tool shortcuts) and through `declare()` (key hooks of extension modules) is added at run time.

Dispatch order of a key_down (bgv/app.py _on_key -> UiManager.on_key):
    text box focused -> everything goes to the box (Esc leaves it, Ctrl+F re-focuses search)
    registry key hooks, in registration order (cinematic, analysis, search, features ...)
    manager built-ins (F1 F3 F11 F12 H 1..9 Esc [ ] F Home) -> active registry tool on_key -> registry hotkeys
    App (Space , .) -> camera (W A S D Q E R F + - arrows)

A context says when a binding is live: "global" always, "tour" while the cinematic tour runs, "lapse" while the time-lapse
plays, "analysis tool" while a measure / profile tool of the Analysis menu is active, "tool" while a registry tool is active,
"selection" while a building is selected.  Two bindings of the same key + modifiers conflict when their contexts can be live at
the same time and they belong to different owners, unless both are in the same declared priority chain (`chain`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

MODS_ORDER = ("Control", "Alt", "Shift")
_MOD_LABEL = {"Control": "Ctrl", "Alt": "Alt", "Shift": "Shift"}


@dataclass(frozen=True)
class Binding:
    key: str                     # key name as the canvas reports it, lower case for letters ("g", "F6", "Escape", " ")
    mods: tuple = ()             # subset of ("Control", "Alt", "Shift")
    owner: str = ""              # module / feature that handles it
    context: str = "global"
    description: str = ""
    group: str = "Keyboard"      # cheat-sheet section
    chain: str = ""              # intentional priority chain (e.g. "esc"): members never conflict with each other
    label: str = ""              # cheat-sheet key text when one row stands for several keys ("1 .. 9")
    row: bool = True             # False: covered by another row's label (still checked for conflicts)


def norm_mods(mods) -> tuple:
    m = set(mods or ())
    if "Meta" in m:
        m.discard("Meta")
        m.add("Control")
    return tuple(x for x in MODS_ORDER if x in m)


def key_text(b: Binding) -> str:
    if b.label:
        return b.label
    k = {" ": "Space", "Escape": "Esc"}.get(b.key, b.key.upper() if len(b.key) == 1 else b.key)
    return "+".join([_MOD_LABEL[m] for m in b.mods] + [k])


def _B(key, mods, owner, ctx, desc, group="Keyboard", **kw):
    return Binding(key, norm_mods(mods), owner, ctx, desc, group, **kw)


_DIGITS = "123456789"
BUILTIN: list[Binding] = [
    # camera (bgv/gpu/camera.py) - held keys
    *[_B(k, (), "camera", "global", "Move the camera (Shift = faster)", label="W A S D  /  arrows", row=(k == "w"))
      for k in ("w", "a", "s", "d", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight")],
    _B("q", (), "camera", "global", "Rotate left / right", label="Q  E"),
    _B("e", (), "camera", "global", "", row=False),
    _B("r", (), "camera", "global", "Tilt (F flies to the selection when one exists)", label="R  F"),
    _B("f", (), "camera", "global", "", row=False, chain="f"),
    _B("f", (), "manager", "selection", "Fly to the selected building", chain="f"),
    _B("+", (), "camera", "global", "Zoom in / out", label="+  -"),
    _B("-", (), "camera", "global", "", row=False),
    # App._on_key
    _B(" ", (), "app", "global", "Play / pause (pauses the tour / time-lapse while those run)", chain="space"),
    _B(",", (), "app", "global", "Step one hour back / forward", label=",  .   or   [  ]"),
    _B(".", (), "app", "global", "", row=False),
    # manager.on_key
    _B("[", (), "manager", "global", "", row=False),
    _B("]", (), "manager", "global", "", row=False),
    *[_B(d, (), "manager", "global", "Pick the n-th storm (storms 10-12: sidebar or Storm menu)", label="1 .. 9", row=(d == "1"))
      for d in _DIGITS],
    _B("Home", (), "manager", "global", "Overview (reset view)"),
    _B("h", (), "manager", "global", "Hide / show the interface"),
    _B("F1", (), "manager", "global", "Help and this cheat sheet"),
    _B("F3", (), "manager", "global", "Performance overlay"),
    _B("F11", (), "manager", "global", "Fullscreen"),
    _B("F12", (), "manager", "global", "Screenshot (Shift+F12: without the interface)"),
    _B("F12", ("Shift",), "manager", "global", "", row=False),
    _B("Escape", (), "manager", "global", "Leave the active tool, close dialogs, deselect (in that order)", chain="esc"),
    # search key hook (bgv/ui/panels_search.py)
    _B("f", ("Control",), "search", "global", "Search an address (typeahead; Enter jumps, Up / Down choose)", "Search and reports",
       label="Ctrl+F  or  /"),
    _B("/", (), "search", "global", "", "Search and reports", row=False),
    _B("m", (), "search", "global", "Show / hide the critical-facility markers", "Search and reports"),
    _B("l", ("Control",), "search", "global", "Open / close the roads-at-risk list", "Search and reports"),
    _B("p", ("Control",), "search", "global", "Save a risk report (HTML + PNG)", "Search and reports"),
    _B("Escape", (), "search", "global", "Clear the search box / close the result card", "Search and reports", chain="esc"),
    # cinematic key hook (bgv/ui/panels_cinematic.py)
    _B("b", ("Control",), "cinematic", "global", "Save the current view as a bookmark", "Cinematic"),
    *[_B(d, ("Control",), "cinematic", "global", "Fly to bookmark 1 .. 9", "Cinematic", label="Ctrl+1 .. 9", row=(d == "1"))
      for d in _DIGITS],
    *[_B(d, ("Alt",), "cinematic", "global", "Overwrite bookmark 1 .. 9 with the current view", "Cinematic", label="Alt+1 .. 9",
         row=(d == "1")) for d in _DIGITS],
    _B("e", ("Control",), "cinematic", "global", "Cinematic panel, export tab", "Cinematic"),
    _B("t", (), "cinematic", "global", "Start / stop the automatic tour", "Cinematic"),
    _B("l", (), "cinematic", "global", "Play / pause the time-lapse", "Cinematic"),
    _B("c", (), "cinematic", "global", "Show / hide the Cinematic panel", "Cinematic"),
    _B("n", (), "cinematic", "tour", "Next place (tour running)", "Cinematic", label="N  P"),
    _B("p", (), "cinematic", "tour", "", "Cinematic", row=False),
    _B(" ", (), "cinematic", "tour", "", "Cinematic", row=False, chain="space"),
    _B(" ", (), "cinematic", "lapse", "", "Cinematic", row=False, chain="space"),
    _B("Escape", (), "cinematic", "tour", "Stop the tour / camera flight", "Cinematic", chain="esc"),
    # analysis key hook (bgv/ui/panels_analysis.py)
    _B("Enter", (), "analysis", "analysis tool", "Finish the measurement", "Analysis tools"),
    _B("Backspace", (), "analysis", "analysis tool", "Remove the last point", "Analysis tools"),
    _B("Escape", (), "analysis", "analysis tool", "Leave the measure / profile tool", "Analysis tools", chain="esc"),
]

EXTRA: list[Binding] = []          # declare(): keys handled by extension key hooks (not visible to the registry)

MOUSE_ROWS = [
    ("Left drag", "Pan the map"),
    ("Right / middle drag", "Orbit (rotate and tilt)"),
    ("Mouse wheel", "Zoom toward the cursor"),
    ("Click a building", "Select it (card + highlight)"),
    ("Click a facility marker", "Show its flood depths and the first storm that floods it"),
    ("Click the compass", "Face north"),
    ("Click / drag in the timeline chart", "Seek in time"),
]

GROUP_ORDER = ("Search and reports", "Mouse", "Keyboard", "Cinematic", "Analysis tools", "Features")


def declare(key, mods=(), *, owner, context="global", description="", group="Features", chain="", label="", row=True):
    b = _B(key, mods, owner, context, description, group, chain=chain, label=label, row=row)
    if b not in EXTRA:
        EXTRA.append(b)
    return b


def from_registry() -> list[Binding]:
    from . import registry
    out = []
    tool_keys = {t.shortcut.lower() for t in registry.TOOLS.values() if t.shortcut}
    for h in registry.HOTKEYS:
        k = h.key if len(h.key) != 1 else h.key.lower()
        out.append(_B(k, h.mods, "registry" + (" tool" if k in tool_keys else ""), "global",
                      h.description, "Features"))
    return out


def all_bindings() -> list[Binding]:
    return list(BUILTIN) + list(EXTRA) + from_registry()


def _ctx_overlap(a: str, b: str) -> bool:
    return a == b or "global" in (a, b) or {a, b} <= {"selection", "tour", "lapse", "analysis tool", "tool"}


def conflicts(bindings=None) -> list[tuple[Binding, Binding]]:
    """Pairs of bindings that would both want the same key at the same time (different owners, overlapping contexts, not one
    declared priority chain).  The registry runs its tools / hotkeys only when no built-in consumed the key, so a registry key on a
    built-in key is a conflict (the registry one would never fire)."""
    bs = all_bindings() if bindings is None else list(bindings)
    out = []
    for i, a in enumerate(bs):
        for b in bs[i + 1:]:
            if a.key != b.key or a.mods != b.mods or a.owner == b.owner:
                continue
            if a.chain and a.chain == b.chain:
                continue
            if _ctx_overlap(a.context, b.context):
                out.append((a, b))
    return out


def help_groups(bindings=None) -> list:
    """[(group, [(key text, description), ...]), ...] in the format of bgv.search.helpdoc.SHORTCUTS."""
    bs = all_bindings() if bindings is None else list(bindings)
    groups: dict = {g: [] for g in GROUP_ORDER}
    groups["Mouse"] = list(MOUSE_ROWS)
    seen = set()
    for b in bs:
        if not b.row or not b.description:
            continue
        kt = key_text(b)
        if (b.group, kt) in seen:
            continue
        seen.add((b.group, kt))
        groups.setdefault(b.group, []).append((kt, b.description))
    return [(g, rows) for g, rows in groups.items() if rows]


def install_help() -> None:
    """Replace the F1 cheat sheet with the generated one (in place: help.py reads helpdoc.SHORTCUTS each frame)."""
    from ..search import helpdoc
    helpdoc.SHORTCUTS[:] = help_groups()
