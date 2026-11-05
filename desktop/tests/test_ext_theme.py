"""Extension registry, interface theme / colour palette switching and the persistence of the new UI settings (CPU only)."""
from __future__ import annotations

import types

from bgv import palette
from bgv.ui import registry, theme
from bgv.ui.state import UiState


def test_palettes_are_four_distinct_tiers_and_cvd_gets_darker_with_depth():
    for key in palette.PALETTES:
        assert len(set(palette.hexes(key))) == 4
    lum = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in palette.linear("cvd")]
    assert lum == sorted(lum, reverse=True)                   # survives greyscale: depth = darker
    assert palette.hexes("nonsense") == palette.hexes("std")


def test_theme_palette_and_mode_switch_and_restore():
    try:
        theme.set_palette("cvd")
        assert theme.TIER_RGB[0] == theme.hexrgb("#f0da46") and theme.PALETTE_KEY == "cvd"
        theme.set_palette("bogus")
        assert theme.PALETTE_KEY == "std"
        dark_text = theme.TEXT
        theme.set_mode("light")
        assert theme.LIGHT_ON and theme.TEXT != dark_text and theme.BG1[0] > 0.9
    finally:
        theme.set_mode("dark")
        theme.set_palette("std")
    assert not theme.LIGHT_ON and theme.TEXT == dark_text


def test_ui_state_round_trips_theme_tiles_and_ignores_garbage():
    st = UiState()
    st.theme_mode, st.use_tiles, st.tile_stats = "light", False, True
    d = st.to_dict()
    back = UiState.from_dict(d)
    assert (back.theme_mode, back.use_tiles, back.tile_stats) == ("light", False, True)
    bad = UiState.from_dict({"theme_mode": "neon", "use_tiles": "yes please"})
    assert bad.theme_mode == "dark" and bad.use_tiles is True


class _FakeUI:
    def __init__(self):
        self.app = types.SimpleNamespace(settings={}, report_error=lambda *a: None)
        self.tool = None

    def activate_tool(self, name):
        cur = self.tool
        if cur is not None and (name is None or cur.name != name):
            cur.active = False
            self.tool = None
        if name is not None and name in registry.TOOLS and self.tool is None:
            self.tool = registry.TOOLS[name]
            self.tool.active = True

    toggle_tool = lambda self, n: self.activate_tool(None if (self.tool and self.tool.name == n) else n)  # noqa: E731


def test_registry_items_tools_state_and_failing_callbacks():
    ui = _FakeUI()
    hits = []

    class T(registry.Tool):
        name, title, shortcut = "unit_test_tool", "Unit test tool", "u"

        def on_click(self, ui_, x, y, pick):
            hits.append((x, y))
            return True

    n_items, n_keys = len(registry.MENU_ITEMS), len(registry.HOTKEYS)
    try:
        registry.register_tool(T())
        registry.register_menu_item("Tools", "Zed", lambda ui_: hits.append("zed"), order=999)
        names = [m.label for m in registry.menu_items("Tools")]
        assert names.index("Unit test tool") < names.index("Zed")          # tools first (order 50), then ordered items
        ui.toggle_tool("unit_test_tool")
        assert ui.tool is registry.TOOLS["unit_test_tool"] and ui.tool.active
        assert ui.tool.on_click(ui, 3, 4, None) is True and hits == [(3, 4)]
        ui.toggle_tool("unit_test_tool")
        assert ui.tool is None
        # a callback that raises is swallowed and reported once
        boom = lambda ui_: 1 / 0  # noqa: E731
        assert registry.call(ui, boom, default="x", what="boom") == "x"
        assert registry.call(ui, boom, default="y", what="boom") == "y"
        assert registry.state(ui.app, "k") is registry.state(ui.app, "k")
        registry.state(ui.app, "k")["a"] = 1
        assert ui.app.settings["ext"]["k"] == {"a": 1}                     # persisted with the other settings
    finally:
        registry.TOOLS.pop("unit_test_tool", None)
        del registry.MENU_ITEMS[n_items:]
        del registry.HOTKEYS[n_keys:]


def test_extension_loader_survives_a_broken_module(tmp_path, monkeypatch):
    import sys
    import bgv.ui as ui_pkg
    (tmp_path / "ext_good_unit.py").write_text("from bgv.ui import registry\nregistry.register_menu_item('Help', 'Good unit', lambda ui: None)\n")
    (tmp_path / "ext_bad_unit.py").write_text("raise RuntimeError('broken on purpose')\n")
    monkeypatch.setattr(ui_pkg, "__path__", list(ui_pkg.__path__) + [str(tmp_path)])
    monkeypatch.setattr(registry, "LOADED", [])
    monkeypatch.setattr(registry, "FAILED", {})
    n = len(registry.MENU_ITEMS)
    try:
        ok = registry.load_extensions()
        assert "ext_good_unit" in ok and "ext_bad_unit" in registry.FAILED and "broken on purpose" in registry.FAILED["ext_bad_unit"]
    finally:
        del registry.MENU_ITEMS[n:]
        for m in ("bgv.ui.ext_good_unit", "bgv.ui.ext_bad_unit"):
            sys.modules.pop(m, None)
