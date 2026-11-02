"""Dark theme (colours of the web viewer: navy surfaces, #4dabf7 accent, orange / red depth tiers), fonts and a few
colour helpers.  `setup(imgui, scale)` is called once after the imgui context exists (before the first frame)."""
from __future__ import annotations

import math
from pathlib import Path


def hexrgb(h: str, a: float = 1.0) -> tuple:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, a)


def with_alpha(c: tuple, a: float) -> tuple:
    return (c[0], c[1], c[2], a)


def mix(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


def col32(c: tuple) -> int:
    """(r,g,b,a) floats -> imgui packed ABGR u32."""
    r, g, b, a = c
    r = 0 if r <= 0.0 else 255 if r >= 1.0 else int(r * 255.0 + 0.5)
    g = 0 if g <= 0.0 else 255 if g >= 1.0 else int(g * 255.0 + 0.5)
    b = 0 if b <= 0.0 else 255 if b >= 1.0 else int(b * 255.0 + 0.5)
    a = 0 if a <= 0.0 else 255 if a >= 1.0 else int(a * 255.0 + 0.5)
    return (a << 24) | (b << 16) | (g << 8) | r


# ------------------------------------------------------------------------------------------------ palette
BG0 = hexrgb("#07070f")
BG1 = hexrgb("#0d0d1a")
BG2 = hexrgb("#13132a")
BG3 = hexrgb("#1a1a3a")
BORDER = hexrgb("#262648")
ACCENT = hexrgb("#4dabf7")
ACCENT2 = hexrgb("#74c0fc")
TEXT = hexrgb("#d6d9ea")
TEXT_DIM = hexrgb("#8e93ad")
TEXT_MUTE = hexrgb("#5f6480")
RED = hexrgb("#C8102E")
OK = hexrgb("#4cd08a")
WARN = hexrgb("#ffb347")
ERR = hexrgb("#ff6b5e")

TIER_RGB = (hexrgb("#4dabf7"), hexrgb("#00c9c9"), hexrgb("#ff8c00"), hexrgb("#e74c3c"))   # shallow..severe (web viewer)
TIER_NAMES = ("Shallow", "Moderate", "Deep", "Severe")


def tier_color(tier: int, a: float = 1.0) -> tuple:
    if 1 <= tier <= 4:
        return with_alpha(TIER_RGB[tier - 1], a)
    return (0.5, 0.5, 0.55, a)


# Depth colour ramp of the water shader (terrain.py `depth_ramp`).  The legend reads the constants straight out of the WGSL
# source so it can never drift from what the shader draws; the literals below are only the fallback.
_RAMP = ((0.34, 0.62, 0.66), (0.15, 0.50, 0.74), (0.06, 0.33, 0.64), (0.03, 0.18, 0.46), (0.01, 0.07, 0.26))
_EDGES = ((0.0, 0.3), (0.3, 0.9), (0.9, 2.2), (2.2, 5.5))
RAMP_FROM_WGSL = False


def _load_ramp_from_wgsl() -> bool:
    global _RAMP, _EDGES, RAMP_FROM_WGSL
    try:
        import re
        from ..gpu.layers import terrain
        src = terrain.WATER_WGSL
        body = src[src.index("fn depth_ramp"):]
        body = body[:body.index("\n}")]
        cols = re.findall(r"let c(\d) = vec3<f32>\(([\d.]+), ([\d.]+), ([\d.]+)\);", body)
        edges = re.findall(r"smoothstep\(([\d.]+), ([\d.]+), d\)", body)
        if len(cols) == 5 and len(edges) == 4:
            _RAMP = tuple(tuple(float(v) for v in c[1:]) for c in cols)
            _EDGES = tuple((float(a), float(b)) for a, b in edges)
            RAMP_FROM_WGSL = True
            return True
    except Exception:
        pass
    return False


def _smooth(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def depth_ramp(d: float) -> tuple:
    """CPU copy of the WGSL `depth_ramp(d)` (linear colour; see `to_srgb` for what the screen shows)."""
    if not RAMP_FROM_WGSL:
        _load_ramp_from_wgsl()
    c0, c1, c2, c3, c4 = _RAMP

    def lerp(a, b, t):
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
    c = lerp(c0, c1, _smooth(*_EDGES[0], d))
    c = lerp(c, c2, _smooth(*_EDGES[1], d))
    c = lerp(c, c3, _smooth(*_EDGES[2], d))
    return lerp(c, c4, _smooth(*_EDGES[3], d))


def to_srgb(c) -> tuple:
    """The shader writes LINEAR colour into an sRGB target: the screen shows the sRGB encoding of the ramp value."""
    def enc(x):
        x = max(0.0, min(1.0, x))
        return 12.92 * x if x <= 0.0031308 else 1.055 * (x ** (1 / 2.4)) - 0.055
    return (enc(c[0]), enc(c[1]), enc(c[2]), 1.0)


def depth_ramp_srgb(d: float) -> tuple:
    return to_srgb(depth_ramp(d))


# Flow-speed colours: the particle layer is free to restyle its streaks; this is the legend's ramp (slow -> fast).
SPEED_STOPS = ((0.0, hexrgb("#9fd4ff")), (0.6, hexrgb("#ffffff")), (1.5, hexrgb("#ffe27a")), (2.6, hexrgb("#ff8c42")))


def speed_ramp(v: float) -> tuple:
    stops = SPEED_STOPS
    if v <= stops[0][0]:
        return stops[0][1]
    for (v0, c0), (v1, c1) in zip(stops, stops[1:]):
        if v <= v1:
            return mix(c0, c1, (v - v0) / (v1 - v0))
    return stops[-1][1]


# ------------------------------------------------------------------------------------------------ fonts / style
_FONT_DIRS = None


def _font_file(name: str) -> str | None:
    try:
        import imgui_bundle
        p = Path(imgui_bundle.__file__).parent / "assets" / "fonts" / "Roboto" / name
        if p.is_file():
            return str(p)
    except Exception:
        pass
    for p in (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if p.is_file():
            return str(p)
    return None


class Fonts:
    regular = None
    bold = None
    big = None
    loaded = False


def load_fonts(imgui, size: float = 15.0) -> bool:
    """Roboto (bundled with imgui_bundle).  Dynamic font sizes (imgui >= 1.92): one regular + one bold face."""
    try:
        io = imgui.get_io()
        reg = _font_file("Roboto-Regular.ttf")
        bold = _font_file("Roboto-Bold.ttf")
        if reg is None:
            return False
        Fonts.regular = io.fonts.add_font_from_file_ttf(reg, size)
        Fonts.bold = io.fonts.add_font_from_file_ttf(bold or reg, size)
        Fonts.loaded = Fonts.regular is not None
        return Fonts.loaded
    except Exception:
        Fonts.loaded = False
        return False


def setup(imgui, scale: float = 1.0) -> None:
    """Fonts + style + colours.  `scale` multiplies fonts and sizes (user UI scale); the framebuffer / DPI factor is
    handled by imgui itself (display_framebuffer_scale), so coordinates here are LOGICAL pixels."""
    load_fonts(imgui)
    try:
        from imgui_bundle import implot
        if implot.get_current_context() is None:
            implot.create_context()
    except Exception:
        pass
    st = imgui.get_style()
    st.window_rounding = 8.0
    st.child_rounding = 6.0
    st.frame_rounding = 5.0
    st.popup_rounding = 6.0
    st.grab_rounding = 5.0
    st.scrollbar_rounding = 6.0
    st.tab_rounding = 5.0
    st.window_border_size = 1.0
    st.child_border_size = 0.0
    st.frame_border_size = 0.0
    st.popup_border_size = 1.0
    st.window_padding = (12.0, 10.0)
    st.frame_padding = (9.0, 5.0)
    st.item_spacing = (8.0, 6.0)
    st.item_inner_spacing = (6.0, 4.0)
    st.cell_padding = (6.0, 3.0)
    st.indent_spacing = 14.0
    st.scrollbar_size = 11.0
    st.grab_min_size = 12.0
    st.window_title_align = (0.0, 0.5)
    st.separator_text_border_size = 1.0
    set_scale(imgui, scale)
    V = imgui.ImVec4
    C = imgui.Col_
    cols = {
        C.text: TEXT, C.text_disabled: TEXT_DIM,
        C.window_bg: with_alpha(BG1, 0.94), C.child_bg: (0.0, 0.0, 0.0, 0.0), C.popup_bg: with_alpha(BG2, 0.98),
        C.border: with_alpha(BORDER, 0.9), C.border_shadow: (0, 0, 0, 0),
        C.frame_bg: with_alpha(BG3, 0.75), C.frame_bg_hovered: with_alpha(mix(BG3, ACCENT, 0.22), 0.9),
        C.frame_bg_active: with_alpha(mix(BG3, ACCENT, 0.36), 1.0),
        C.title_bg: BG1, C.title_bg_active: BG2, C.title_bg_collapsed: BG1, C.menu_bar_bg: with_alpha(BG1, 0.97),
        C.scrollbar_bg: (0.0, 0.0, 0.0, 0.0), C.scrollbar_grab: with_alpha(mix(BG3, TEXT, 0.25), 0.9),
        C.scrollbar_grab_hovered: with_alpha(mix(BG3, TEXT, 0.4), 1.0), C.scrollbar_grab_active: ACCENT,
        C.check_mark: ACCENT2, C.slider_grab: ACCENT, C.slider_grab_active: ACCENT2,
        C.button: with_alpha(mix(BG3, ACCENT, 0.30), 1.0), C.button_hovered: with_alpha(mix(BG3, ACCENT, 0.55), 1.0),
        C.button_active: with_alpha(mix(BG3, ACCENT, 0.75), 1.0),
        C.header: with_alpha(mix(BG2, ACCENT, 0.20), 1.0), C.header_hovered: with_alpha(mix(BG2, ACCENT, 0.38), 1.0),
        C.header_active: with_alpha(mix(BG2, ACCENT, 0.5), 1.0),
        C.separator: with_alpha(BORDER, 1.0), C.separator_hovered: ACCENT, C.separator_active: ACCENT2,
        C.resize_grip: with_alpha(ACCENT, 0.15), C.resize_grip_hovered: with_alpha(ACCENT, 0.5),
        C.resize_grip_active: with_alpha(ACCENT, 0.8),
        C.tab: with_alpha(BG2, 1.0), C.tab_hovered: with_alpha(ACCENT, 0.6), C.tab_selected: with_alpha(mix(BG3, ACCENT, 0.4), 1.0),
        C.plot_lines: ACCENT, C.plot_histogram: ACCENT,
        C.table_header_bg: with_alpha(BG2, 1.0), C.table_border_strong: with_alpha(BORDER, 1.0),
        C.table_border_light: with_alpha(BORDER, 0.5), C.table_row_bg: (0, 0, 0, 0), C.table_row_bg_alt: (1, 1, 1, 0.025),
        C.text_selected_bg: with_alpha(ACCENT, 0.35), C.nav_cursor: ACCENT,
    }
    for col, rgba in cols.items():
        try:
            st.set_color_(col, V(*rgba))
        except Exception:
            pass


def set_scale(imgui, scale: float) -> None:
    """User UI scale: font size only (layout constants in panels.py multiply by the same factor)."""
    try:
        imgui.get_style().font_scale_main = float(scale)
    except Exception:
        pass
