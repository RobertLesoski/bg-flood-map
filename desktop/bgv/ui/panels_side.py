"""Left sidebar: Storm & Time, Layers, Water & Flow, Quality, Statistics, Info (collapsible, state persisted)."""
from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui

from . import bridge, theme, units
from .menus import source_info
from .theme import col32
from .widgets import bold, chip_toggle, dim, font_text, hbar, icon_button, pill, pill_size, section, text_c

V2 = imgui.ImVec2

# layers whose options live in "Water & Flow"; everything else with options goes to "Quality"
WATER_LAYERS = ("water", "particles", "arrows")
HIDDEN_LAYERS = ("overlay", "panels")

# quality presets: layer name -> {attribute: value per preset}; applied by attribute so layers need no extra API
QUALITY = {
    "Low":    {"terrain": {"lod_distance": 450.0}, "water": {"ripple": 0.5}},
    "Medium": {"terrain": {"lod_distance": 800.0}, "water": {"ripple": 0.8}},
    "High":   {"terrain": {"lod_distance": 1100.0}, "water": {"ripple": 1.0}},
    "Ultra":  {"terrain": {"lod_distance": 1800.0}, "water": {"ripple": 1.25}},
}


APP_QUALITY = {"Low": "low", "Medium": "medium", "High": "high", "Ultra": "high"}     # app.quality: read by particles / arrows


def apply_quality(app, preset: str):
    """Set the known knobs of every layer; layers with a `set_quality(preset)` method get called as well."""
    perf = getattr(app, "perf", None)
    if perf is not None:
        perf.gov.set_mode("manual")                    # picking a preset hands the settings to the user (the governor stands down)
    app.quality = APP_QUALITY.get(preset, "high")
    for lname, attrs in QUALITY.get(preset, {}).items():
        layer = app.get_layer(lname)
        if layer is None:
            continue
        for k, v in attrs.items():
            if hasattr(layer, k):
                setattr(layer, k, v)
    for layer in app.layers:
        fn = getattr(layer, "set_quality", None)
        if callable(fn):
            try:
                fn(preset)
            except Exception as e:
                app.report_error(f"{layer.name}.set_quality", e)


def draw_sidebar(ui):
    L, st, app = ui.L, ui.st, ui.app
    if not ui.begin_panel("##sidebar", 0.0, L.top, L.side_w, L.side_h, rounding=0.0,
                          padding=(12.0 * ui.s, 10.0 * ui.s)):
        ui.end_panel()
        return
    try:
        _header(ui)
        _sec_time(ui)
        _sec_layers(ui)
        _sec_water(ui)
        _sec_quality(ui)
        _sec_stats(ui)
        _sec_info(ui)
        for name, msg in list(ui.panel_errors.items()):
            text_c(theme.ERR, f"{name}: {msg[:70]}")
        for msg in list(app.errors)[-2:]:
            imgui.push_text_wrap_pos(0.0)
            text_c(theme.ERR, msg[:110])
            imgui.pop_text_wrap_pos()
    finally:
        ui.end_panel()


def _header(ui):
    bold("DSOC Flood Map", theme.ACCENT2)
    imgui.same_line()
    dim("  Bowling Green, KY")
    imgui.dummy(V2(0, 2))


def _open(ui, key: str, label: str) -> bool:
    ui.st.sections[key] = section(label, ui.st.sections.get(key, True))
    return ui.st.sections[key]


# ------------------------------------------------------------------------------------------------ storm & time
def _sec_time(ui):
    from ..app import SPEEDS
    app = ui.app
    if not _open(ui, "time", "Storm & Time"):
        return
    st = app.storm
    imgui.spacing()
    cur = app.storms_available.index(st.name) if st.name in app.storms_available else 0
    imgui.set_next_item_width(-1)
    ch, idx = imgui.combo("##storm", cur, [app.storm_label(s) for s in app.storms_available])
    if ch:
        app.set_storm(app.storms_available[idx])
    # source badge
    label, full = source_info(st)
    col = theme.OK if full else theme.WARN
    p = imgui.get_cursor_screen_pos()
    w, h = pill_size(label)
    pill(imgui.get_window_draw_list(), p.x, p.y, label, col, theme.with_alpha(col, 0.14), border=theme.with_alpha(col, 0.55))
    imgui.dummy(V2(w, h))
    if imgui.is_item_hovered():
        imgui.set_tooltip(str(st.meta.get("source_note", ""))[:600] or label)
    ready, total = app.flow.progress()
    if ready < total:
        hbar(ready / max(total, 1), theme.ACCENT, 5.0)
        dim(f"loading hours {ready}/{total}")
    imgui.spacing()
    # time slider
    dim("Time")
    rt = f"{app.t:5.2f} h   {units.hour_label(app.t)}"
    imgui.same_line(0.0, 0.0)
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - imgui.calc_text_size(rt).x)
    imgui.text(rt)
    imgui.set_next_item_width(-1)
    ch, t = imgui.slider_float("##t", app.t, 0.0, 24.0, "", imgui.SliderFlags_.always_clamp)
    if ch:
        app.t = float(t)
    # transport
    sz = 32.0 * ui.s
    if icon_button("##first", "first", sz, tip="Back to hour 0 ( [ steps back )"):
        app.t = 0.0
    imgui.same_line()
    if icon_button("##prev", "prev", sz, tip="Previous hour  ["):
        app.t = max(0.0, math.ceil(app.t - 1.0 - 1e-6))
    imgui.same_line()
    if icon_button("##play", "pause" if app.playing else "play", sz, active=app.playing, tip="Play / pause  (Space)"):
        app.playing = not app.playing
    imgui.same_line()
    if icon_button("##next", "next", sz, tip="Next hour  ]"):
        app.t = min(24.0, math.floor(app.t + 1.0 + 1e-6))
    imgui.same_line()
    if app.buffering:
        text_c(theme.WARN, "buffering...")
    else:
        dim(units.hour_of_24(app.t))
    imgui.spacing()
    dim("Speed")
    for i, sp in enumerate(SPEEDS):
        imgui.same_line()
        on = i == app.speed_idx
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*theme.mix(theme.BG3, theme.ACCENT, 0.7 if on else 0.15)))
        if imgui.button(f"{sp:g}x##sp{i}", V2(0, 0)):
            app.speed_idx = i
        imgui.pop_style_color()
    imgui.spacing()


# ------------------------------------------------------------------------------------------------ layers
def _sec_layers(ui):
    app, st = ui.app, ui.st
    if not _open(ui, "layers", "Layers"):
        return
    imgui.spacing()
    col2 = imgui.get_content_region_avail().x * 0.5
    k = 0
    for l in app.layers:
        if l.name in HIDDEN_LAYERS or (not l.draws and not l.has_ui):
            continue
        if k % 2:
            imgui.same_line(col2)
        imgui.push_id("lay" + l.name)
        ch, v = imgui.checkbox(l.label(), bool(l.enabled))
        if ch:
            l.enabled = bool(v)
        imgui.pop_id()
        k += 1
    imgui.spacing()
    imgui.set_next_item_open(False, imgui.Cond_.once)          # also in the View menu; closed to leave room for the statistics
    if imgui.tree_node("Overlays"):
        for k, (label, key) in enumerate((("Legends", "show_legends"), ("Compass", "show_compass"), ("Scale bar", "show_scalebar"),
                                          ("Probe tooltip", "show_probe"), ("At-risk list", "show_atrisk"),
                                          ("Building card", "show_selected"))):
            if k % 2:
                imgui.same_line(col2)
            imgui.push_id(key)
            cur = ui.atrisk_shown() if key == "show_atrisk" else bool(getattr(st, key))
            ch, v = imgui.checkbox(label, cur)
            if ch:
                if key == "show_atrisk":
                    ui.set_atrisk(bool(v))
                else:
                    setattr(st, key, bool(v))
            imgui.pop_id()
        imgui.tree_pop()
    imgui.spacing()


# ------------------------------------------------------------------------------------------------ water & flow
def _layer_ui(ui, l, default_open=True):
    imgui.push_id("ui" + l.name)
    try:
        imgui.set_next_item_open(default_open, imgui.Cond_.once)
        if imgui.tree_node_ex(l.label(), imgui.TreeNodeFlags_.framed if False else 0):
            imgui.push_item_width(-150.0 * ui.s)           # room for the layers' slider labels on the right
            try:
                l.ui(ui.app)
            except Exception as e:
                ui.app.report_error(f"{l.name}.ui", e)
                text_c(theme.ERR, f"{type(e).__name__}: {e}"[:80])
            finally:
                imgui.pop_item_width()
            imgui.tree_pop()
    finally:
        imgui.pop_id()


def _sec_water(ui):
    app = ui.app
    if not _open(ui, "water", "Water & Flow"):
        return
    imgui.spacing()
    imgui.set_next_item_width(-1)
    ch, e = imgui.slider_float("##exag", app.exag, 1.0, 10.0, "vertical exaggeration  x%.1f")
    if ch:
        app.set_exag(e)
    dim("Lighting")
    imgui.set_next_item_width(-1)
    ch, az = imgui.slider_float("##saz", app.sun_az, 0.0, 360.0, "sun azimuth  %.0f°")
    if ch:
        app.sun_az = az
    imgui.set_next_item_width(-1)
    ch, el = imgui.slider_float("##sel", app.sun_el, 3.0, 85.0, "sun elevation  %.0f°")
    if ch:
        app.sun_el = el
    imgui.set_next_item_width(-1)
    ch, hz = imgui.slider_float("##haze", app.haze_density * 1e5, 0.0, 6.0, "haze  %.1f")
    if ch:
        app.haze_density = hz * 1e-5
    imgui.spacing()
    for l in app.layers:
        if l.name in WATER_LAYERS and l.has_ui:
            _layer_ui(ui, l)
    imgui.spacing()


# ------------------------------------------------------------------------------------------------ quality
def _sec_quality(ui):
    app, st = ui.app, ui.st
    if not _open(ui, "quality", "Quality"):
        return
    imgui.spacing()
    dim("Preset")
    from .state import PRESETS
    gov = app.perf.gov if getattr(app, "perf", None) is not None else None
    if gov is not None:                                 # automatic quality: frame-time driven (bgv/perf.py), the default
        auto = gov.mode == "auto"
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*theme.mix(theme.BG3, theme.ACCENT, 0.7 if auto else 0.15)))
        if imgui.button("Auto", V2(0, 0)):
            gov.set_mode("auto")
        imgui.pop_style_color()
        imgui.same_line()
    for i, name in enumerate(PRESETS):
        if i or gov is not None:
            imgui.same_line()
        on = st.quality_preset == name and (gov is None or gov.mode != "auto")
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*theme.mix(theme.BG3, theme.ACCENT, 0.7 if on else 0.15)))
        if imgui.button(name, V2(0, 0)):
            st.quality_preset = name
            apply_quality(app, name)
        imgui.pop_style_color()
    if gov is not None:
        sc = getattr(app, "render_scale", 1.0)
        dim(gov.label + (f"   (scene at {sc * 100:.0f}% resolution)" if sc < 0.999 else "   (full resolution)")
            + ("" if gov.mode != "auto" else "  - adapts to your frame rate"), wrap=True)
    imgui.spacing()
    imgui.set_next_item_width(-1)
    pct = st.ui_scale * 100.0
    ch, pct = imgui.slider_float("##uis", pct, 75.0, 175.0, "interface size  %.0f%%")
    if ch:
        st.ui_scale = float(pct) / 100.0
    for l in app.layers:
        if l.has_ui and l.name not in WATER_LAYERS and l.name not in HIDDEN_LAYERS:
            _layer_ui(ui, l, default_open=False)
    imgui.spacing()
    dim(f"MSAA {app.sample_count}x (set at launch)    VRAM ~{app.ctx.vram_mb():.0f} MB", wrap=True)
    dim(f"{app.ctx.adapter_name}", wrap=True)
    imgui.spacing()


# ------------------------------------------------------------------------------------------------ statistics
def tier_rows(fl: dict) -> list[tuple[str, tuple, int]]:
    tc = fl.get("tier_counts") or {}
    keys = ("shallow", "moderate", "deep", "severe")
    return [(units.TIER_NAMES[i], theme.TIER_RGB[i], int(tc.get(k, 0))) for i, k in enumerate(keys)]


def max_depth_lines(fl: dict) -> tuple[str, str]:
    """(headline, note) honest max depth text: tier data only guarantees '>= 1.2 m' for the deepest tier."""
    d = float(fl.get("max_depth_m", 0.0))
    if fl.get("source") == "tiers":
        if d >= units.SEVERE_EDGE_M:
            return (f">= {units.SEVERE_EDGE_M:.1f} m  ({units.m_to_ft(units.SEVERE_EDGE_M):.1f} ft)",
                    f"depth tiers: the severe tier is open ended. Model estimate {d:.2f} m, indicative only")
        return (f"~{d:.2f} m  (~{units.m_to_ft(d):.1f} ft)", "depth tiers: depth known to about 0.1 m")
    return (f"{d:.2f} m  ({units.m_to_ft(d):.1f} ft)", "")


def stats_header(fl: dict) -> str:
    h = int(fl.get("hour", 0))
    return f"as of hour {h}  ({units.hour_label(h)})" + ("  - updating" if fl.get("pending") else "")


def _sec_stats(ui):
    app, st = ui.app, ui.st
    if not _open(ui, "stats", "Statistics"):
        return
    imgui.spacing()
    fl = bridge.flood(app)
    if fl is None or not fl.get("ready"):
        state, msg, frac = bridge.scene_state(app)
        dim("Statistics are being prepared...")
        if state in ("loading", "ready", "missing"):
            hbar(frac if state != "missing" else 0.0, theme.ACCENT, 5.0)
            dim(msg[:46] if msg else "", wrap=False)
        elif state == "error":
            text_c(theme.ERR, msg[:80])
        return
    text_c(theme.ACCENT2, stats_header(fl))
    imgui.spacing()
    n_all = max(int(fl.get("n_buildings_in_grid") or fl.get("n_buildings") or 1), 1)
    n_fl = int(fl.get("flooded_count", 0))
    # --- flooded buildings headline
    font_text(theme.Fonts.bold, 26.0, f"{n_fl:,}", theme.TIER_RGB[3] if n_fl else theme.TEXT)
    imgui.same_line()
    dim(f"  flooded buildings   ({100.0 * n_fl / n_all:.1f}% of {n_all:,})")
    imgui.spacing()
    rows = tier_rows(fl)
    mx = max([c for _, _, c in rows] + [1])
    total_w = imgui.get_content_region_avail().x
    for k, (name, col, cnt) in enumerate(rows):
        p = imgui.get_cursor_screen_pos()
        dl = imgui.get_window_draw_list()
        dl.add_rect_filled(V2(p.x, p.y + 3), V2(p.x + 10, p.y + 13), col32(col), 3.0)
        imgui.set_cursor_screen_pos(V2(p.x + 18, p.y))
        imgui.text(name)
        imgui.same_line(104 * ui.s)
        dim(units.tier_range_text(k + 1))
        txt = f"{cnt:,}"
        tw = imgui.calc_text_size(txt).x
        imgui.same_line(total_w - tw)
        imgui.text(txt)
        hbar(cnt / mx, col, 5.0)
    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    # --- other numbers
    head, note = max_depth_lines(fl)
    _kv("Max depth", head)
    if note:
        imgui.push_text_wrap_pos(0.0)
        dim(note, wrap=True)
        imgui.pop_text_wrap_pos()
    g = fl.get("grid")
    _kv("Wet area", units.fmt_area_km2(float(fl.get("wet_area_km2", 0.0))) + "   ("
        + units.fmt_area_km2(float(fl.get("wet_area_km2", 0.0)), True) + ")")
    if g is not None:
        _kv("Water volume", units.fmt_volume(float(g.volume_m3)))
    r = fl.get("roads")
    rk = float(fl.get("road_flooded_km", 0.0))
    if r is not None:
        tot_km = float(r.total_length_m) / 1000.0
        _kv("Roads flooded", f"{float(fl.get('road_pct', 0.0)):.1f}%   ({rk:.1f} of {tot_km:.0f} km)")
        _kv("Roads affected", f"{int(r.flooded_roads):,} road segments")
    imgui.spacing()
    imgui.push_text_wrap_pos(0.0)
    dim(f"{units.hour_of_24(fl.get('hour', 0))}: building flooded = water >= 0.1 m on its footprint.", wrap=True)
    imgui.pop_text_wrap_pos()
    imgui.spacing()


def _kv(k: str, v: str):
    dim(k)
    imgui.same_line(imgui.get_font_size() * 7.0)        # scales with the interface size
    imgui.text(v)


# ------------------------------------------------------------------------------------------------ info
def _sec_info(ui):
    app, st = ui.app, ui.st
    if not _open(ui, "info", "Info & Honesty"):
        return
    imgui.spacing()
    imgui.push_text_wrap_pos(0.0)
    text_c(theme.WARN, "Flow velocity is derived from the water-surface slope (Manning), not solver output.")
    imgui.pop_text_wrap_pos()
    imgui.spacing()
    label, full = source_info(app.storm)
    col = theme.OK if full else theme.WARN
    text_c(col, label)
    if not full:
        imgui.push_text_wrap_pos(0.0)
        dim("Depth is the smoothest water surface consistent with the solver's depth tiers: about 0.1 m accuracy, the "
            "severe tier is only known as >= 1.2 m.", wrap=True)
        imgui.pop_text_wrap_pos()
    ch, op = imgui.checkbox("Show source details", bool(ui.show_source))
    if ch:
        ui.show_source = bool(op)
    imgui.spacing()
    if imgui.button("Controls (F1)"):
        ui.show_help = True
    imgui.same_line()
    if imgui.button("About"):
        ui.show_about = True
    imgui.spacing()
