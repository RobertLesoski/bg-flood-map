"""Bottom timeline bar: transport + speed + hour slider and an implot mini chart (flooded buildings / wet area per hour)
with the current-time marker.  Click / drag in the chart to seek."""
from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui, implot

from . import bridge, theme, units
from .timeline import chart_series, value_at
from .theme import col32
from .widgets import bold, dim, icon_button, text_c

V2 = imgui.ImVec2
V4 = imgui.ImVec4


def draw_timeline(ui):
    from ..app import SPEEDS
    L, app, st = ui.L, ui.app, ui.st
    x0 = L.side_w
    y0 = L.H - L.foot - L.tl_h
    if not ui.begin_panel("##timeline", x0, y0, L.W - x0, L.tl_h, rounding=0.0,
                          padding=(14.0 * ui.s, 8.0 * ui.s)):
        ui.end_panel()
        return
    try:
        # ---- row 1: transport, speed, slider, label
        sz = 28.0 * ui.s
        if icon_button("##tprev", "prev", sz, tip="Previous hour  ["):
            app.t = max(0.0, math.ceil(app.t - 1.0 - 1e-6))
        imgui.same_line()
        if icon_button("##tplay", "pause" if app.playing else "play", sz, active=app.playing, tip="Play / pause  (Space)"):
            app.playing = not app.playing
        imgui.same_line()
        if icon_button("##tnext", "next", sz, tip="Next hour  ]"):
            app.t = min(24.0, math.floor(app.t + 1.0 + 1e-6))
        imgui.same_line()
        for i, sp in enumerate(SPEEDS):
            on = i == app.speed_idx
            imgui.push_style_color(imgui.Col_.button, V4(*theme.mix(theme.BG3, theme.ACCENT, 0.7 if on else 0.12)))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, V2(6.0 * ui.s, 0.0))
            if imgui.button(f"{sp:g}x##tsp{i}", V2(0, sz)):
                app.speed_idx = i
            imgui.pop_style_var()
            imgui.pop_style_color()
            imgui.same_line()
        if app.buffering:
            text_c(theme.WARN, "buffering")
            imgui.same_line()
        label = f"Hour {units.snapshot_hour(app.t)} of 24   t = {units.hour_label(app.t)}{units.hour_note(app)}"
        tw = imgui.calc_text_size(label).x
        avail = imgui.get_content_region_avail().x
        sl_w = max(avail - tw - 18.0 * ui.s, 120.0)
        imgui.set_next_item_width(sl_w)
        ch, t = imgui.slider_float("##tslider", app.t, 0.0, 24.0, "", imgui.SliderFlags_.always_clamp)
        if ch:
            app.t = float(t)
        imgui.same_line()
        bold(label, theme.TEXT)
        # ---- row 2: chart
        h = imgui.get_content_region_avail().y
        if h > 40.0:
            _chart(ui, h)
    finally:
        ui.end_panel()


def _chart(ui, h: float):
    app, st = ui.app, ui.st
    meta = getattr(app.storm, "meta", None)
    tl = bridge.timeline(app) if _timeline_matches(app) else None
    s = ui.cached(("chart", id(tl), (tl or {}).get("done"), app.storm.name), lambda: chart_series(tl, meta))
    mode = st.timeline_mode
    show_flooded = mode in ("flooded", "both") and s.has_flooded
    show_wet = mode in ("wet_area", "both") or not s.has_flooded
    wet_on_y2 = show_flooded and show_wet
    # mode switch (small chips on the right of the plot header would cost space: use the plot's own context menu-free combo)
    imgui.push_style_var(imgui.StyleVar_.frame_padding, V2(6.0 * ui.s, 1.0))
    dim("Chart")
    for i, (key, lab) in enumerate((("both", "Both"), ("flooded", "Flooded buildings"), ("wet_area", "Wet area"))):
        imgui.same_line()
        on = st.timeline_mode == key
        imgui.push_style_color(imgui.Col_.button, V4(*theme.mix(theme.BG3, theme.ACCENT, 0.6 if on else 0.1)))
        if imgui.button(lab + f"##tm{i}"):
            st.timeline_mode = key
        imgui.pop_style_color()
    imgui.same_line()
    if not s.complete:
        dim(f"   computing building counts {s.done}/25")
    imgui.pop_style_var()
    h = imgui.get_content_region_avail().y
    if h < 30:
        return
    C = implot.Col_
    for col, rgba in ((C.plot_bg, theme.with_alpha(theme.BG0, 0.55)), (C.frame_bg, (0, 0, 0, 0)),
                      (C.plot_border, theme.with_alpha(theme.BORDER, 0.8)), (C.axis_text, theme.TEXT_DIM),
                      (C.axis_grid, theme.with_alpha(theme.BORDER, 0.9)), (C.axis_tick, theme.TEXT_MUTE)):
        implot.push_style_color(col, V4(*rgba))
    implot.push_style_var(implot.StyleVar_.plot_padding, V2(4.0, 2.0))
    flags = (implot.Flags_.no_title | implot.Flags_.no_legend | implot.Flags_.no_menus | implot.Flags_.no_box_select
             | implot.Flags_.no_mouse_text)
    try:
        if implot.begin_plot("##timeline_plot", V2(-1, h), flags):
            AF = implot.AxisFlags_
            # tick step from the room per label (1366 px at 200 % scale fits only a few "00:00" labels)
            lab_w = imgui.calc_text_size("00:00").x * 1.6
            plot_w = max(imgui.get_content_region_avail().x - 90.0 * ui.s, 1.0)
            step = next((k for k in (3, 6, 12) if plot_w / (24 / k) >= lab_w), 24)
            ticks = np.arange(0, 25, step, dtype=np.float64)
            implot.setup_axis(implot.ImAxis_.x1, None, AF.no_highlight | AF.lock | AF.no_menus)
            implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, 24.0, implot.Cond_.always)
            implot.setup_axis_ticks(implot.ImAxis_.x1, ticks, [f"{int(v):02d}:00" for v in ticks])
            fmax = float(np.nanmax(s.flooded)) if s.has_flooded else 1.0
            wmax = float(np.nanmax(s.wet_area)) if np.isfinite(s.wet_area).any() else 1.0
            ymax1 = max(fmax if show_flooded else wmax, 1e-6) * 1.12
            implot.setup_axis(implot.ImAxis_.y1, None, AF.no_highlight | AF.lock | AF.no_menus)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, ymax1, implot.Cond_.always)
            implot.setup_axis_format(implot.ImAxis_.y1, "%.0f" if show_flooded else "%.1f")
            if wet_on_y2:
                implot.setup_axis(implot.ImAxis_.y2, None, AF.opposite | AF.no_grid_lines | AF.no_highlight | AF.lock | AF.no_menus)
                implot.setup_axis_limits(implot.ImAxis_.y2, 0.0, max(wmax, 1e-6) * 1.12, implot.Cond_.always)
                implot.setup_axis_format(implot.ImAxis_.y2, "%.0f km²")
            implot.setup_finish()
            if show_flooded:
                m = np.isfinite(s.flooded)
                xs, ys = s.hours[m], s.flooded[m]
                implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y1)
                implot.plot_shaded("flooded", xs, ys, 0.0, spec=implot.Spec(fill_color=V4(*theme.TIER_RGB[3]), fill_alpha=0.18))
                implot.plot_line("flooded", xs, ys, spec=implot.Spec(line_color=V4(*theme.TIER_RGB[3]), line_weight=2.2))
            if show_wet:
                m = np.isfinite(s.wet_area)
                xs, ys = s.hours[m], s.wet_area[m]
                implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2 if wet_on_y2 else implot.ImAxis_.y1)
                col = theme.ACCENT
                implot.plot_shaded("wet", xs, ys, 0.0, spec=implot.Spec(fill_color=V4(*col), fill_alpha=0.14))
                implot.plot_line("wet", xs, ys, spec=implot.Spec(line_color=V4(*col), line_weight=2.2))
            implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y1)
            # current-time marker
            implot.plot_inf_lines("now", np.array([min(max(app.t, 0.0), 24.0)]),
                                  spec=implot.Spec(line_color=V4(*theme.ACCENT2), line_weight=2.0))
            hovered = implot.is_plot_hovered()
            mp = implot.get_plot_mouse_pos()
            if hovered and imgui.is_mouse_clicked(0):
                ui.tl_drag = True
            if not imgui.is_mouse_down(0):
                ui.tl_drag = False
            if getattr(ui, "tl_drag", False):
                app.t = float(min(max(mp.x, 0.0), 24.0))
            if hovered and not getattr(ui, "tl_drag", False):
                hx = min(max(mp.x, 0.0), 24.0)
                dlp = implot.get_plot_draw_list()
                p = implot.plot_to_pixels(hx, 0.0)
                p2 = implot.plot_to_pixels(hx, ymax1)
                dlp.add_line(V2(p.x, p.y), V2(p2.x, p2.y), col32(theme.with_alpha(theme.TEXT, 0.35)), 1.0)
                _chart_tip(ui, s, hx, show_flooded, show_wet)
            implot.end_plot()
    finally:
        implot.pop_style_var()
        implot.pop_style_color(6)


def _timeline_matches(app) -> bool:
    tl = bridge.timeline(app)
    return tl is not None and tl.get("storm") == getattr(app.storm, "name", None)


def _chart_tip(ui, s, hx: float, show_flooded: bool, show_wet: bool):
    hr = int(round(hx))
    lines = [f"Hour {hr}  ({units.hour_label(hr)})"]
    if s.has_flooded:
        v = s.flooded[hr]
        lines.append("Flooded buildings: " + (f"{int(v):,}" if np.isfinite(v) else "computing"))
    if np.isfinite(s.wet_area[hr]):
        lines.append(f"Wet area: {s.wet_area[hr]:.2f} km²")
    if np.isfinite(s.max_depth[hr]):
        lines.append(f"Max depth: {s.max_depth[hr]:.2f} m")
    lines.append("click or drag to seek")
    imgui.set_tooltip("\n".join(lines))
