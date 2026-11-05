"""Overlay widgets over the 3D view: footer honesty line, legends, compass, scale bar, probe tooltip, performance."""
from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui

from . import bridge, theme, units
from .menus import source_info
from .theme import col32
from .widgets import bold, dim, pill, shadow_text, text_c, wrapped_lines

V2 = imgui.ImVec2
V4 = imgui.ImVec4

HONESTY = "Flow velocity is derived from the water-surface slope (Manning), not solver output."
HONESTY_SHORT = "Velocity is derived from the water-surface slope (Manning), not solver output."

OVERLAY_FLAGS = int(imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
                    | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings
                    | imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_scrollbar
                    | imgui.WindowFlags_.no_scroll_with_mouse)


# ------------------------------------------------------------------------------------------------ footer
def draw_footer(ui):
    L, app = ui.L, ui.app
    if not ui.begin_panel("##footer", 0.0, L.H - L.foot, L.W, L.foot, rounding=0.0,
                          padding=(12.0 * ui.s, 4.0 * ui.s), extra_flags=OVERLAY_FLAGS):
        ui.end_panel()
        return
    try:
        dl = imgui.get_window_draw_list()
        p = imgui.get_window_pos()
        cy = p.y + L.foot * 0.5
        fh = imgui.calc_text_size("Ag").y
        # info glyph
        cx = p.x + 20.0
        dl.add_circle_filled(V2(cx, cy), 7.0, col32(theme.with_alpha(theme.WARN, 0.2)))
        dl.add_circle(V2(cx, cy), 7.0, col32(theme.WARN), 0, 1.2)
        dl.add_text(V2(cx - 2.2, cy - fh * 0.5), col32(theme.WARN), "i")
        label, full = source_info(app.storm)
        right = "Source: " + label + "   (click: details)"
        rw = imgui.calc_text_size(right).x
        avail = L.W - 40.0 - rw - 40.0
        msg = HONESTY if imgui.calc_text_size(HONESTY).x <= avail else HONESTY_SHORT
        if imgui.calc_text_size(msg).x > avail:
            msg = "Velocity is derived (Manning), not solver output."
        dl.add_text(V2(cx + 14.0, cy - fh * 0.5), col32(theme.TEXT), msg)
        col = theme.OK if full else theme.WARN
        x = L.W - rw - 16.0
        dl.add_text(V2(x, cy - fh * 0.5), col32(col), right)
        imgui.set_cursor_screen_pos(V2(x, p.y))
        if imgui.invisible_button("##src", V2(rw, L.foot)):
            ui.show_source = not ui.show_source
    finally:
        ui.end_panel()


# ------------------------------------------------------------------------------------------------ legends
_GRAD_CACHE: dict = {}


def _gradient_colors(fn, vmax: float, n: int):
    """Packed colours of the n+1 gradient stops (computed once: the ramp functions are pure python and slow)."""
    key = (fn.__name__, vmax, n)
    cols = _GRAD_CACHE.get(key)
    if cols is None:
        cols = [col32(fn(vmax * i / n)) for i in range(n + 1)]
        _GRAD_CACHE[key] = cols
    return cols


def _gradient(dl, x, y, w, h, fn, vmax, n=24):
    cols = _gradient_colors(fn, vmax, n)
    for i in range(n):
        c0, c1 = cols[i], cols[i + 1]
        dl.add_rect_filled_multi_color(V2(x + w * i / n, y), V2(x + w * (i + 1) / n + 0.6, y + h), c0, c1, c1, c0)
    dl.add_rect(V2(x, y), V2(x + w, y + h), col32((1, 1, 1, 0.25)), 2.0, 1.0)


def _ticks(dl, x, y, w, vmax, stops, labels, color=theme.TEXT_DIM):
    for v, lab in zip(stops, labels):
        tx = x + w * v / vmax
        dl.add_line(V2(tx, y), V2(tx, y + 4.0), col32(color), 1.0)
        ts = imgui.calc_text_size(lab)
        px = min(max(tx - ts.x * 0.5, x - 2.0), x + w - ts.x + 6.0)
        dl.add_text(V2(px, y + 4.0), col32(color), lab)


DEPTH_STOPS_M = (0.0, 1.0, 2.0, 3.0, 4.0)
DEPTH_STOPS_FT = (0.0, 3.0, 6.0, 9.0, 12.0)
SPEED_MAX = 2.6


def legend_labels(imperial: bool):
    if imperial:
        return [units.ft_to_m(v) for v in DEPTH_STOPS_FT], [f"{v:g}" for v in DEPTH_STOPS_FT[:-1]] + [f"{DEPTH_STOPS_FT[-1]:g} ft"]
    return list(DEPTH_STOPS_M), [f"{v:g}" for v in DEPTH_STOPS_M[:-1]] + [f"{DEPTH_STOPS_M[-1]:g} m"]


def draw_legends(ui):
    L, st, app = ui.L, ui.st, ui.app
    if not st.show_legends:
        return
    s = ui.s
    w = 280.0 * s
    lh = imgui.get_text_line_height() + 2.0
    small = (L.sy1 - L.sy0) < 520.0 * s                 # short window: the legend starts collapsed, '+' opens it
    open_ = ui.legend_small_open if small else st.legend_open
    x = L.sx0 + 14.0 * s
    y = L.sy1 - 14.0 * s
    wp = w if open_ else 118.0 * s                      # collapsed: just the title chip (keeps clear of the scale bar)
    if not ui.begin_panel("##legend", x, y, wp, 0.0, padding=(12.0 * s, 8.0 * s), extra_flags=OVERLAY_FLAGS, bg_alpha=0.9,
                          auto_height=True, pivot=(0.0, 1.0)):
        ui.end_panel()
        return
    try:
        dl = imgui.get_window_draw_list()
        p0 = imgui.get_cursor_screen_pos()
        bold("Legend", theme.ACCENT2)
        imgui.same_line(wp - 36.0 * s)
        if imgui.small_button("-" if open_ else "+"):
            if small:
                ui.legend_small_open = not open_
            else:
                st.legend_open = not open_
        if not open_:
            return
        imgui.spacing()
        bar_w = w - 26.0 * s
        # ---- water depth
        dim("Water depth")
        p = imgui.get_cursor_screen_pos()
        bw = bar_w
        stops, labs = legend_labels(st.imperial)
        _gradient(dl, p.x, p.y, bw, 10.0 * s, theme.depth_ramp_srgb, DEPTH_STOPS_M[-1])
        _ticks(dl, p.x, p.y + 10.0 * s, bw, DEPTH_STOPS_M[-1], stops, labs)
        imgui.dummy(V2(bw, 10.0 * s + lh + 4.0))
        # ---- building tiers
        imgui.spacing()
        dim("Building flood tier  (flooded roads use the same colours)")
        p = imgui.get_cursor_screen_pos()
        ranges = [units.tier_range_text(k + 1) for k in range(4)]
        if st.imperial:
            ranges = ["0.3-1 ft", "1-2 ft", "2-3.9 ft", ">= 3.9 ft"]
        row = lh + 1.0
        for k in range(4):
            cy = p.y + k * row
            dl.add_rect_filled(V2(p.x, cy + 3), V2(p.x + 12.0 * s, cy + 3 + 11.0 * s), col32(theme.TIER_RGB[k]), 3.0)
            dl.add_text(V2(p.x + 20.0 * s, cy), col32(theme.TEXT), units.TIER_NAMES[k])
            tw = imgui.calc_text_size(ranges[k]).x
            dl.add_text(V2(p.x + bar_w - tw, cy), col32(theme.TEXT_DIM), ranges[k])
        imgui.dummy(V2(bar_w, 4 * row))
        # ---- flow speed
        imgui.spacing()
        dim("Flow speed (streaks / arrows)")
        p = imgui.get_cursor_screen_pos()
        _gradient(dl, p.x, p.y, bw, 10.0 * s, theme.speed_ramp, SPEED_MAX)
        if st.imperial:
            sp_stops = [units.mph_to_ms(v) for v in (0.0, 1.0, 2.0, 4.0)]
            sp_labs = ["0", "1", "2", "4 mph"]
        else:
            sp_stops = [0.0, 0.5, 1.0, 1.5, 2.0]
            sp_labs = ["0", "0.5", "1", "1.5", "2 m/s"]
        _ticks(dl, p.x, p.y + 10.0 * s, bw, SPEED_MAX, sp_stops, sp_labs)
        imgui.dummy(V2(bw, 10.0 * s + lh + 4.0))
        dim("derived from surface slope")
    finally:
        ui.end_panel()


# ------------------------------------------------------------------------------------------------ compass
def draw_compass(ui):
    L, st, app = ui.L, ui.st, ui.app
    if not st.show_compass:
        return
    s = ui.s
    size = 84.0 * s
    x = L.sx0 + 14.0 * s
    y = L.sy0 + 14.0 * s
    if not ui.begin_panel("##compass", x, y, size, size + 26.0 * s, padding=(0.0, 0.0), extra_flags=OVERLAY_FLAGS,
                          bg_alpha=0.0, border=False):
        ui.end_panel()
        return
    try:
        dl = imgui.get_window_draw_list()
        p = imgui.get_window_pos()
        c = V2(p.x + size * 0.5, p.y + size * 0.5)
        r = size * 0.5 - 4.0
        yaw = float(app.camera.yaw)
        ang = math.radians(units.north_angle(yaw))        # clockwise from screen-up
        dl.add_circle_filled(c, r, col32((0.04, 0.06, 0.11, 0.78)), 48)
        dl.add_circle(c, r, col32((1, 1, 1, 0.28)), 48, 1.4)
        for k, lab in enumerate(("N", "E", "S", "W")):
            a = ang + math.radians(90 * k)
            dx, dy = math.sin(a), -math.cos(a)
            if k:
                dl.add_line(V2(c.x + dx * (r - 2), c.y + dy * (r - 2)), V2(c.x + dx * (r - 7 * s), c.y + dy * (r - 7 * s)),
                            col32((1, 1, 1, 0.5)), 1.4)
        # needle: red north half, light south half
        nx, ny = math.sin(ang), -math.cos(ang)
        px, py = -ny, nx
        w = 7.0 * s
        tip = V2(c.x + nx * (r - 6 * s), c.y + ny * (r - 6 * s))
        tail = V2(c.x - nx * (r - 6 * s), c.y - ny * (r - 6 * s))
        dl.add_triangle_filled(tip, V2(c.x + px * w, c.y + py * w), V2(c.x - px * w, c.y - py * w), col32(theme.hexrgb("#ff5a4d")))
        dl.add_triangle_filled(tail, V2(c.x + px * w, c.y + py * w), V2(c.x - px * w, c.y - py * w), col32((0.85, 0.9, 0.98, 0.9)))
        dl.add_circle_filled(c, 2.2 * s, col32((0.05, 0.07, 0.12, 1)))
        # N label just outside the needle tip
        lt = imgui.calc_text_size("N")
        lx = c.x + nx * (r - 17 * s) - lt.x * 0.5
        ly = c.y + ny * (r - 17 * s) - lt.y * 0.5
        dl.add_text(V2(lx, ly), col32((1, 1, 1, 1)), "N")
        imgui.set_cursor_screen_pos(V2(p.x, p.y))
        if imgui.invisible_button("##north", V2(size, size)):
            ui.look_north()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Click to face north")
        hd = f"{yaw:.0f}° {units.cardinal(yaw, 8)}"
        tw = imgui.calc_text_size(hd).x
        shadow_text(dl, (p.x + size * 0.5 - tw * 0.5, p.y + size + 1.0), (1, 1, 1, 0.9), hd)
    finally:
        ui.end_panel()


# ------------------------------------------------------------------------------------------------ scale bar
def scale_bar_spec(app, s: float, target_px: float = 130.0):
    """-> (px_width_metric, label_metric, px_width_imperial, label_imperial) at the camera target."""
    cam = app.camera
    mpp = units.metres_per_pixel(float(cam.dist), float(cam.fov), float(cam.height))
    tm = target_px * s * mpp
    lm, lab_m = units.nice_length(tm, False)
    li, lab_i = units.nice_length(tm, True)
    return lm / mpp, lab_m, li / mpp, lab_i


def draw_scalebar(ui):
    L, st, app = ui.L, ui.st, ui.app
    if not st.show_scalebar:
        return
    s = ui.s
    wm, lm, wi, li = scale_bar_spec(app, s)
    right = L.sx1 - 16.0 * s
    base = L.sy1 - 18.0 * s
    fh = imgui.calc_text_size("Ag").y
    pad = 8.0 * s
    bw = max(wm, wi)
    box0 = V2(right - bw - 2 * pad, base - 2 * fh - 22.0 * s)
    box1 = V2(right + 2.0, base + 8.0 * s)
    # a real (input-less) window, not the foreground draw list, so dialogs and tooltips stay above it
    if not ui.begin_panel("##scalebar", box0.x, box0.y, box1.x - box0.x, box1.y - box0.y, padding=(0.0, 0.0), border=False,
                          bg_alpha=0.0, extra_flags=OVERLAY_FLAGS | int(imgui.WindowFlags_.no_inputs)):
        ui.end_panel()
        return
    try:
        _scalebar_body(imgui.get_window_draw_list(), box0, box1, right, base, fh, pad, wm, wi, lm, li, s)
    finally:
        ui.end_panel()


def _scalebar_body(dl, box0, box1, right, base, fh, pad, wm, wi, lm, li, s):
    dl.add_rect_filled(box0, box1, col32((0.04, 0.06, 0.11, 0.62)), 6.0)
    x1 = right - pad
    x0m = x1 - wm
    x0i = x1 - wi
    ym = base - fh - 6.0 * s
    col = col32((1, 1, 1, 0.95))
    # metric above the line, imperial below
    dl.add_line(V2(x0m, ym), V2(x1, ym), col, 2.0)
    for xx in (x0m, x1):
        dl.add_line(V2(xx, ym - 5.0 * s), V2(xx, ym + 5.0 * s), col, 2.0)
    for xx in (x0i, x0i + wi):
        dl.add_line(V2(xx, ym), V2(xx, ym + 5.0 * s), col, 2.0)
    dl.add_line(V2(x0i, ym), V2(x0i + wi, ym), col, 2.0)
    tm = imgui.calc_text_size(lm)
    ti = imgui.calc_text_size(li)
    dl.add_text(V2(x1 - tm.x, ym - 6.0 * s - tm.y), col, lm)
    dl.add_text(V2(x1 - ti.x, ym + 7.0 * s), col32((1, 1, 1, 0.75)), li)


# ------------------------------------------------------------------------------------------------ probe tooltip
def draw_probe(ui):
    st, app = ui.st, ui.app
    if not st.show_probe or ui.mouse is None or ui.ground_pick is None:
        return
    pr = ui.probe_worker.latest
    mx, my = ui.mouse
    s = ui.s
    io = imgui.get_io()
    W, H = io.display_size.x, io.display_size.y
    flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
             | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_saved_settings
             | imgui.WindowFlags_.no_focus_on_appearing | imgui.WindowFlags_.no_nav | imgui.WindowFlags_.no_inputs)
    ox = 20.0 * s
    oy = 22.0 * s
    wide = 360.0 if (pr is not None and getattr(pr, 'tile_hour', None) is not None) else 270.0
    flip_x = mx + ox + wide * s > W
    flip_y = my + oy + (195.0 if wide > 300.0 else 150.0) * s > H
    imgui.set_next_window_pos(V2(mx - ox if flip_x else mx + ox, my - oy if flip_y else my + oy),
                              imgui.Cond_.always, V2(1.0 if flip_x else 0.0, 1.0 if flip_y else 0.0))
    imgui.set_next_window_bg_alpha(0.94)
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(11.0 * s, 8.0 * s))
    try:
        vis, _ = imgui.begin("##probe", None, flags)
        if vis:
            try:
                _probe_body(ui, pr, imgui.get_window_draw_list())
            finally:
                pass
        imgui.end()
    finally:
        imgui.pop_style_var()


def _probe_body(ui, pr, dl):
    app = ui.app
    s = ui.s
    ground = float(pr.ground_m) if pr is not None and math.isfinite(pr.ground_m) else float(ui.ground_pick[2] / max(app.exag, 1e-6))
    if pr is None:
        dim("sampling...")
        return
    if not pr.valid:
        text_c(theme.TEXT_DIM, "Outside the model domain")
        dim(f"Ground elevation  {units.elevation_text(ground)}")
        return
    d = pr.depth_m
    has_tiles = _tile_block(ui, pr, dl)
    grid_label = "Grid depth  " if has_tiles else "Water depth  "
    if has_tiles and d < 0.02:
        dim("Flow-grid depth   dry (below 2 cm)")
    if d >= 0.02:
        p = imgui.get_cursor_screen_pos()
        col = theme.depth_ramp_srgb(d)
        dl.add_rect_filled(V2(p.x, p.y + 2), V2(p.x + 12 * s, p.y + 14 * s), col32(col), 3.0)
        imgui.set_cursor_screen_pos(V2(p.x + 18 * s, p.y))
        approx = app.storm.source == "tiers"
        bold(grid_label, theme.TEXT_DIM)
        imgui.same_line()
        if approx and d >= units.SEVERE_EDGE_M:        # open-ended severe tier: only the lower bound is known
            bold(units.depth_text(d, tier=4, approx=True), theme.TEXT)
            dim("estimate")
            imgui.same_line(84.0 * s)
            dim(f"model {d:.2f} m, tiers only certify >= 1.2 m")
        else:
            bold(units.depth_text(d, approx=approx), theme.TEXT)
        sp = pr.speed_ms
        if sp >= 0.02:
            dim("Flow")
            imgui.same_line(84.0 * s)
            q = imgui.get_cursor_screen_pos()
            _arrow(dl, q.x + 10 * s, q.y + 8 * s, 9 * s, units.screen_arrow(pr.u, pr.v, float(app.camera.yaw)))
            imgui.dummy(V2(22 * s, 16 * s))
            imgui.same_line()
            imgui.text(f"{units.speed_text(sp)}")
            dim("")
            imgui.same_line(84.0 * s)
            dim(f"toward {units.cardinal(pr.bearing)}  ({pr.bearing:.0f}°)")
        else:
            dim("Flow      still water")
    else:
        text_c(theme.TEXT_DIM, "Dry ground")
    dim("Ground")
    imgui.same_line(84.0 * s)
    imgui.text(units.elevation_text(ground))
    b = bridge.buildings(app)
    bid = ui.hover_building
    if b is not None and bid is not None and 0 <= bid < b.n:
        imgui.separator()
        fl = bridge.flood(app)
        bs = fl.get("buildings") if fl else None
        addr = str(b.address[bid]).strip() or f"Building #{bid}"
        bold(addr[:38], theme.ACCENT2)
        if bs is not None:
            tier = int(bs.tier[bid])
            dep = float(bs.depth_m[bid])
            approx = bool(fl and fl.get("source") == "tiers")
            if tier:
                text_c(theme.tier_color(tier), f"{units.TIER_NAMES[tier - 1]}   {units.depth_text(dep, tier=tier, approx=approx)}")
            else:
                dim("not flooded at this hour")
        dim("click to select")
    if d >= 0.02 and pr.speed_ms >= 0.02:
        dim("velocity derived from surface slope")


def _tile_text(smp, imperial: bool = False) -> str:
    """One value of the solver tiles as text: 'no water' and 'no data' are different things."""
    if smp is None:
        return "-"
    if smp.status == "nodata":
        return "no data (outside the modelled area)"
    if smp.status == "unavailable":
        return smp.note or "not available"
    if smp.status == "dry":
        return "no water (0 cm)"
    return f"{smp.cm:.0f} cm  ({smp.ft:.2f} ft)"


def _tile_block(ui, pr, dl) -> bool:
    """Solver depth under the cursor from the native tiles (peak of the run + nearest hour file).  False when the storm has no
    tiles or they are switched off (then the tooltip is exactly the old one)."""
    pk, hr = getattr(pr, "tile_peak", None), getattr(pr, "tile_hour", None)
    if pk is None and hr is None:
        return False
    s = ui.s
    ref = hr if hr is not None else pk
    bold("Solver depth  ", theme.TEXT_DIM)
    imgui.same_line()
    col = theme.TEXT
    if hr is not None and hr.status in ("nodata", "unavailable"):
        col = theme.TEXT_DIM
    bold(_tile_text(hr), col)
    dim(f"hour {hr.hour}" if hr is not None and hr.hour is not None else "hour")
    imgui.same_line(84.0 * s)
    if pk is not None and pk.status in ("depth", "dry"):
        imgui.text("peak " + _tile_text(pk))
    else:
        dim("peak " + _tile_text(pk))
    dim(ref.label + (f", {ref.scope} run" if ref.scope == "county" and ui.app.domain.key == "city" else ""))
    imgui.spacing()
    return True


def _arrow(dl, cx, cy, r, d):
    dx, dy = d
    if dx == 0 and dy == 0:
        dl.add_circle_filled(V2(cx, cy), 2.5, col32(theme.TEXT_DIM))
        return
    col = col32(theme.ACCENT2)
    tip = V2(cx + dx * r, cy + dy * r)
    tail = V2(cx - dx * r * 0.9, cy - dy * r * 0.9)
    dl.add_line(tail, tip, col, 2.2)
    px, py = -dy, dx
    hb = V2(cx + dx * r * 0.25, cy + dy * r * 0.25)
    dl.add_triangle_filled(tip, V2(hb.x + px * r * 0.55, hb.y + py * r * 0.55), V2(hb.x - px * r * 0.55, hb.y - py * r * 0.55), col)


# ------------------------------------------------------------------------------------------------ performance (F3)
def draw_perf(ui):
    st, app, L = ui.st, ui.app, ui.L
    if not st.show_perf:
        return
    s = ui.s
    w = 316.0 * s
    h = 300.0 * s
    x = L.sx1 - w - 14.0 * s
    y = L.sy0 + 14.0 * s
    if not ui.begin_panel("##perf", x, y, w, h, padding=(12.0 * s, 8.0 * s), extra_flags=OVERLAY_FLAGS, bg_alpha=0.88):
        ui.end_panel()
        return
    try:
        fm = np.array(app.frame_ms, np.float32) if len(app.frame_ms) else np.array([16.7], np.float32)
        bold("Performance  (F3)", theme.ACCENT2)
        imgui.same_line(w - 76.0 * s)
        dim(f"{app.fps:.0f} fps")
        gpu = "n/a" if app.gpu_ms is None else f"{app.gpu_ms:.2f} ms"
        mean, p99, mx = float(fm.mean()), float(np.percentile(fm, 99)), float(fm.max())
        imgui.text(f"frame {mean:.1f} ms   p99 {p99:.1f}   max {mx:.1f}")
        top = max(33.4, min(float(fm.max()) * 1.1, 100.0))
        imgui.push_style_color(imgui.Col_.plot_lines, V4(*theme.ACCENT2))
        imgui.plot_lines("##ft", fm[-180:], 0, None, 0.0, top, V2(w - 24.0 * s, 54.0 * s))
        imgui.pop_style_color()
        # 16.7 ms guide
        p = imgui.get_item_rect_min()
        q = imgui.get_item_rect_max()
        gy = q.y - (q.y - p.y) * (16.7 / top)
        imgui.get_window_draw_list().add_line(V2(p.x, gy), V2(q.x, gy), col32(theme.with_alpha(theme.OK, 0.5)), 1.0)
        dim(f"GPU scene pass {gpu}")
        ui_ms = sum(ui.panel_ms.values())
        dim(f"interface build {ui_ms:.1f} ms   refresh: " + ("half rate (auto, below 58 fps)" if ui.lite else "every frame"))
        flow = getattr(app, "flow", None)
        if flow is not None:
            dim(f"texture upload {flow.last_pump_ms:.2f} ms (max {flow.max_pump_ms:.1f})")
        hub = bridge.hub(app)
        if hub is not None:
            dim(f"stats hook {hub.last_call_ms:.2f} ms   snapshots computed: {len(hub.compute_ms)}")
        tm = getattr(app, "terrain_mesh", None)
        if tm is not None:
            dim(f"terrain chunks {tm.n_visible}/{tm.nx * tm.ny}   tris {tm.tris / 1e6:.2f} M   water {tm.tris_water / 1e6:.2f} M")
        dim(f"VRAM ~{app.ctx.vram_mb():.0f} MB   (tracked)")
        for k, v in list(app.ctx.vram_breakdown().items())[:3]:
            dim(f"   {k}: {v:.0f} MB")
        dim(f"{app.ctx.adapter_name}  {app.ctx.backend}", wrap=True)
        if not app.ctx.has_timestamps:
            dim("(GPU timestamp queries unavailable)")
    finally:
        ui.end_panel()
