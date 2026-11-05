"""Analysis tools: the floating Analysis window (Measure / Profile / Point / Timeline / Compare), the map overlays (measure lines and
labels, the cross-section line, the picked point, the swipe divider and the difference legend) and the click / key handling of the
map tools.  State and background work: `bgv.analysis.layer.AnalysisLayer`  (`app.get_layer("analysis")`).

Hooks (all in shared files, see docs/ANALYSIS_HOOKS.md):  menus.draw -> `menu(ui)`, UiManager.frame -> `draw(ui)` / `draw_hidden(ui)`,
UiManager._click -> `handle_click(ui, x, y)`, UiManager.on_key -> `on_key(ui, key)`.
"""
from __future__ import annotations

import math
import time

import numpy as np
from imgui_bundle import imgui, implot

from ..analysis import compare as cmpmod
from ..analysis import geodesy
from ..analysis.layer import COMPARE_MODES, TABS
from . import bridge, theme, units
from .theme import col32
from .widgets import bold, dim, hbar, pill, shadow_text, text_c, wrapped_lines

V2 = imgui.ImVec2
V4 = imgui.ImVec4

TAB_LABELS = {"measure": "Measure", "profile": "Profile", "point": "Point", "timeline": "Timeline", "compare": "Compare"}
TOOL_FOR_TAB = {"measure": "measure_line", "profile": "profile", "point": "point"}
C_MEASURE = theme.hexrgb("#ffd43b")
C_PROFILE = theme.hexrgb("#ff922b")
C_POINT = theme.hexrgb("#69db7c")
C_GROUND = theme.hexrgb("#a68a64")
C_WATER = theme.hexrgb("#4dabf7")
C_A = theme.hexrgb("#74c0fc")
C_B = theme.hexrgb("#ffa94d")
DOUBLE_CLICK_S = 0.4


def layer(ui):
    app = ui.app
    an = getattr(app, "analysis", None)
    return an if an is not None and getattr(an, "cmp", None) is not None else None


# ---------------------------------------------------------------------------------------------------- projection
class Proj:
    """World metres -> logical window pixels for the current camera (one view-projection per frame)."""

    def __init__(self, app):
        cam = app.camera
        self.vp = np.asarray(cam.view_proj(), np.float64)
        self.w, self.h = float(cam.width), float(cam.height)
        self.exag = float(app.exag)
        self.sampler = app.terrain.sampler

    def screen(self, xy, lift: float = 1.0):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        if xy.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros(0, bool)
        z = np.asarray(self.sampler.sample(xy[:, 0], xy[:, 1]), np.float64) * self.exag + lift
        P = np.column_stack([xy, z, np.ones(len(xy))]) @ self.vp.T
        w = P[:, 3]
        ok = w > 1e-6
        ww = np.where(ok, w, 1.0)
        px = (P[:, 0] / ww * 0.5 + 0.5) * self.w
        py = (0.5 - P[:, 1] / ww * 0.5) * self.h
        return np.column_stack([px, py]), ok


def drape(xy, step_m: float = 30.0, max_per_seg: int = 80):
    """Insert points along the segments so a line follows the terrain on screen."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    if len(xy) < 2:
        return xy
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = int(min(max(math.ceil(math.hypot(*(b - a)) / step_m), 1), max_per_seg))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.append(a + (b - a) * t)
    return np.vstack(out)


def _polyline(dl, pts, ok, color, width, outline=None):
    """Draw runs of consecutive visible points."""
    n = len(pts)
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= 2:
            seg = [V2(float(x), float(y)) for x, y in pts[i:j]]
            if outline is not None:
                dl.add_polyline(seg, col32(outline), width + 3.0, 0)
            dl.add_polyline(seg, col32(color), width, 0)
        i = j


def _marker(dl, p, color, r=5.0, fill=True):
    c = V2(float(p[0]), float(p[1]))
    dl.add_circle_filled(c, r + 1.8, col32((0.02, 0.03, 0.06, 0.85)))
    dl.add_circle_filled(c, r, col32(color if fill else (1, 1, 1, 1)))
    dl.add_circle(c, r, col32((1, 1, 1, 0.95)), 0, 1.5)


def _label(dl, p, text, fg=theme.TEXT, bg=(0.04, 0.06, 0.11, 0.86), dx=10.0, dy=-24.0):
    pill(dl, float(p[0]) + dx, float(p[1]) + dy, text, fg, bg, (7.0, 2.0), 7.0)


# ---------------------------------------------------------------------------------------------------- map clicks
def ground_pick(ui, x, y):
    pk = ui.app.camera.pick(x, y)
    return None if pk is None else (float(pk[0]), float(pk[1]))


def tool_active(ui) -> bool:
    an = layer(ui)
    return bool(an is not None and an.tool != "none")


def handle_click(ui, x: float, y: float) -> bool:
    """A click on the 3D view while a map tool is active: add a vertex / pick the target.  True = consumed."""
    an = layer(ui)
    if an is None or an.tool == "none":
        return False
    now = time.perf_counter()
    last = getattr(ui, "_an_last_click", None)
    ui._an_last_click = (x, y, now)
    dbl = last is not None and now - last[2] < DOUBLE_CLICK_S and math.hypot(x - last[0], y - last[1]) < 8.0
    tool = an.tool
    if tool == "point":
        b = bridge.hover_building(ui.app, x, y)
        if b is not None:
            an.set_target_building(int(b))
            ui.toasts.push(f"Depth series for building {int(b)}", theme.OK, 2.5)
        else:
            p = ground_pick(ui, x, y)
            if p is not None:
                lng, lat = geodesy.world_to_lnglat(an.frame, p[0], p[1])
                an.set_target_point(float(lng), float(lat))
        an.st.tab = "point"
        an.st.open = True
        return True
    p = ground_pick(ui, x, y)
    if p is None:
        return True
    if tool in ("measure_line", "measure_area"):
        if dbl and an.measures and not an.measures[-1].done:
            an.finish_measure()
            return True
        an.add_measure_point(*p)
        return True
    if tool == "profile":
        if dbl and len(an.profile_line) >= 2:
            an.tool = "none"
            return True
        an.profile_line.append(p)
        if len(an.profile_line) >= 2:
            an.request_profile()
        return True
    return True


def on_key(ui, k: str) -> bool:
    an = layer(ui)
    if an is None:
        return False
    if k == "Escape":
        if an.tool != "none":
            if an.tool in ("measure_line", "measure_area") and an.measures and not an.measures[-1].done:
                an.measures.pop()
            an.tool = "none"
            return True
        return False
    if an.tool == "none":
        return False
    if k in ("Enter", "Return"):
        if an.tool in ("measure_line", "measure_area"):
            an.finish_measure()
        else:
            an.tool = "none"
        return True
    if k == "Backspace":
        if an.tool in ("measure_line", "measure_area") and an.measures and not an.measures[-1].done:
            m = an.measures[-1]
            if m.pts:
                m.pts.pop()
            if not m.pts:
                an.measures.pop()
        elif an.tool == "profile" and an.profile_line:
            an.profile_line.pop()
            if len(an.profile_line) >= 2:
                an.request_profile()
            else:
                an.profile = None
        return True
    return False


# ---------------------------------------------------------------------------------------------------- menu
def menu(ui):
    an = layer(ui)
    if an is None:
        return
    if not imgui.begin_menu("Analysis"):
        return
    try:
        st = an.st

        def tool_item(label, tool, tab):
            if imgui.menu_item(label, "", an.tool == tool)[0]:
                an.tool = "none" if an.tool == tool else tool
                if an.tool != "none":
                    st.tab, st.open = tab, True
        tool_item("Measure distance", "measure_line", "measure")
        tool_item("Measure area", "measure_area", "measure")
        tool_item("Elevation profile (draw a line)", "profile", "profile")
        tool_item("Depth series at a point / building", "point", "point")
        if imgui.menu_item("Clear measurements", "", False)[0]:
            an.clear_measures()
        imgui.separator()
        if imgui.menu_item("Flood timeline", "", False)[0]:
            st.tab, st.open = "timeline", True
        if imgui.begin_menu("Compare storms"):
            for key, lab in (("off", "Off"), ("swipe", "Swipe (storm B on one side)"), ("diff", "Difference (B minus A)")):
                if imgui.menu_item(lab, "", st.mode == key)[0]:
                    st.mode = key
                    if key != "off":
                        st.tab, st.open = "compare", True
            imgui.end_menu()
        imgui.separator()
        _, st.open = imgui.menu_item("Analysis window", "", st.open)
        imgui.end_menu()
    except Exception:
        imgui.end_menu()
        raise


# ---------------------------------------------------------------------------------------------------- frame entry points
def _set_view_rect(ui, an):
    L = ui.L
    W, H = max(L.W, 1.0), max(L.H, 1.0)
    an.cmp.view_rect = (L.sx0 / W, L.sy0 / H, L.sx1 / W, L.sy1 / H)
    an.cmp.win_size = (W, H)


def draw_hidden(ui):
    an = layer(ui)
    if an is not None:
        _set_view_rect(ui, an)


def draw(ui):
    an = layer(ui)
    if an is None:
        return
    _set_view_rect(ui, an)
    an.poll_profile()
    _hover_world(ui, an)
    _draw_map_overlay(ui, an)
    _draw_compare_overlay(ui, an)
    if an.st.open:
        _draw_window(ui, an)


def _hover_world(ui, an):
    gp = ui.ground_pick if ui.mouse is not None else None
    an.hover_world = None if gp is None else (float(gp[0]), float(gp[1]))


# ---------------------------------------------------------------------------------------------------- map overlay
def _draw_map_overlay(ui, an):
    L, app = ui.L, ui.app
    dl = imgui.get_background_draw_list()
    dl.push_clip_rect(V2(L.sx0, L.sy0), V2(L.sx1, L.sy1), True)
    try:
        pr = Proj(app)
        frame = an.frame
        imp = an.st.imperial
        s = ui.s
        # ---- measurements
        for mi, m in enumerate(an.measures):
            if not m.pts:
                continue
            xy = np.asarray(m.pts, np.float64)
            live = (not m.done) and an.hover_world is not None and an.tool in ("measure_line", "measure_area")
            pts = xy if not live else np.vstack([xy, np.asarray(an.hover_world)[None]])
            closed = m.kind == "area" and (m.done or len(pts) >= 3)
            path = np.vstack([pts, pts[:1]]) if closed else pts
            sp, ok = pr.screen(drape(path))
            if closed and len(pts) >= 3:
                vs, vok = pr.screen(pts)
                if vok.all():
                    try:
                        dl.add_concave_poly_filled([V2(float(a), float(b)) for a, b in vs], col32(theme.with_alpha(C_MEASURE, 0.17)))
                    except Exception:
                        pass
            _polyline(dl, sp, ok, C_MEASURE, 2.4 * s, outline=(0.02, 0.03, 0.06, 0.55))
            vs, vok = pr.screen(xy)
            for p, o in zip(vs, vok):
                if o:
                    _marker(dl, p, C_MEASURE, 4.5 * s)
            # segment labels (length of each segment) and the running total at the last point
            lng, lat = geodesy.world_to_lnglat(frame, pts[:, 0], pts[:, 1])
            seg = geodesy.segment_lengths_m(lng, lat)
            ps, pok = pr.screen(pts)
            for k, d in enumerate(seg):
                if pok[k] and pok[k + 1] and len(seg) <= 40:
                    mid = (ps[k] + ps[k + 1]) * 0.5
                    if abs(ps[k] - ps[k + 1]).max() > 46.0:
                        _label(dl, mid, geodesy.fmt_length(float(d), imp)[0], theme.TEXT, dx=-20.0, dy=-9.0)
            if m.kind == "line" and len(pts) >= 2 and pok[-1]:
                tot = float(seg.sum())
                _label(dl, ps[-1], "Total " + geodesy.fmt_length(tot, imp)[0], C_MEASURE, dx=12.0, dy=-28.0)
            if m.kind == "area" and len(xy) >= 3:
                cen = xy.mean(axis=0)
                cs, cok = pr.screen(cen[None])
                if cok[0]:
                    a2 = geodesy.polygon_area_m2(*geodesy.world_to_lnglat(frame, pts[:, 0], pts[:, 1])) if closed else 0.0
                    _label(dl, cs[0], geodesy.fmt_area(a2, imp)[0], C_MEASURE, dx=-30.0, dy=-10.0)
        # ---- profile line
        if an.profile_line:
            xy = np.asarray(an.profile_line, np.float64)
            pts = xy if (an.tool != "profile" or an.hover_world is None) else np.vstack([xy, np.asarray(an.hover_world)[None]])
            sp, ok = pr.screen(drape(pts))
            _polyline(dl, sp, ok, C_PROFILE, 2.6 * s, outline=(0.02, 0.03, 0.06, 0.55))
            vs, vok = pr.screen(xy)
            for k, (p, o) in enumerate(zip(vs, vok)):
                if o:
                    _marker(dl, p, C_PROFILE, 4.5 * s)
            if vok[0]:
                _label(dl, vs[0], "A", C_PROFILE, dx=8.0, dy=-24.0)
            if len(xy) >= 2 and vok[-1]:
                _label(dl, vs[-1], "B", C_PROFILE, dx=8.0, dy=-24.0)
            d = getattr(an, "hover_profile_d", None)
            if d is not None and an.profile is not None:
                r = an.profile.at(d)
                hs, hok = pr.screen(np.array([[r["x"], r["y"]]]))
                if hok[0]:
                    _marker(dl, hs[0], (1, 1, 1, 1), 5.5 * s)
        # ---- picked point / building
        t = an.target
        if t is not None:
            lng, lat = float(np.mean(t.lng_t)), float(np.mean(t.lat_t))
            X, Y = geodesy.lnglat_to_world(frame, lng, lat)
            ts, tok = pr.screen(np.array([[float(X), float(Y)]]), lift=3.0)
            if tok[0]:
                c = V2(float(ts[0][0]), float(ts[0][1]))
                dl.add_circle(c, 11.0 * s, col32(theme.with_alpha(C_POINT, 0.9)), 0, 2.4)
                dl.add_circle_filled(c, 3.2 * s, col32(C_POINT))
                _label(dl, ts[0], t.label[:34], C_POINT, dx=14.0, dy=-26.0)
        # ---- cursor hint of the active tool
        if an.tool != "none" and ui.mouse is not None:
            hint = {"measure_line": "Click to add points - double-click / Enter to finish - Esc cancels",
                    "measure_area": "Click to add corners - double-click / Enter to close - Esc cancels",
                    "profile": "Click to draw the cross-section line - double-click / Enter to finish - Esc cancels",
                    "point": "Click a building or the ground to see its depth over time - Esc ends"}[an.tool]
            mx, my = ui.mouse
            shadow_text(dl, (mx + 18.0, my + 18.0), theme.TEXT, hint)
            dl.add_circle(V2(mx, my), 7.0, col32((1, 1, 1, 0.9)), 0, 1.4)
    finally:
        dl.pop_clip_rect()


# ---------------------------------------------------------------------------------------------------- compare overlay
def _storm_name(app, name: str) -> str:
    try:
        return app.storm_label(name)
    except Exception:
        return name


def _draw_compare_overlay(ui, an):
    st = an.st
    if st.mode == "off":
        return
    cc = an.cmp
    app, L, s = ui.app, ui.L, ui.s
    a, ha = cc.a_state(app)
    b, hb = cc.b_state(app)
    dl = imgui.get_background_draw_list()                 # under the windows: the divider never cuts through a panel
    dl.push_clip_rect(V2(L.sx0, L.sy0), V2(L.sx1, L.sy1), True)
    try:
        la = f"A  {_storm_name(app, a)}   hour {ha}"
        lb = f"B  {_storm_name(app, b)}   hour {hb}"
        if st.mode == "swipe":
            if st.vertical:
                x = L.sx0 + st.swipe * (L.sx1 - L.sx0)
                dl.add_line(V2(x, L.sy0), V2(x, L.sy1), col32((1, 1, 1, 0.95)), 2.0)
                gy = (L.sy0 + L.sy1) * 0.5
                dl.add_circle_filled(V2(x, gy), 15.0 * s, col32((0.05, 0.07, 0.13, 0.95)))
                dl.add_circle(V2(x, gy), 15.0 * s, col32((1, 1, 1, 0.95)), 0, 2.0)
                for sgn in (-1, 1):
                    dl.add_triangle_filled(V2(x + sgn * 4.0 * s, gy - 6.0 * s), V2(x + sgn * 4.0 * s, gy + 6.0 * s),
                                           V2(x + sgn * 10.0 * s, gy), col32((1, 1, 1, 1)))
                wa = imgui.calc_text_size(la).x
                pill(dl, x - wa - 30.0, L.sy0 + 12.0 * s, la, C_A, (0.04, 0.06, 0.11, 0.9), (9.0, 4.0), 8.0, border=C_A)
                pill(dl, x + 14.0, L.sy0 + 12.0 * s, lb, C_B, (0.04, 0.06, 0.11, 0.9), (9.0, 4.0), 8.0, border=C_B)
            else:
                y = L.sy0 + st.swipe * (L.sy1 - L.sy0)
                dl.add_line(V2(L.sx0, y), V2(L.sx1, y), col32((1, 1, 1, 0.95)), 2.0)
                gx = (L.sx0 + L.sx1) * 0.5
                dl.add_circle_filled(V2(gx, y), 15.0 * s, col32((0.05, 0.07, 0.13, 0.95)))
                dl.add_circle(V2(gx, y), 15.0 * s, col32((1, 1, 1, 0.95)), 0, 2.0)
                pill(dl, L.sx0 + 70.0 * s, y - 30.0 * s, la, C_A, (0.04, 0.06, 0.11, 0.9), (9.0, 4.0), 8.0, border=C_A)
                pill(dl, L.sx0 + 70.0 * s, y + 8.0 * s, lb, C_B, (0.04, 0.06, 0.11, 0.9), (9.0, 4.0), 8.0, border=C_B)
    finally:
        dl.pop_clip_rect()
    if st.mode == "swipe":
        _swipe_handle(ui, an)
    elif st.mode == "diff":
        _diff_legend(ui, an, a, ha, b, hb)


def _swipe_handle(ui, an):
    """An invisible, draggable strip over the divider (capturing the mouse, so the camera does not pan)."""
    L, s, st = ui.L, ui.s, an.st
    wid = 26.0 * s
    if st.vertical:
        x = L.sx0 + st.swipe * (L.sx1 - L.sx0)
        pos, size = V2(x - wid * 0.5, L.sy0), V2(wid, L.sy1 - L.sy0)
    else:
        y = L.sy0 + st.swipe * (L.sy1 - L.sy0)
        pos, size = V2(L.sx0, y - wid * 0.5), V2(L.sx1 - L.sx0, wid)
    flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
             | imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_background | imgui.WindowFlags_.no_saved_settings
             | imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_nav)
    imgui.set_next_window_pos(pos, imgui.Cond_.always)
    imgui.set_next_window_size(size)
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(0.0, 0.0))
    imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
    try:
        vis, _ = imgui.begin("##swipe_handle", None, flags)
        if vis:
            imgui.invisible_button("##swipe_grip", size)
            if imgui.is_item_hovered() or imgui.is_item_active():
                imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew if st.vertical else imgui.MouseCursor_.resize_ns)
            if imgui.is_item_active():
                mp = imgui.get_io().mouse_pos
                f = (mp.x - L.sx0) / max(L.sx1 - L.sx0, 1.0) if st.vertical else (mp.y - L.sy0) / max(L.sy1 - L.sy0, 1.0)
                st.swipe = float(min(max(f, 0.03), 0.97))
        imgui.end()
    finally:
        imgui.pop_style_var(2)


def _diff_legend(ui, an, a, ha, b, hb):
    L, s, st, app = ui.L, ui.s, an.st, ui.app
    cc = an.cmp
    w = min(470.0 * s, (L.sx1 - L.sx0) - 24.0 * s)
    x = (L.sx0 + L.sx1) * 0.5
    if not ui.begin_panel("##diff_legend", x, L.sy0 + 12.0 * s, w, 0.0, padding=(12.0 * s, 8.0 * s), bg_alpha=0.93,
                          auto_height=True, pivot=(0.5, 0.0),
                          extra_flags=int(imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_inputs)):
        ui.end_panel()
        return
    try:
        title = f"Added depth: {_storm_name(app, b)} h{hb}  minus  {_storm_name(app, a)} h{ha}"
        bold(title, theme.TEXT)
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        bw = w - 24.0 * s
        n = 48
        R = st.range_m
        for i in range(n):
            d0 = -R + 2.0 * R * i / n
            d1 = -R + 2.0 * R * (i + 1) / n
            c0 = col32(cmpmod.diverging_srgb(d0 if abs(d0) >= cmpmod.FLAT_M else (cmpmod.FLAT_M * 1.01 if d0 >= 0 else -cmpmod.FLAT_M * 1.01), R) + (1.0,))
            c1 = col32(cmpmod.diverging_srgb(d1 if abs(d1) >= cmpmod.FLAT_M else (cmpmod.FLAT_M * 1.01 if d1 >= 0 else -cmpmod.FLAT_M * 1.01), R) + (1.0,))
            dl.add_rect_filled_multi_color(V2(p.x + bw * i / n, p.y), V2(p.x + bw * (i + 1) / n + 0.6, p.y + 12.0 * s), c0, c1, c1, c0)
        dl.add_rect(V2(p.x, p.y), V2(p.x + bw, p.y + 12.0 * s), col32((1, 1, 1, 0.3)), 2.0, 1.0)
        imp = st.imperial
        fmt = (lambda v: f"{v * 3.28084:+.1f} ft") if imp else (lambda v: f"{v:+.1f} m")
        for frac, txt in ((0.0, fmt(-R)), (0.25, fmt(-R / 2)), (0.5, "no change"), (0.75, fmt(R / 2)), (1.0, fmt(R))):
            tx = p.x + bw * frac
            dl.add_line(V2(tx, p.y + 12.0 * s), V2(tx, p.y + 16.0 * s), col32(theme.TEXT_DIM), 1.0)
            ts = imgui.calc_text_size(txt)
            dl.add_text(V2(min(max(tx - ts.x * 0.5, p.x - 4), p.x + bw - ts.x + 4), p.y + 17.0 * s), col32(theme.TEXT_DIM), txt)
        imgui.dummy(V2(bw, 12.0 * s + imgui.get_text_line_height() + 6.0 * s))
        dl.add_text(V2(p.x, p.y - 0.0), col32((0, 0, 0, 0)), "")
        text_c(C_A, "blue: shallower in B")
        imgui.same_line()
        text_c(theme.TEXT_DIM, "|")
        imgui.same_line()
        text_c(theme.WARN, "orange / red: deeper in B")
        ast = cc.area_stats()
        if ast is not None:
            S = ast[0]
            km2 = (lambda v: f"{v * 0.386102:.2f} mi²") if imp else (lambda v: f"{v:.2f} km²")
            dim(f"Newly wet in B {km2(S.new_wet_km2)}   deeper {km2(S.deeper_km2)}   max added "
                + (f"{S.max_added_m * 3.28084:.1f} ft" if imp else f"{S.max_added_m:.2f} m"))
        elif cc.status:
            dim(cc.status)
        if cc.error:
            text_c(theme.ERR, cc.error[:90])
        dim("Source: " + an.depth_label(b), wrap=True)
    finally:
        ui.end_panel()


# ---------------------------------------------------------------------------------------------------- window
def _draw_window(ui, an):
    L, s, st = ui.L, ui.s, an.st
    w0, h0 = st.win[2] * s, st.win[3] * s
    if st.win[0] < 0:
        pos = V2(max(L.sx1 - w0 - 14.0 * s, L.sx0 + 8.0), max(L.sy1 - h0 - 14.0 * s, L.sy0 + 8.0))
    else:
        pos = V2(st.win[0], st.win[1])
    imgui.set_next_window_pos(pos, imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(V2(w0, h0), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size_constraints(V2(380.0 * s, 260.0 * s), V2(1400.0 * s, 1000.0 * s))
    imgui.set_next_window_bg_alpha(0.96)
    flags = int(imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_collapse)
    vis, opened = imgui.begin("Analysis tools##an_win", True, flags)
    try:
        if not opened:
            st.open = False
            an.tool = "none"
        if vis and opened:
            wp, wsz = imgui.get_window_pos(), imgui.get_window_size()
            st.win = [float(wp.x), float(wp.y), float(wsz.x) / s, float(wsz.y) / s]
            _tab_bar(ui, an)
            imgui.separator()
            tab = st.tab
            if tab == "measure":
                _tab_measure(ui, an)
            elif tab == "profile":
                _tab_profile(ui, an)
            elif tab == "point":
                _tab_point(ui, an)
            elif tab == "timeline":
                _tab_timeline(ui, an)
            else:
                _tab_compare(ui, an)
    finally:
        imgui.end()


def _tab_bar(ui, an):
    st = an.st
    imgui.push_style_var(imgui.StyleVar_.frame_padding, V2(9.0 * ui.s, 3.0 * ui.s))
    for i, t in enumerate(TABS):
        if i:
            imgui.same_line()
        on = st.tab == t
        imgui.push_style_color(imgui.Col_.button, V4(*theme.mix(theme.BG3, theme.ACCENT, 0.65 if on else 0.08)))
        if imgui.button(TAB_LABELS[t] + f"##antab{t}"):
            st.tab = t
            tool = TOOL_FOR_TAB.get(t)
            if an.tool != "none" and tool is not None and an.tool not in ("measure_area",):
                an.tool = tool
        imgui.pop_style_color()
    imgui.pop_style_var()


def _source_line(ui, an, storm=None):
    lab = an.depth_label(storm)
    pending = "pending" in lab or "preview" in lab or "DEMO" in lab
    imgui.push_text_wrap_pos(0.0)
    text_c(theme.WARN if pending else theme.OK, "Depth: " + lab)
    imgui.pop_text_wrap_pos()
    if an.source.tiles is not None:
        imgui.same_line()
    clicked = False
    if an.source.tiles is not None:
        mode = an.st.depth_source
        if imgui.small_button(("use flow grid" if mode == "auto" else "use tiles") + "##srcsw"):
            an.set_depth_mode("grid" if mode == "auto" else "auto")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Switch between the solver depth tiles (6.25 m) and the flow grid the 3D view is built from")


def _tool_button(ui, an, label, tool, tip=""):
    on = an.tool == tool
    imgui.push_style_color(imgui.Col_.button, V4(*theme.mix(theme.BG3, C_MEASURE if tool.startswith("measure") else C_PROFILE
                                                           if tool == "profile" else C_POINT, 0.55 if on else 0.12)))
    if imgui.button(label + f"##tool{tool}"):
        an.tool = "none" if on else tool
        if an.tool in ("measure_line", "measure_area"):
            pass
    imgui.pop_style_color()
    if tip and imgui.is_item_hovered():
        imgui.set_tooltip(tip)


# ---- measure
def _tab_measure(ui, an):
    st = an.st
    frame = an.frame
    _tool_button(ui, an, "Distance", "measure_line", "Click points on the map; double-click or Enter finishes")
    imgui.same_line()
    _tool_button(ui, an, "Area", "measure_area", "Click the corners of a polygon; double-click or Enter closes it")
    imgui.same_line()
    if imgui.button("Clear##clrm"):
        an.clear_measures()
    imgui.same_line()
    imgui.text_disabled("   Units")
    imgui.same_line()
    if imgui.radio_button("Metric##um", not st.imperial):
        st.imperial = False
    imgui.same_line()
    if imgui.radio_button("Imperial##ui", st.imperial):
        st.imperial = True
    imgui.spacing()
    if not an.measures:
        dim("Pick Distance or Area, then click on the map.  Lengths are true ground distances on the WGS84 ellipsoid "
            "(not the flat map units, which can be off by 0.3 % across the county); areas are exact ellipsoidal areas.  "
            "Backspace removes the last point, Esc cancels.", wrap=True)
        return
    kill = None
    for i, m in enumerate(an.measures):
        if len(m.pts) == 0:
            continue
        if m.kind == "line":
            ln = m.length_m(frame)
            a, b = geodesy.fmt_length(ln, st.imperial)
            bold(f"Distance {i + 1}", C_MEASURE)
            imgui.same_line()
            bold(a, theme.TEXT)
            imgui.same_line()
            dim(f"({b})   {len(m.pts)} points" + ("" if m.done else "   drawing..."))
        else:
            ar = m.area_m2(frame)
            per = m.length_m(frame, closed=True)
            a, b = geodesy.fmt_area(ar, st.imperial)
            bold(f"Area {i + 1}", C_MEASURE)
            imgui.same_line()
            bold(a, theme.TEXT)
            imgui.same_line()
            pa, pb = geodesy.fmt_length(per, st.imperial)
            dim(f"({b})   perimeter {pa}   {len(m.pts)} corners" + ("" if m.done else "   drawing..."))
        imgui.same_line()
        if imgui.small_button(f"x##del{i}"):
            kill = i
    if kill is not None:
        an.measures.pop(kill)
    # a quick all-units table for the newest finished measure
    done = [m for m in an.measures if m.done]
    if done:
        imgui.spacing()
        m = done[-1]
        if m.kind == "line":
            ln = m.length_m(frame)
            dim(f"Latest: {ln:,.1f} m  =  {ln / geodesy.M_PER_FT:,.0f} ft  =  {ln / 1000:.3f} km  =  {ln / geodesy.M_PER_MI:.3f} mi")
        else:
            ar = m.area_m2(frame)
            dim(f"Latest: {ar:,.0f} m²  =  {ar / geodesy.M2_PER_ACRE:,.2f} acres  =  {ar / geodesy.M2_PER_HA:,.2f} ha  =  "
                f"{ar / geodesy.M2_PER_FT2:,.0f} ft²")


# ---- profile
def _runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _plot_style(ui):
    C = implot.Col_
    for col, rgba in ((C.plot_bg, theme.with_alpha(theme.BG0, 0.55)), (C.frame_bg, (0, 0, 0, 0)),
                      (C.plot_border, theme.with_alpha(theme.BORDER, 0.8)), (C.axis_text, theme.TEXT_DIM),
                      (C.axis_grid, theme.with_alpha(theme.BORDER, 0.9)), (C.axis_tick, theme.TEXT_MUTE)):
        implot.push_style_color(col, V4(*rgba))
    implot.push_style_var(implot.StyleVar_.plot_padding, V2(6.0, 4.0))


def _plot_style_end():
    implot.pop_style_var()
    implot.pop_style_color(6)


PLOT_FLAGS = (implot.Flags_.no_title | implot.Flags_.no_menus | implot.Flags_.no_box_select | implot.Flags_.no_mouse_text)


def _tab_profile(ui, an):
    st = an.st
    _tool_button(ui, an, "Draw line", "profile", "Click points on the map; double-click or Enter finishes")
    imgui.same_line()
    if imgui.button("Clear##clrp"):
        an.profile_line.clear()
        an.profile = None
        an.profile_key = None
    imgui.same_line()
    imgui.text_disabled("   Units")
    imgui.same_line()
    if imgui.radio_button("Metric##pm", not st.imperial):
        st.imperial = False
    imgui.same_line()
    if imgui.radio_button("Imperial##pi", st.imperial):
        st.imperial = True
    p = an.profile
    app = ui.app
    if len(an.profile_line) < 2:
        dim("Draw a line across a stream, road or levee.  The plot shows the ground along the line and the water surface of the selected "
            "storm at the current hour (hour as of the snapshot).  Move the timeline to watch the water rise.", wrap=True)
        _source_line(ui, an)
        return
    if p is None:
        text_c(theme.TEXT_DIM, "computing profile..." if not an.profile_error else "profile failed: " + an.profile_error)
        return
    imp = st.imperial
    f = geodesy.M_PER_FT if imp else 1.0
    ul = "ft" if imp else "m"
    bold(f"{app.storm_label(p.storm)}   hour {int(p.hour)}", theme.TEXT)
    imgui.same_line()
    a, b = geodesy.fmt_length(p.length_m, imp)
    dim(f"   length {a}   |   under water {geodesy.fmt_length(p.wet_length_m, imp)[0]}   |   max depth "
        + (f"{p.max_depth_m / f:.1f} ft" if imp else f"{p.max_depth_m:.2f} m"))
    d = p.dist_m / (geodesy.M_PER_MI if (imp and p.length_m / f > 5280) else (1000.0 if (not imp and p.length_m > 3000) else f))
    xu = "mi" if (imp and p.length_m / f > 5280) else ("km" if (not imp and p.length_m > 3000) else ul)
    xs = np.ascontiguousarray(d)
    g = np.array(p.ground_m, np.float64)
    fin = np.isfinite(g)
    if fin.any():
        g = np.interp(np.arange(len(g)), np.flatnonzero(fin), g[fin])
    else:
        g = np.zeros_like(g)
    surf = np.where(p.depth_m > 0, g + p.depth_m, g)
    gz, sz = np.ascontiguousarray(g / f), np.ascontiguousarray(surf / f)
    lo = float(np.min(gz)) - 0.08 * max(float(np.ptp(gz)), 1.0)
    hi = float(max(np.max(sz), np.max(gz)))
    hi += 0.12 * max(hi - lo, 1.0)
    avail = imgui.get_content_region_avail()
    ph = max(avail.y - 44.0 * ui.s, 120.0)
    _plot_style(ui)
    try:
        if implot.begin_plot("##profile_plot", V2(-1, ph), PLOT_FLAGS | implot.Flags_.no_legend):
            AF = implot.AxisFlags_
            implot.setup_axes(f"distance ({xu})", f"elevation ({ul})", AF.no_highlight, AF.no_highlight)
            implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, float(xs[-1]), implot.Cond_.always)
            implot.setup_axis_limits(implot.ImAxis_.y1, lo, hi, implot.Cond_.always)
            implot.setup_finish()
            floor = np.full_like(gz, lo)
            implot.plot_shaded("ground", xs, gz, floor, spec=implot.Spec(fill_color=V4(*C_GROUND), fill_alpha=0.55))
            implot.plot_line("ground line", xs, gz, spec=implot.Spec(line_color=V4(0.82, 0.70, 0.50, 1.0), line_weight=1.8))
            if (p.depth_m > 0).any():
                implot.plot_shaded("water", xs, gz, sz, spec=implot.Spec(fill_color=V4(*C_WATER), fill_alpha=0.62))
                for k, (i, j) in enumerate(_runs(p.depth_m > 0)):          # drawn after the ground line so a thin film stays visible
                    j2 = min(j + 1, len(xs))
                    implot.plot_line(f"water surface##{k}", xs[max(i - 1, 0):j2], sz[max(i - 1, 0):j2],
                                     spec=implot.Spec(line_color=V4(0.55, 0.85, 1.0, 1.0), line_weight=3.0))
            vd = np.ascontiguousarray(p.vertex_dist_m / (geodesy.M_PER_MI if xu == "mi" else 1000.0 if xu == "km" else f))
            if len(vd) > 2:
                implot.plot_inf_lines("vertices", vd[1:-1], spec=implot.Spec(line_color=V4(*theme.with_alpha(C_PROFILE, 0.5)), line_weight=1.0))
            hovered = implot.is_plot_hovered()
            an.hover_profile_d = None
            if hovered:
                mp = implot.get_plot_mouse_pos()
                dd = min(max(mp.x, 0.0), float(xs[-1]))
                scale = p.length_m / float(xs[-1]) if xs[-1] > 0 else 1.0
                r = p.at(dd * scale)
                an.hover_profile_d = r["dist_m"]
                pa = implot.plot_to_pixels(dd, lo)
                pb = implot.plot_to_pixels(dd, hi)
                dlp = implot.get_plot_draw_list()
                dlp.add_line(V2(pa.x, pa.y), V2(pb.x, pb.y), col32(theme.with_alpha(theme.TEXT, 0.5)), 1.0)
                lines = [f"{geodesy.fmt_length(r['dist_m'], imp)[0]} from A",
                         f"ground  {r['ground_m'] / f:.1f} {ul}" if math.isfinite(r["ground_m"]) else "ground  no data",
                         ("water depth  " + (f"{r['depth_m'] / f:.2f} ft" if imp else f"{r['depth_m']:.2f} m")) if r["depth_m"] > 0 else "dry"]
                if r["depth_m"] > 0 and math.isfinite(r["surface_m"]):
                    lines.append(f"water surface  {r['surface_m'] / f:.1f} {ul}")
                imgui.set_tooltip("\n".join(lines))
            implot.end_plot()
    finally:
        _plot_style_end()
    _source_line(ui, an, p.storm)
    dim(p.ground_label + f"   |   {p.n} samples every {p.step_m:.1f} m", wrap=False)


# ---- point / building series
def _tab_point(ui, an):
    st, app = an.st, ui.app
    _tool_button(ui, an, "Pick point / building", "point", "Click a building or the ground")
    sel = bridge.selection(app).get("building")
    if sel is not None:
        imgui.same_line()
        if imgui.button("Use selected building##usesel"):
            an.set_target_building(int(sel))
    imgui.same_line()
    imgui.text_disabled("   Units")
    imgui.same_line()
    if imgui.radio_button("m##pu", not st.imperial):
        st.imperial = False
    imgui.same_line()
    if imgui.radio_button("ft##pf", st.imperial):
        st.imperial = True
    r = an.series()
    if an.target is None or r is None:
        dim("Click a building or any spot on the map.  You get its depth hour by hour for the selected storm, the same spot in all the other "
            "storms, and its peak depth per storm.", wrap=True)
        _source_line(ui, an)
        return
    imp = st.imperial
    f = geodesy.M_PER_FT if imp else 1.0
    ul = "ft" if imp else "m"
    bold(an.target.label, C_POINT)
    imgui.same_line()
    dim("building (max over its footprint)" if an.target.kind == "building" else "point (bilinear)")
    if not r.complete:
        hbar(r.done / max(r.total, 1), theme.ACCENT, 6.0)
        dim(f"reading {r.done}/{r.total} storms (current storm first)")
    cur = r.index(app.storm.name)
    xs = np.arange(25, dtype=np.float64)
    avail = imgui.get_content_region_avail()
    h1 = max((avail.y - 56.0 * ui.s) * 0.58, 90.0)
    _plot_style(ui)
    try:
        if implot.begin_plot("##series_plot", V2(-1, h1), PLOT_FLAGS | implot.Flags_.no_legend):
            AF = implot.AxisFlags_
            top = max([np.nanmax(r.depth[i]) for i in range(len(r.storms)) if np.isfinite(r.depth[i]).any()] + [0.3])
            implot.setup_axes("hour", f"depth ({ul})", AF.no_highlight, AF.no_highlight)
            implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, 24.0, implot.Cond_.always)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, float(top / f) * 1.12, implot.Cond_.always)
            implot.setup_finish()
            for i, name in enumerate(r.storms):
                if i == cur or not np.isfinite(r.depth[i]).any():
                    continue
                y = np.ascontiguousarray(np.nan_to_num(r.depth[i], nan=0.0) / f)
                implot.plot_line(name, xs, y, spec=implot.Spec(line_color=V4(*theme.with_alpha(theme.TEXT_DIM, 0.55)), line_weight=1.1))
            if cur >= 0 and np.isfinite(r.depth[cur]).any():
                y = np.ascontiguousarray(np.nan_to_num(r.depth[cur], nan=0.0) / f)
                implot.plot_shaded("sel", xs, y, 0.0, spec=implot.Spec(fill_color=V4(*C_POINT), fill_alpha=0.16))
                implot.plot_line(r.storms[cur], xs, y, spec=implot.Spec(line_color=V4(*C_POINT), line_weight=2.6))
            implot.plot_inf_lines("now", np.array([min(max(app.t, 0.0), 24.0)]),
                                  spec=implot.Spec(line_color=V4(*theme.ACCENT2), line_weight=1.8))
            if implot.is_plot_hovered():
                mp = implot.get_plot_mouse_pos()
                hr = int(round(min(max(mp.x, 0.0), 24.0)))
                lines = [f"hour {hr}"]
                order = sorted(range(len(r.storms)), key=lambda i: -(r.depth[i][hr] if np.isfinite(r.depth[i][hr]) else -1))
                for i in order[:8]:
                    v = r.depth[i][hr]
                    lines.append(f"{'> ' if i == cur else '  '}{r.storms[i]:<6} " + (f"{v / f:.2f} {ul}" if np.isfinite(v) else "..."))
                imgui.set_tooltip("\n".join(lines))
                if imgui.is_mouse_clicked(0):
                    app.t = float(hr)
            implot.end_plot()
        h2 = max(imgui.get_content_region_avail().y - 36.0 * ui.s, 70.0)
        if implot.begin_plot("##peak_plot", V2(-1, h2), PLOT_FLAGS | implot.Flags_.no_legend):
            AF = implot.AxisFlags_
            n = len(r.storms)
            peaks = np.array([r.peak(i) if np.isfinite(r.depth[i]).any() else 0.0 for i in range(n)])
            ymax = float(max(peaks.max() / f, 0.3 / f)) * 1.25
            implot.setup_axes("", f"peak depth ({ul})", AF.no_highlight | AF.no_grid_lines, AF.no_highlight)
            implot.setup_axis_limits(implot.ImAxis_.x1, -0.7, n - 0.3, implot.Cond_.always)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, ymax, implot.Cond_.always)
            implot.setup_axis_ticks(implot.ImAxis_.x1, np.arange(n, dtype=np.float64), list(r.storms))
            implot.setup_finish()
            idx = np.arange(n, dtype=np.float64)
            others = np.array([peaks[i] / f if i != cur else 0.0 for i in range(n)])
            implot.plot_bars("others", idx, np.ascontiguousarray(others), 0.62,
                             spec=implot.Spec(fill_color=V4(*theme.with_alpha(theme.ACCENT, 0.55)), line_color=V4(*theme.ACCENT)))
            if cur >= 0:
                one = np.zeros(n)
                one[cur] = peaks[cur] / f
                implot.plot_bars("selected", idx, np.ascontiguousarray(one), 0.62,
                                 spec=implot.Spec(fill_color=V4(*C_POINT), line_color=V4(1, 1, 1, 1)))
            for i in range(n):
                if peaks[i] > 0:
                    implot.plot_text(f"{peaks[i] / f:.2f}", float(i), float(peaks[i] / f), V2(0.0, -9.0))
            if implot.is_plot_hovered():
                mp = implot.get_plot_mouse_pos()
                i = int(round(mp.x))
                if 0 <= i < n:
                    ph_ = r.peak_hour(i)
                    imgui.set_tooltip(f"{r.storms[i]}: peak {peaks[i] / f:.2f} {ul} at hour {ph_}\n{r.labels[i]}\nclick to show this storm")
                    if imgui.is_mouse_clicked(0):
                        app.set_storm(r.storms[i])
            implot.end_plot()
    finally:
        _plot_style_end()
    dim("Peak of the hourly snapshots (the solver may peak between two hours).", wrap=False)
    labs = sorted(set(l for l in r.labels if l))
    if labs:
        imgui.push_text_wrap_pos(0.0)
        for l in labs[:3]:
            dim("Depth: " + l)
        imgui.pop_text_wrap_pos()
    if r.error:
        text_c(theme.ERR, r.error[:100])


# ---- timeline
def _mini_plot(ui, pid, tl, ys, color, ylab, fmt, now_t, h):
    app = ui.app
    xs = np.arange(25, dtype=np.float64)
    m = np.isfinite(ys)
    top = float(np.nanmax(ys[m])) * 1.15 if m.any() and np.nanmax(ys[m]) > 0 else 1.0
    _plot_style(ui)
    try:
        if implot.begin_plot("##" + pid, V2(-1, h), PLOT_FLAGS | implot.Flags_.no_legend):
            AF = implot.AxisFlags_
            implot.setup_axes("hour" if pid.endswith("area") else "", ylab, AF.no_highlight, AF.no_highlight)
            implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, 24.0, implot.Cond_.always)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, top, implot.Cond_.always)
            implot.setup_axis_format(implot.ImAxis_.y1, fmt)
            implot.setup_finish()
            if m.any():
                implot.plot_shaded(pid, xs[m], np.ascontiguousarray(ys[m]), 0.0, spec=implot.Spec(fill_color=V4(*color), fill_alpha=0.2))
                implot.plot_line(pid, xs[m], np.ascontiguousarray(ys[m]), spec=implot.Spec(line_color=V4(*color), line_weight=2.2))
            implot.plot_inf_lines("now" + pid, np.array([min(max(now_t, 0.0), 24.0)]),
                                  spec=implot.Spec(line_color=V4(*theme.ACCENT2), line_weight=1.8))
            if implot.is_plot_hovered():
                mp = implot.get_plot_mouse_pos()
                hr = int(round(min(max(mp.x, 0.0), 24.0)))
                v = ys[hr]
                imgui.set_tooltip(f"hour {hr}: " + (fmt % v if np.isfinite(v) else "computing") + "\nclick to seek")
                if imgui.is_mouse_down(0):
                    app.t = float(min(max(mp.x, 0.0), 24.0))
            implot.end_plot()
    finally:
        _plot_style_end()


def _tab_timeline(ui, an):
    app, st = ui.app, an.st
    tl = an.timeline()
    storm = app.storm.name
    bold(f"Flood timeline - {app.storm_label(storm)}", theme.TEXT)
    imgui.same_line()
    imgui.text_disabled("   Units")
    imgui.same_line()
    if imgui.radio_button("Metric##tm", not st.imperial):
        st.imperial = False
    imgui.same_line()
    if imgui.radio_button("Imperial##ti", st.imperial):
        st.imperial = True
    if tl is None:
        dim("computing...")
        return
    imp = st.imperial
    hr = units.snapshot_hour(app.t)
    if not tl.complete:
        hbar(tl.done / 25.0, theme.ACCENT, 6.0)
        dim(f"computing hour {tl.done}/25" + ("  (full-resolution tiles, once; cached on disk afterwards)" if tl.source == "tiles" else ""))
    now = lambda a: (a[hr] if np.isfinite(a[hr]) else float("nan"))                 # noqa: E731
    fl, rk, ar = now(tl.flooded), now(tl.road_km), now(tl.area_km2)
    if np.isfinite(fl):
        dim(f"As of hour {hr}:  {int(fl):,} flooded buildings   |   "
            + (f"{rk * 0.621371:.1f} mi" if imp else f"{rk:.1f} km") + " of wet roads   |   "
            + (f"{ar * 0.386102:.2f} mi²" if imp else f"{ar:.2f} km²") + " flooded area")
    avail = imgui.get_content_region_avail().y - 44.0 * ui.s
    h = max(avail / 3.0, 62.0)
    rk_arr = tl.road_km * (0.621371 if imp else 1.0)
    ar_arr = tl.area_km2 * (0.386102 if imp else 1.0)
    _mini_plot(ui, "tl_bld", tl, tl.flooded, theme.TIER_RGB[3], "buildings", "%.0f", app.t, h)
    _mini_plot(ui, "tl_road", tl, rk_arr, theme.TIER_RGB[2], "roads (mi)" if imp else "roads (km)", "%.1f", app.t, h)
    _mini_plot(ui, "tl_area", tl, ar_arr, theme.ACCENT, "area (mi²)" if imp else "area (km²)", "%.1f", app.t, h)
    imgui.push_text_wrap_pos(0.0)
    lab = tl.label
    text_c(theme.WARN if ("pending" in lab or "preview" in lab) else theme.OK, "Numbers from: " + lab)
    imgui.pop_text_wrap_pos()
    if tl.source == "tiles":
        dim("Flooded = depth >= 0.1 m (building corners + centroid, 10 m road samples); area = cells >= 2 cm.", wrap=True)
    else:
        dim("Same numbers as the sidebar statistics (flow grid, all-touched building pixels).", wrap=True)


# ---- compare
def _tab_compare(ui, an):
    app, st = ui.app, an.st
    cc = an.cmp
    names = list(app.storms_available)
    bold("Compare two storms or two hours", theme.TEXT)
    for i, key in enumerate(COMPARE_MODES):
        if i:
            imgui.same_line()
        lab = {"off": "Off", "swipe": "Swipe", "diff": "Difference"}[key]
        if imgui.radio_button(lab + "##cm", st.mode == key):
            st.mode = key
    imgui.same_line()
    imgui.text_disabled("   Units")
    imgui.same_line()
    if imgui.radio_button("m##cu", not st.imperial):
        st.imperial = False
    imgui.same_line()
    if imgui.radio_button("ft##cf", st.imperial):
        st.imperial = True
    imgui.separator()
    a, ha = cc.a_state(app)
    b, hb = cc.b_state(app)
    bold("A", C_A)
    imgui.same_line()
    imgui.text(f"{app.storm_label(a)}, hour {ha}   (the viewer's storm and time)")
    bold("B", C_B)
    imgui.same_line()
    cur = names.index(b) if b in names else 0
    imgui.set_next_item_width(190.0 * ui.s)
    ch, i = imgui.combo("##bstorm", cur, [app.storm_label(n) for n in names])
    if ch:
        st.b_storm = names[i]
    imgui.same_line()
    _, st.b_same_hour = imgui.checkbox("same hour as A##bsame", st.b_same_hour)
    if not st.b_same_hour:
        imgui.set_next_item_width(-1)
        ch, v = imgui.slider_float("##bhour", st.b_hour, 0.0, 24.0, "hour %.0f")
        if ch:
            st.b_hour = float(round(v))
    if a == b and ha == hb:
        text_c(theme.WARN, "A and B are the same storm and hour: pick another storm or turn off 'same hour' for B.")
    if st.mode == "swipe":
        ok = cc.same_frame(app, b)
        if not ok:
            text_c(theme.ERR, f"{app.storm_label(b)} is on a different grid than the 3D view: use the difference layer for it.")
        _, st.vertical = imgui.checkbox("vertical divider (A left, B right)##vert", st.vertical)
        imgui.same_line()
        imgui.set_next_item_width(140.0 * ui.s)
        _, st.swipe = imgui.slider_float("divider##swp", st.swipe, 0.03, 0.97, "%.2f")
        dim("Drag the divider on the map.  Water of A and B is drawn the same way; buildings, roads, arrows and particles still show storm A "
            "on both sides.  Storm B costs about 190 MB of video memory while this mode is on.", wrap=True)
    elif st.mode == "diff":
        imgui.set_next_item_width(160.0 * ui.s)
        _, st.range_m = imgui.slider_float("range +-##rng", st.range_m, 0.25, 6.0, "%.2f m")
        imgui.same_line()
        imgui.set_next_item_width(130.0 * ui.s)
        _, st.opacity = imgui.slider_float("opacity##opc", st.opacity, 0.2, 1.0, "%.2f")
        if cc.status:
            dim(cc.status)
        if cc.error:
            text_c(theme.ERR, cc.error)
        ast = cc.area_stats()
        if ast is not None:
            imgui.spacing()
            for line in ast[0].as_lines():
                imgui.text(line)
            dim(f"Areas and volumes at the {ast[1]}; the picture is drawn at the flow-grid cell size (tiles are averaged down from 6.25 m).",
                wrap=True)
        elif cc.shown_key is not None:
            dim("measuring at tile resolution...")
    bs = cc.building_stats()
    if st.mode != "off" and bs is not None:
        _, sa, sb = bs
        new = np.setdiff1d(sb.flooded_ids, sa.flooded_ids).size
        gone = np.setdiff1d(sa.flooded_ids, sb.flooded_ids).size
        imgui.spacing()
        bold("Buildings and roads", theme.TEXT_DIM)
        imgui.text(f"Flooded buildings: A {sa.flooded:,}   B {sb.flooded:,}   (+{new:,} newly flooded in B, {gone:,} dry again)")
        km = (lambda v: f"{v * 0.621371:.1f} mi") if st.imperial else (lambda v: f"{v:.1f} km")
        imgui.text(f"Wet road length:   A {km(sa.road_km)}   B {km(sb.road_km)}")
    elif st.mode != "off":
        dim("counting flooded buildings and roads...")
    imgui.spacing()
    _source_line(ui, an, b)
