"""Utilities layer (water towers / tanks / treatment / dams / hydro / power plants / substations + power lines and pipelines).

Data: bgv/data/utilities.py (layers/utilities.geojson.gz from scripts/build_utilities.py).  Drawn with the imgui background draw
list, so it costs no GPU work: the projection is one numpy matrix product per frame, only the visible markers are emitted, the
per-storm depth arrays are rebuilt on a storm / domain change only.  Registered through the extension registry: sidebar section
(layer toggle, per-type toggles = legend), View menu item, click hook (card), search hits (see panels_search `kind == "utility"`).
"""
from __future__ import annotations

import math
import os

import numpy as np
from imgui_bundle import imgui

from ..data import utilities as ud
from . import registry, theme
from .theme import col32
from .widgets import bold, dim, text_c

V2 = imgui.ImVec2
MARK_PX = 7.0
PICK_PX = 13.0
MAX_LINE_VERTS = 1500            # vertices emitted per frame (strided when more are visible)


class _S:
    """Per-process draw state (one UI at a time)."""
    view = None
    view_key = None
    dcm = None                   # int depth cm per point for the displayed storm (-1 = unknown)
    lfrac = None                 # flooded fraction per line (0..1), -1 unknown
    ldcm = None
    storm_key = None
    sx = sy = vis = None         # last projected points (picking)
    sel = None                   # selected feature index
    hover = None


def cfg(app) -> dict:
    d = registry.state(app, "utilities")
    d.setdefault("on", os.environ.get("BGV_UTILITIES") == "1")      # default off; the env var turns it on for benchmarks / devshots
    d.setdefault("groups", {})
    return d


def enabled(app) -> bool:
    return bool(cfg(app).get("on"))


def group_on(app, g: str) -> bool:
    return bool(cfg(app)["groups"].get(g, True))


def uset(app):
    return ud.for_root(app.root)


def _ensure(app, us):
    key = (id(app.terrain), app.domain.key)
    if _S.view is None or _S.view_key != key:
        _S.view = ud.build_view(us, app.domain.key, app.terrain.frame, getattr(app.terrain, "sampler", None))
        _S.view_key = key
        _S.storm_key = None
    sk = (key, app.storm.name if app.storm is not None else None)
    if _S.storm_key != sk:
        name = sk[1]
        v = _S.view
        _S.dcm = np.array([us.depth_cm(us.features[i], name) if us.depth_cm(us.features[i], name) is not None else -1
                           for i in v.ids], np.int64) if len(v.ids) else np.zeros(0, np.int64)
        k = us.storm_index(name)
        fr, mx = [], []
        for i in v.line_ids:
            f = us.features[i]
            fr.append(f.f10[k] / 100.0 if (k >= 0 and f.f10 and k < len(f.f10)) else -1.0)
            mx.append(f.d[k] if (k >= 0 and k < len(f.d)) else -1)
        _S.lfrac, _S.ldcm = np.array(fr, float), np.array(mx, np.int64)
        _S.storm_key = sk


def _project(app, xyz):
    P = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1) @ app.camera.view_proj().T
    w = P[:, 3]
    ok = w > 1e-6
    ws = np.where(ok, w, 1.0)
    cam = app.camera
    return (P[:, 0] / ws * 0.5 + 0.5) * cam.width, (0.5 - P[:, 1] / ws * 0.5) * cam.height, ok


# ------------------------------------------------------------------------------------------------ marker shapes
def _marker(dl, shape, x, y, r, rgb):
    f, o = col32((*rgb, 1.0)), col32((0.02, 0.03, 0.05, 1.0))
    c = V2(x, y)
    if shape == "circle":
        dl.add_circle_filled(c, r, f, 14)
        dl.add_circle(c, r, o, 14, 1.6)
    elif shape == "triangle":
        pts = [V2(x, y - r * 1.2), V2(x + r * 1.1, y + r * 0.8), V2(x - r * 1.1, y + r * 0.8)]
        dl.add_triangle_filled(*pts, f)
        dl.add_triangle(*pts, o, 1.6)
    elif shape == "square":
        dl.add_rect_filled(V2(x - r * 0.9, y - r * 0.9), V2(x + r * 0.9, y + r * 0.9), f)
        dl.add_rect(V2(x - r * 0.9, y - r * 0.9), V2(x + r * 0.9, y + r * 0.9), o, 0.0, 1.6)
    elif shape == "diamond":
        pts = [V2(x, y - r * 1.2), V2(x + r * 1.2, y), V2(x, y + r * 1.2), V2(x - r * 1.2, y)]
        dl.add_quad_filled(*pts, f)
        dl.add_quad(*pts, o, 1.6)
    elif shape in ("hexagon", "pentagon"):
        n = 6 if shape == "hexagon" else 5
        dl.add_ngon_filled(c, r * 1.1, f, n)
        dl.add_ngon(c, r * 1.1, o, n, 1.6)
    elif shape == "cross":
        a, b = r * 1.25, r * 0.45
        for p0, p1 in (((x - a, y - b), (x + a, y + b)), ((x - b, y - a), (x + b, y + a))):
            dl.add_rect_filled(V2(*p0), V2(*p1), f)
        dl.add_rect(V2(x - a, y - b), V2(x + a, y + b), o, 0.0, 1.2)
        dl.add_rect(V2(x - b, y - a), V2(x + b, y + a), o, 0.0, 1.2)
        dl.add_rect_filled(V2(x - b + 1, y - b + 1), V2(x + b - 1, y + b - 1), f)
    else:
        dl.add_circle_filled(c, r * 0.6, f, 10)
        dl.add_circle(c, r * 0.6, o, 10, 1.2)


def _line_colour(frac):
    if frac < 0:
        return (0.55, 0.6, 0.66, 0.7)
    a, b = np.array((0.50, 0.58, 0.72)), np.array((0.84, 0.10, 0.10))
    c = a + (b - a) * min(1.0, frac * 1.6)
    return (float(c[0]), float(c[1]), float(c[2]), 0.85 + 0.15 * min(1.0, frac))


# ------------------------------------------------------------------------------------------------ drawing
def draw(ui):
    app = ui.app
    if not enabled(app) or app.loading or app.terrain is None or app.storm is None:
        _S.sx = None
        return
    us = uset(app)
    if us is None:
        return
    _ensure(app, us)
    v, s, L = _S.view, ui.s, ui.L
    dl = imgui.get_background_draw_list()
    cam = app.camera
    exag = float(app.exag)
    r = MARK_PX * s
    dl.push_clip_rect(V2(L.sx0, L.sy0), V2(L.sx1, L.sy1), True)
    try:
        if len(v.line_ids) and any(group_on(app, g) for g in ud.LINE_GROUPS):
            _draw_lines(ui, app, v, dl, exag, s)
        if len(v.ids):
            xyz = np.stack([v.xy[:, 0], v.xy[:, 1], v.z * exag + 6.0], axis=1)
            sx, sy, ok = _project(app, xyz)
            vis = ok & (sx > L.sx0 - 20) & (sx < L.sx1 + 20) & (sy > L.sy0 - 20) & (sy < L.sy1 + 20)
            _S.sx, _S.sy, _S.vis = sx, sy, vis
            io = imgui.get_io()
            mp = io.mouse_pos
            _S.hover = None
            best = PICK_PX * s
            for n in np.flatnonzero(vis):
                g = v.group[n]
                if not group_on(app, g):
                    continue
                x, y = float(sx[n]), float(sy[n])
                if _S.dcm[n] >= ud.FLOOD_CM:
                    dl.add_circle(V2(x, y), r + 4.5 * s, col32((1, 1, 1, 0.95)), 18, 4.0 * s)
                    dl.add_circle(V2(x, y), r + 4.5 * s, col32((0.85, 0.1, 0.1, 1.0)), 18, 2.2 * s)
                _, _, rgb, shape, _ = ud.GROUPS[g]
                _marker(dl, shape, x, y, r, rgb)
                if not io.want_capture_mouse:
                    d = math.hypot(x - mp.x, y - mp.y)
                    if d < best:
                        best, _S.hover = d, int(v.ids[n])
            sel = _S.sel
            if sel is not None:
                hit = np.flatnonzero(v.ids == sel)
                if len(hit) and vis[hit[0]]:
                    x, y = float(sx[hit[0]]), float(sy[hit[0]])
                    dl.add_circle(V2(x, y), r + 9 * s, col32((1, 1, 1, 1)), 24, 2.5 * s)
    finally:
        dl.pop_clip_rect()
    if _S.hover is not None and not io.want_capture_mouse:
        f = us.features[_S.hover]
        imgui.begin_tooltip()
        imgui.text(f.title())
        dim(f"{ud.TYPE_LABEL[f.type]}  |  click for details")
        imgui.end_tooltip()
    _card(ui, app, us)


def _draw_lines(ui, app, v, dl, exag, s):
    off, xyz = v.line_off, v.line_xyz
    P = np.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2] * exag + 4.0], axis=1)
    sx, sy, ok = _project(app, P)
    L = ui.L
    inside = ok & (sx > L.sx0 - 50) & (sx < L.sx1 + 50) & (sy > L.sy0 - 50) & (sy < L.sy1 + 50)
    n_vis = int(inside.sum())
    if n_vis == 0:
        return
    stride = max(1, int(math.ceil(n_vis / MAX_LINE_VERTS)))
    anyin = np.add.reduceat(inside.astype(np.int32), off[:-1]) > 0
    sxl, syl, okl = sx.tolist(), sy.tolist(), ok.tolist()
    on = {g: group_on(app, g) for g in ud.LINE_GROUPS}
    for li in np.flatnonzero(anyin).tolist():
        grp = v.line_group[li]
        if not on[grp]:
            continue
        a, b = int(off[li]), int(off[li + 1])
        idx = list(range(a, b, stride))
        if idx[-1] != b - 1:
            idx.append(b - 1)
        pts = [V2(sxl[i], syl[i]) for i in idx if okl[i]]
        if len(pts) < 2:
            continue
        fr = float(_S.lfrac[li])
        col = _line_colour(fr)
        if grp == "pipelines" and fr < 0.05:
            col = (0.62, 0.45, 0.78, col[3])
        dl.add_polyline(pts, col32(col), (1.3 + 1.6 * max(fr, 0.0)) * s, 0)


# ------------------------------------------------------------------------------------------------ card / click
def _depth_text(cm):
    if cm is None or cm < 0:
        return "no data for this storm"
    if cm < ud.FLOOD_CM:
        return f"{cm} cm  (below the 10 cm flood threshold)" if cm else "dry"
    return f"{cm} cm  ({cm / 2.54:.0f} in)"


def _card(ui, app, us):
    if _S.sel is None:
        return
    f = us.features[_S.sel]
    s, L = ui.s, ui.L
    w = 330.0 * s
    if not ui.begin_panel("##utilcard", L.sx1 - w - 14.0 * s, L.sy0 + 70.0 * s, w, 0.0, auto_height=True, padding=(12.0, 9.0), bg_alpha=0.96):
        ui.end_panel()
        return
    imgui.push_text_wrap_pos(w - 22.0 * s)
    try:
        bold(f.title())
        imgui.same_line(w - 40.0 * s)
        if imgui.small_button("x##utilclose"):
            _S.sel = None
        dim(ud.TYPE_LABEL[f.type] + (f"  |  {f.voltage} V" if f.voltage else ""))
        if f.operator:
            imgui.text(f"Operator: {f.operator}")
        if f.source:
            imgui.text(f"Source: {f.source}")
        imgui.separator()
        storm = app.storm.name
        cm = us.depth_cm(f, storm)
        text_c(theme.WARN if (cm or 0) >= ud.FLOOD_CM else theme.TEXT, f"Max depth, {storm}: {_depth_text(cm)}")
        if f.is_line:
            k = us.storm_index(storm)
            if k >= 0 and f.f10:
                imgui.text(f"Flooded share of its length: {f.f10[k]}% at >= 10 cm, {f.f30[k]}% at >= 30 cm")
        imgui.text("First design storm that floods it: " + (f.first or "none (2yr to 100yr)"))
        dim("Depth: " + ud.SOURCE_LABEL, wrap=True)
        dim(f"OpenStreetMap {f.osm_id}; OSM completeness varies.", wrap=True)
        if imgui.small_button("Fly to##utilfly"):
            fly_to(ui, f)
    finally:
        imgui.pop_text_wrap_pos()
        ui.end_panel()


def _pick_xy(ui, x, y):
    if _S.sx is None or not enabled(ui.app):
        return None
    v = _S.view
    best, bi = PICK_PX * ui.s, None
    for n in np.flatnonzero(_S.vis):
        if not group_on(ui.app, v.group[n]):
            continue
        d = math.hypot(float(_S.sx[n]) - x, float(_S.sy[n]) - y)
        if d < best:
            best, bi = d, int(v.ids[n])
    return bi


def on_click(ui, x, y) -> bool:
    i = _pick_xy(ui, x, y)
    if i is None:
        return False
    _S.sel = i
    return True


def select(app, feature_index):
    _S.sel = feature_index


def fly_to(ui, f):
    app = ui.app
    fr = app.terrain.frame
    X, Y = ud._world(fr, [f.lng], [f.lat])
    cam = app.camera
    ui.nav.fly(cam, float(X[0]), float(Y[0]), 900.0 if not f.is_line else 3000.0, float(cam.yaw), float(cam.pitch))


def choose_hit(ui, hit):
    """A search hit of kind 'utility' (idx = feature index)."""
    app = ui.app
    cfg(app)["on"] = True
    us = uset(app)
    f = us.features[hit.idx]
    _S.sel = f.i
    fly_to(ui, f)


def hits_for(ui, text):
    """Search hits for the address dropdown (kind 'utility'); [] when the layer data is absent."""
    from ..search.addrindex import Hit
    app = ui.app
    us = uset(app)
    if us is None:
        return []
    near = None
    try:
        near = app.terrain.frame.world_to_lnglat(float(app.camera.target[0]), float(app.camera.target[1]))
    except Exception:
        pass
    out = []
    for f in us.search(text, 4, near):
        if app.domain.key == "city" and not f.in_city:
            continue
        out.append(Hit("utility", f.i, f.title(), ud.TYPE_LABEL[f.type] + (f"  |  {f.operator}" if f.operator else "")))
    return out


# ------------------------------------------------------------------------------------------------ sidebar section (layer toggle + legend)
def _swatch(shape_rgb_shape, s):
    rgb, shape = shape_rgb_shape
    p = imgui.get_cursor_screen_pos()
    h = imgui.get_text_line_height()
    _marker(imgui.get_window_draw_list(), shape, p.x + h * 0.5, p.y + h * 0.5, h * 0.36, rgb)
    imgui.dummy(V2(h, h))
    imgui.same_line()


def section(ui):
    app, s = ui.app, ui.s
    c = cfg(app)
    us = uset(app)
    ch, on = imgui.checkbox("Show utilities layer", bool(c["on"]))
    if ch:
        c["on"] = bool(on)
    if us is None:
        dim("No layers/utilities.geojson.gz: run scripts/build_utilities.py", wrap=True)
        return
    if not c["on"]:
        dim("Water towers, tanks, treatment, dams, power plants, substations, power lines.", wrap=True)
        return
    counts = us.counts
    for g, (label, types, rgb, shape, _) in ud.GROUPS.items():
        n = sum(counts.get(t, 0) for t in types)
        if n == 0:
            continue
        _swatch((rgb, shape), s)
        ch, v = imgui.checkbox(f"{label} ({n})##ug_{g}", group_on(app, g))
        if ch:
            c["groups"][g] = bool(v)
    for g, (label, types) in ud.LINE_GROUPS.items():
        n = sum(counts.get(t, 0) for t in types)
        if n == 0:
            continue
        ch, v = imgui.checkbox(f"{label} ({n})##ug_{g}", group_on(app, g))
        if ch:
            c["groups"][g] = bool(v)
    imgui.text("Ring = flooded >= 10 cm;")
    dim("lines: blue-grey dry, thicker and red as more of their length floods (storm shown).", wrap=True)
    dim("Data: OpenStreetMap; completeness varies (relations not read, so some plants / reservoirs are missing). "
        "Depth: " + ud.SOURCE_LABEL, wrap=True)


def _view_toggle(ui):
    c = cfg(ui.app)
    c["on"] = not c["on"]


registry.register_sidebar_section("utilities", "Utilities", section, order=60)
registry.register_menu_item("View", "Utilities layer (water, power)", _view_toggle, checked=lambda ui: enabled(ui.app), order=95)
registry.register_panel("utilities", draw, order=45)
registry.register_click_hook("utilities click", on_click)
