"""Search / facilities / roads-at-risk / report panels and the first-run hints (imgui).  All logic and data live in
`bgv.search.*` (SearchController); this file only draws and routes input.  Entry points used by the shared UI files
(see desktop/docs/SEARCH_HOOKS.md):

    draw(ui)                    once per frame, called from UiManager.frame()  (markers, search bar, cards, windows, hints, help)
    menu(ui)                    the body of the "Find" menu (menus.py)
    on_key(ui, ev) -> bool      first look at every key_down (UiManager.on_key)
    on_click(ui, x, y) -> bool  a click on the 3D view: a facility marker opens its card (the click still selects the building, UiManager._click)
    post_frame(app, manager)    after the frame: writes a queued risk report (OverlayLayer.post_frame)
"""
from __future__ import annotations

import math
import os

import numpy as np
from imgui_bundle import imgui

from ..search import facilities as fac
from ..search import report as rpt
from ..search import tiles as tl
from ..search.addrindex import TIER_NAMES as TNAMES
from . import bridge, theme, units
from .theme import col32
from .widgets import bold, chip_toggle, dim, shadow_text, text_c

V2 = imgui.ImVec2
V4 = imgui.ImVec4

CTL_ATTR = "_search_ctl"
WIN_FLAGS = int(imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings)
# NOTE: imgui puts a NoBringToFrontOnFocus window at the BACK of the stack (behind the legend and every other overlay panel), so the
# cards and the dropdown deliberately do not use that flag
CARD_FLAGS = int(imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
                 | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize)
TIER_LABEL = ("below 10 cm", "Shallow", "Moderate", "Deep", "Severe")


# ------------------------------------------------------------------------------------------------ controller access
def controller(ui):
    c = getattr(ui, CTL_ATTR, None)
    if c is None:
        from ..search.controller import SearchController
        c = SearchController(ui)
        setattr(ui, CTL_ATTR, c)
    return c


def _tier_col(t: int, a: float = 1.0):
    return theme.tier_color(t, a) if t >= 1 else (*theme.OK[:3], a)


def _depth(cm, tier: int, approx: bool) -> str:
    if cm is None:
        return "-"
    if approx and tier == 4:
        return ">= 120 cm"
    if cm < 1:
        return "dry"
    return f"{'~' if approx else ''}{cm} cm"


def _cm(cm, approx: bool) -> str:
    """Depth in cm for the lists; a depth-tier preview never claims more than '>= 120 cm' for the severe tier."""
    if approx and cm >= 120:
        return ">= 120 cm"
    if cm < 1:
        return "-"
    return ("~" if approx else "") + f"{int(cm)} cm"


def _source_colour(label: str):
    return theme.ACCENT2 if label.startswith("solver") else theme.TEXT_DIM


# ------------------------------------------------------------------------------------------------ entry points
def draw(ui):
    c = controller(ui)
    c.sync()
    if not ui.st.hidden:
        ui.safe("search.markers", lambda u: _markers(u, c))
        ui.safe("search.road", lambda u: _road_outline(u, c))
        ui.safe("search.bar", lambda u: _search_bar(u, c))
        ui.safe("search.card", lambda u: _cards(u, c))
        ui.safe("search.facilities", lambda u: _facility_window(u, c))
        ui.safe("search.roads", lambda u: _roads_window(u, c))
        ui.safe("search.dropdown", lambda u: _dropdown(u, c))
        ui.safe("search.hints", lambda u: _onboarding(u, c))


def menu(ui):
    c = controller(ui)
    if imgui.menu_item("Search address...", "Ctrl+F", False)[0]:
        c.focus = True
    ch, v = imgui.menu_item("Critical-facility markers", "M", c.markers_on)
    if ch:
        c.markers_on = bool(v)
        c.persist()
    if imgui.menu_item("Critical facilities list", "", c.show_fac)[0]:
        c.show_fac = not c.show_fac
    if imgui.menu_item("Roads at risk list", "Ctrl+L", c.show_roads)[0]:
        c.show_roads = not c.show_roads
    imgui.separator()
    if imgui.menu_item("Save risk report (HTML + PNG)", "Ctrl+P", False)[0]:
        c.request_report()
    if c.last_report and imgui.menu_item("Open last report", "", False)[0]:
        _startfile(ui, c.last_report["html"])
    if imgui.menu_item("Open reports folder", "", False)[0]:
        _startfile(ui, str(_report_dir(c)), mkdir=True)
    imgui.separator()
    if imgui.menu_item("Show first-run tips again", "", False)[0]:
        c.onboarded = False
        c.hint_req = True
        c.persist()


def _report_dir(c):
    from pathlib import Path
    return Path((c.app.settings or {}).get("report_dir") or rpt.default_dir())


def _startfile(ui, path, mkdir=False):
    try:
        from pathlib import Path
        if mkdir:
            Path(path).mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))
    except Exception as e:
        ui.toasts.push(f"Cannot open {path}: {e}", theme.ERR)


def on_key(ui, ev) -> bool:
    k = ev.get("key", "")
    mods = set(ev.get("modifiers") or ())
    ctrl = bool(mods & {"Control", "Meta"})
    c = controller(ui)
    kl = k.lower() if len(k) == 1 else k
    if ctrl and kl == "f" and not (mods & {"Alt"}):
        c.focus = True
        ui.st.hidden = False
        return True
    if k == "/" and not mods:
        c.focus = True
        return True
    if ctrl and kl == "p":
        c.request_report()
        return True
    if ctrl and kl == "l" and not (mods & {"Alt"}):          # was plain L; L is the cinematic time-lapse key after the integration
        c.show_roads = not c.show_roads
        return True
    if not mods:
        if kl == "m":
            c.markers_on = not c.markers_on
            c.persist()
            return True
        if k == "Escape" and not (ui.show_help or ui.show_about or ui.st.hidden):     # dialogs / hidden UI keep their old priority
            if c.query:
                c.query, c.committed = "", False
                return True
            if c.result is not None or c.facility_sel >= 0 or c.road_sel is not None:
                c.result, c.facility_sel, c.road_sel = None, -1, None
                return True
    return False


def on_click(ui, x: float, y: float) -> bool:
    """A click on a facility marker opens its card; it does NOT swallow the click, so the building under the marker is selected too."""
    c = controller(ui)
    if c.facilities is None or not c.markers_on:
        return False
    i = c.pick_facility(x, y)
    if i < 0:
        return False
    c.select_facility(i)
    return False            # the click still selects the building under the marker (a school building is both)


def post_frame(app, manager):
    c = getattr(manager, CTL_ATTR, None)
    if c is None or c.report_req is None:
        return
    from . import screenshot

    def cap():
        return screenshot.capture_rgba(app, include_ui=False)[..., :3]
    c.run_report(cap)


# ------------------------------------------------------------------------------------------------ small panel helper
def _panel(ui, name, x, y, w, maxh=1.0e5, pad=(12.0, 9.0), bg=0.95, flags=CARD_FLAGS):
    s = ui.s
    imgui.set_next_window_pos(V2(x, y), int(imgui.Cond_.always), V2(0, 0))
    imgui.set_next_window_size_constraints(V2(w, 0.0), V2(w, max(maxh, 80.0)))
    imgui.set_next_window_bg_alpha(bg)
    imgui.push_style_var(imgui.StyleVar_.window_padding, V2(pad[0] * s, pad[1] * s))
    imgui.push_style_var(imgui.StyleVar_.window_rounding, 8.0)
    vis, _ = imgui.begin(name, None, flags)
    return vis


def _end_panel():
    imgui.end()
    imgui.pop_style_var(2)


def _scene_w(ui):
    return ui.L.sx1 - ui.L.sx0


# ------------------------------------------------------------------------------------------------ markers
def _shape(dl, kind, cx, cy, r, fill, outline):
    f, o = col32(fill), col32(outline)
    if kind.endswith("_small"):
        r *= 0.78
        kind = kind[:-6]
    if kind == "circle":
        dl.add_circle_filled(V2(cx, cy), r, f, 16)
        dl.add_circle(V2(cx, cy), r, o, 16, 1.6)
    elif kind == "square":
        dl.add_rect_filled(V2(cx - r, cy - r), V2(cx + r, cy + r), f, 2.0)
        dl.add_rect(V2(cx - r, cy - r), V2(cx + r, cy + r), o, 2.0, 1.6)
    elif kind == "diamond":
        p = [V2(cx, cy - r * 1.2), V2(cx + r * 1.2, cy), V2(cx, cy + r * 1.2), V2(cx - r * 1.2, cy)]
        dl.add_convex_poly_filled(p, f)
        dl.add_polyline(p + [p[0]], o, 1.6, 0)
    elif kind == "triangle":
        p = [V2(cx, cy - r * 1.2), V2(cx + r * 1.15, cy + r * 0.85), V2(cx - r * 1.15, cy + r * 0.85)]
        dl.add_convex_poly_filled(p, f)
        dl.add_polyline(p + [p[0]], o, 1.6, 0)
    elif kind == "cross":
        a, b = r * 1.25, r * 0.45
        pts = [V2(cx - b, cy - a), V2(cx + b, cy - a), V2(cx + b, cy - b), V2(cx + a, cy - b), V2(cx + a, cy + b), V2(cx + b, cy + b),
               V2(cx + b, cy + a), V2(cx - b, cy + a), V2(cx - b, cy + b), V2(cx - a, cy + b), V2(cx - a, cy - b), V2(cx - b, cy - b)]
        dl.add_rect_filled(V2(cx - b, cy - a), V2(cx + b, cy + a), f)
        dl.add_rect_filled(V2(cx - a, cy - b), V2(cx + a, cy + b), f)
        dl.add_polyline(pts + [pts[0]], o, 1.5, 0)
    else:
        dl.add_circle_filled(V2(cx, cy), r, f, 16)
        dl.add_circle(V2(cx, cy), r, o, 16, 1.6)


def _markers(ui, c):
    fs = c.facilities
    if fs is None or not c.markers_on or fs.n == 0:
        return
    app, L, s = ui.app, ui.L, ui.s
    scr = c.facility_screen(L.W, L.H)
    if scr is None:
        return
    px, py, ok = scr
    storm = c.storm_key()
    eff = fs.depth_cm(storm)
    mask = c.type_mask()
    dl = imgui.get_background_draw_list()
    near = float(app.camera.dist) < 3500.0
    r0 = 6.5 * s
    mouse = ui.mouse
    hover = -1
    best = 16.0 * s
    clip = (L.sx0 + 4, L.sy0 + 4, L.sx1 - 4, L.sy1 - 4)
    for i in np.flatnonzero(ok & mask):
        x, y = float(px[i]), float(py[i])
        if not (clip[0] <= x <= clip[2] and clip[1] <= y <= clip[3]):
            continue
        f = fs.items[i]
        _, rgb, shape = fac.style(f.type)
        cm = int(eff[i])
        tier = tl.tier_of_cm(cm)
        sel = (i == c.facility_sel)
        if tier >= 1:                                              # flooded: ring in the tier colour (+ soft halo)
            tc = _tier_col(tier)
            dl.add_circle_filled(V2(x, y), r0 + 9.0 * s, col32((*tc[:3], 0.20)), 24)
            dl.add_circle(V2(x, y), r0 + 6.0 * s, col32((*tc[:3], 1.0)), 24, 3.0 * s)
            dl.add_circle(V2(x, y), r0 + 9.2 * s, col32((1, 1, 1, 0.85)), 24, 1.0)
        if sel:
            dl.add_circle(V2(x, y), r0 + 12.0 * s, col32((1, 1, 1, 1.0)), 24, 2.0)
        _shape(dl, shape, x, y, r0 * (1.15 if sel else 1.0), (*rgb, 1.0), (0.04, 0.05, 0.1, 1.0))
        if mouse is not None:
            d = math.hypot(mouse[0] - x, mouse[1] - y)
            if d < best:
                best, hover = d, i
        if near and (tier >= 1 or sel):
            shadow_text(dl, (x + r0 + 8.0 * s, y - 7.0 * s), (1, 1, 1, 1), f.name or fac.style(f.type)[0])
    if hover >= 0 and not imgui.get_io().want_capture_mouse:
        f = fs.items[hover]
        cm = int(eff[hover])
        imgui.begin_tooltip()
        imgui.text(f.name or "(unnamed)")
        dim(f"{fac.style(f.type)[0]}  |  {storm}: " + (f"{_cm(cm, fs.is_approx(storm))}  {TIER_LABEL[tl.tier_of_cm(cm)]}" if cm >= 10 else "not flooded"))
        dim("click for details")
        imgui.end_tooltip()


def _road_outline(ui, c):
    rs = c.road_sel
    c.resolve_pending_road()
    if not rs or not rs.get("coords"):
        return
    app = ui.app
    fr = app.terrain.frame
    a = np.asarray(rs["coords"], np.float64)
    if a.shape[0] > 2000:
        a = a[:: int(math.ceil(a.shape[0] / 2000))]
    key = (rs["id"], id(app.terrain))
    if rs.get("_key") != key:
        xy = np.array([fr.lnglat_to_world(float(p[0]), float(p[1])) for p in a])
        rs["_xy"] = xy
        rs["_z"] = np.asarray(app.terrain.sampler.sample(xy[:, 0], xy[:, 1]), np.float64)
        rs["_key"] = key
    xy, z = rs["_xy"], rs["_z"] * float(app.exag) + 3.0
    P = np.stack([xy[:, 0], xy[:, 1], z, np.ones(len(z))], axis=1) @ app.camera.view_proj().T
    w = P[:, 3]
    if (w <= 1e-6).any():
        return
    cam = app.camera
    sx = (P[:, 0] / w * 0.5 + 0.5) * cam.width
    sy = (0.5 - P[:, 1] / w * 0.5) * cam.height
    pts = [V2(float(x), float(y)) for x, y in zip(sx, sy)]
    dl = imgui.get_background_draw_list()
    dl.add_polyline(pts, col32((0, 0, 0, 0.75)), 9.0 * ui.s, 0)
    dl.add_polyline(pts, col32((1.0, 0.86, 0.2, 1.0)), 5.0 * ui.s, 0)
    dl.add_polyline(pts, col32((1.0, 1.0, 1.0, 0.9)), 1.6 * ui.s, 0)


# ------------------------------------------------------------------------------------------------ search bar
def _search_bar(ui, c):
    L, s = ui.L, ui.s
    sw = _scene_w(ui)
    w = min(480.0 * s, max(sw - 130.0 * s, 260.0 * s))
    x = L.sx0 + 112.0 * s
    y = L.sy0 + 10.0 * s
    c.bar_rect = (x, y, w)
    if not _panel(ui, "##search", x, y, w, pad=(10.0, 7.0), bg=0.94):
        _end_panel()
        return
    try:
        want_focus = bool(c.focus)
        if want_focus and getattr(c, "_focus_tries", 0) % 3 == 0:   # pending until the field is really active; re-issued only
            imgui.set_keyboard_focus_here()            # a window that is appearing / fading in skips items, so a
            c.committed = False                          # one-shot request was sometimes silently dropped
        imgui.push_item_width(-62.0 * s)
        ch, txt = imgui.input_text_with_hint("##q", "Search address (Ctrl+F)  e.g. 1309 Scottsville Rd", c.query,
                                             int(imgui.InputTextFlags_.enter_returns_true))
        entered = bool(ch)
        if not entered and txt != c.query:
            c.query, c.sel_hit, c.committed = txt[:80], 0, False
        elif entered:
            c.query = txt[:80]
        imgui.pop_item_width()
        active = imgui.is_item_active()
        c.input_active = active
        if want_focus:
            tries = getattr(c, "_focus_tries", 0) + 1
            if active or tries >= 10:
                c.focus, tries = False, 0
            c._focus_tries = tries
        imgui.same_line()
        if imgui.button("Clear" if c.query else "Help", V2(54.0 * s, 0)):
            if c.query:
                c.query, c.committed = "", False
                c.result = None
            else:
                ui.show_help = True
        hits = c.hits_for(c.query) if c.query.strip() else []
        c.cur_hits = hits
        if active:
            n = len(hits)
            if n and imgui.is_key_pressed(imgui.Key.down_arrow):
                c.sel_hit = (c.sel_hit + 1) % n
            if n and imgui.is_key_pressed(imgui.Key.up_arrow):
                c.sel_hit = (c.sel_hit - 1) % n
            if imgui.is_key_pressed(imgui.Key.escape):
                c.query, c.committed = "", False
                imgui.internal.clear_active_id()
        if entered and hits:
            _choose(ui, c, hits[min(c.sel_hit, len(hits) - 1)])
        c.bar_h = float(imgui.get_window_height())
    finally:
        _end_panel()


def _choose(ui, c, hit):
    if hit.kind == "utility":
        from . import ext_utilities
        c.committed = True
        c.query = hit.label
        ext_utilities.choose_hit(ui, hit)
        return
    if hit.kind == "street":
        c.query = hit.label + " "
        c.sel_hit = 0
        c.focus = True
        return
    c.committed = True
    c.query = hit.label
    c.select_record(hit.idx)


def _dd_visible(c) -> bool:
    if not c.query.strip() or c.committed:
        return False
    return bool(getattr(c, "input_active", False) or c.dd_hover or getattr(c, "force_dd", False))


def _dropdown(ui, c):
    if not _dd_visible(c):
        c.dd_hover = False
        return
    L, s = ui.L, ui.s
    x, y, w = c.bar_rect
    y += getattr(c, "bar_h", 38.0 * s) + 2.0
    maxh = max(L.sy1 - y - 12.0 * s, 120.0 * s)
    if not _panel(ui, "##dropdown", x, y, w, maxh=maxh, pad=(6.0, 6.0), bg=0.97):
        _end_panel()
        return
    try:
        c.dd_hover = imgui.is_window_hovered(imgui.HoveredFlags_.root_and_child_windows | imgui.HoveredFlags_.allow_when_blocked_by_active_item)
        stt = c.status["index"]
        if c.index is None:
            if stt.state in ("missing", "error"):
                imgui.push_text_wrap_pos(w - 20.0 * s)
                text_c(theme.WARN, "Address search is unavailable")
                dim(stt.message, wrap=True)
                imgui.pop_text_wrap_pos()
            else:
                dim("Loading the address index...")
            return
        hits = getattr(c, "cur_hits", [])
        if not hits:
            dim("No match. Type a number and the start of a street, e.g. '1309 scottsville r'.", wrap=True)
            return
        for n, h in enumerate(hits):
            _hit_row(ui, c, n, h, w)
    finally:
        _end_panel()


def _hit_row(ui, c, n, h, w):
    s = ui.s
    rh = 38.0 * s
    sel = n == c.sel_hit
    clicked, _ = imgui.selectable(f"##h{n}", sel, 0, V2(0, rh))
    p0 = imgui.get_item_rect_min()
    p1 = imgui.get_item_rect_max()
    dl = imgui.get_window_draw_list()
    if clicked:
        _choose(ui, c, h)
    if h.kind == "utility":
        dl.add_rect_filled(V2(p0.x + 2, p0.y + 3), V2(p0.x + 6 * s, p1.y - 3), col32((0.34, 0.71, 0.91, 1.0)), 2.0)
        dl.add_text(V2(p0.x + 12 * s, p0.y + 4 * s), col32(theme.TEXT), h.label)
        dl.add_text(V2(p0.x + 12 * s, p0.y + 4 * s + imgui.get_text_line_height()), col32(theme.TEXT_DIM), "utility: " + h.sub)
        return
    if h.kind == "street":
        dl.add_text(V2(p0.x + 12 * s, p0.y + 4 * s), col32(theme.TEXT), h.label)
        dl.add_text(V2(p0.x + 12 * s, p0.y + 4 * s + imgui.get_text_line_height()), col32(theme.TEXT_DIM), h.sub + "   (choose to narrow down)")
        return
    cm, tier, src, final = c.hit_summary(h.idx)
    col = _tier_col(tier)
    dl.add_rect_filled(V2(p0.x + 2, p0.y + 3), V2(p0.x + 6 * s, p1.y - 3), col32(col), 2.0)
    dl.add_text(V2(p0.x + 12 * s, p0.y + 4 * s), col32(theme.TEXT), h.label)
    rec_kind = "lot" if c.index.kind[h.idx] == 1 else "building"
    approx = src == tl.INDEX_LABEL and c.index_is_tiers
    sub = rec_kind + ("  |  up to " + (f"{_cm(cm, approx)} {TIER_LABEL[tier]}" if cm >= 10 else "dry") + "  |  " + (src if final else "index (grid), tiles loading..."))
    dl.add_text(V2(p0.x + 12 * s, p0.y + 4 * s + imgui.get_text_line_height()), col32(theme.TEXT_DIM if tier < 1 else col), sub)


# ------------------------------------------------------------------------------------------------ result / facility cards
def _cards(ui, c):
    L, s = ui.L, ui.s
    if not hasattr(c, "bar_rect"):
        return
    x, y, w = c.bar_rect
    y += getattr(c, "bar_h", 38.0 * s) + 8.0 * s
    maxh = L.sy1 - y - 10.0 * s
    if _dd_visible(c):                      # while the typeahead list is open it takes the place of the card
        return
    if c.result is not None:
        _result_card(ui, c, x, y, w, maxh)
    elif c.facility_sel >= 0 and c.facilities is not None:
        _facility_card(ui, c, x, y, w, maxh)


def _close_x(label="##x"):
    imgui.same_line(imgui.get_window_width() - 36.0 * imgui.get_font_size() / 14.0)
    return imgui.small_button("x" + label)


def _result_card(ui, c, x, y, w, maxh):
    res = c.result
    rec = res.rec
    app, s = ui.app, ui.s
    if not _panel(ui, "##result", x, y, w, maxh):
        _end_panel()
        return
    try:
        imgui.push_text_wrap_pos(w - 52.0 * s)
        bold(rec.address or "(no street address)", theme.TEXT)
        imgui.pop_text_wrap_pos()
        if _close_x():
            c.result = None
            _end_panel()
            return
        kind = "Lot - no building footprint (depth at the lot point)" if rec.kind == 1 else "Building / address"
        bits = [kind]
        if rec.use:
            bits.append(f"land use {rec.use}")
        if rec.value:
            bits.append(f"assessed ${rec.value:,}")
        dim("  |  ".join(bits), wrap=True)
        a = res.assessment
        if a is None:
            imgui.spacing()
            dim("Reading the depth tiles..." if c.tiles is not None else "Preparing...")
            if res.error:
                text_c(theme.ERR, res.error)
            return
        rows = a.rows
        flooded = [r for r in rows if r.flooded]
        worst = max(rows, key=lambda r: r.shown_cm) if rows else None
        imgui.spacing()
        if flooded and worst is not None:
            text_c(_tier_col(worst.shown_tier), f"Flooded in {len(flooded)} of {len(rows)} storms  |  first: {flooded[0].short}")
            approx = worst.source_kind == "tiers" and worst.shown_source == tl.INDEX_LABEL
            dim(f"deepest: {worst.short}  {_depth(worst.shown_cm, worst.shown_tier, approx)}  ({TIER_LABEL[worst.shown_tier]})")
        else:
            text_c(theme.OK, f"Below 10 cm in all {len(rows)} modelled storms")
            dim("a model result - not a guarantee of safety", wrap=True)
        imgui.spacing()
        both = any(r.tile_cm is not None for r in rows)
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.borders_inner_h
        ncol = 6 if both else 4
        if imgui.begin_table("##storms", ncol, int(flags)):
            try:
                TC = imgui.TableColumnFlags_
                imgui.table_setup_column("Storm", TC.width_fixed, 104.0 * s)
                imgui.table_setup_column("Depth" if not both else "Tiles", TC.width_fixed, 66.0 * s)
                imgui.table_setup_column("Tier", TC.width_fixed, 66.0 * s)
                if both:
                    imgui.table_setup_column("Index", TC.width_fixed, 58.0 * s)
                    imgui.table_setup_column("Diff", TC.width_fixed, 56.0 * s)
                imgui.table_setup_column("Flooded", TC.width_fixed, 64.0 * s)
                imgui.table_headers_row()
                for r in rows:
                    _storm_row(ui, c, r, both)
            finally:
                imgui.end_table()
        imgui.spacing()
        src = a.tile_label or tl.INDEX_LABEL
        text_c(_source_colour(src), "Depth source: " + src)
        if a.tile_label:
            dim("Index (grid) is shown for comparison: " + tl.INDEX_LABEL, wrap=True)
        elif any(r.source_kind == "tiers" for r in rows):
            text_c(theme.WARN, "Preview from depth tiers: severe means 1.2 m or more, not a depth")
        for n in a.notes[:3]:
            dim(n, wrap=True)
        imgui.spacing()
        imgui.push_text_wrap_pos(w - 24.0 * s)
        imgui.text_colored(V4(*theme.WARN), "Model result, not validated against observed floods. Depths are maxima over the "
                           "footprint and over all hours of each storm.")
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        if imgui.button("Fly to"):
            c.fly_to(rec.lng, rec.lat, 130.0)
        imgui.same_line()
        if imgui.button("Save risk report"):
            c.request_report()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Self-contained HTML + PNG of the view in Documents\\BG_Flood_Reports   (Ctrl+P)")
        if c.last_report:
            imgui.same_line()
            if imgui.button("Open report"):
                _startfile(ui, c.last_report["html"])
    finally:
        _end_panel()


def _storm_row(ui, c, r, both):
    cur = (r.key == c.storm_key())
    imgui.table_next_row()
    imgui.table_next_column()
    p = imgui.get_cursor_screen_pos()
    dl = imgui.get_window_draw_list()
    col = _tier_col(r.shown_tier) if r.shown_tier else theme.TEXT_DIM
    if cur:
        dl.add_rect_filled(V2(p.x - 4, p.y - 1), V2(p.x - 1, p.y + imgui.get_text_line_height() + 1), col32(theme.ACCENT), 1.0)
    clicked, _ = imgui.selectable(f"{r.short}##r{r.key}", cur, int(imgui.SelectableFlags_.span_all_columns), V2(0, 0))
    if clicked:
        ui.app.set_storm(r.key)
    if imgui.is_item_hovered():
        imgui.set_tooltip(f"{r.label}: {r.shown_source}\nclick to show this storm in the viewer")
    imgui.table_next_column()
    approx = r.source_kind == "tiers" and r.shown_source == tl.INDEX_LABEL
    imgui.text_colored(V4(*col), _depth(r.shown_cm, r.shown_tier, approx))
    imgui.table_next_column()
    imgui.text_colored(V4(*col), TIER_LABEL[r.shown_tier] if r.shown_tier else "-")
    if both:
        imgui.table_next_column()
        ic = r.index.depth_cm
        imgui.text_colored(V4(*theme.TEXT_DIM), f"{ic} cm" if ic >= 1 else "dry")
        imgui.table_next_column()
        if r.comparison is not None and r.comparison.delta_cm is not None:
            d = r.comparison.delta_cm
            imgui.text_colored(V4(*(theme.WARN if r.comparison.disagree else theme.TEXT_DIM)), f"{d:+d}" + (" !" if r.comparison.disagree else ""))
            if imgui.is_item_hovered():
                imgui.set_tooltip(("Tiles and index put this address in different tiers.\n" if r.comparison.disagree else "")
                                  + f"tiles {r.tile_cm} cm ({r.tile_cells} cell{'s' if r.tile_cells != 1 else ''}), index {r.index.depth_cm} cm")
        else:
            imgui.text_colored(V4(*theme.TEXT_MUTE), "-")
    imgui.table_next_column()
    fh, hh = r.index.first_hour, r.index.hours
    imgui.text_colored(V4(*theme.TEXT_DIM), f"h{fh} +{hh}h" if fh else "-")
    if imgui.is_item_hovered() and fh:
        imgui.set_tooltip(f"first flooded in hour {fh}, flooded for {hh} hours (index: flow-grid hourly maps)")


def _facility_card(ui, c, x, y, w, maxh):
    fs = c.facilities
    i = c.facility_sel
    if not (0 <= i < fs.n):
        c.facility_sel = -1
        return
    s = ui.s
    info = fs.info(i, c.storm_key())
    label, rgb, shape = fac.style(info["type"])
    if not _panel(ui, "##facility", x, y, w, maxh):
        _end_panel()
        return
    try:
        p = imgui.get_cursor_screen_pos()
        _shape(imgui.get_window_draw_list(), shape, p.x + 9 * s, p.y + 10 * s, 6.5 * s, (*rgb, 1.0), (0.04, 0.05, 0.1, 1.0))
        imgui.set_cursor_screen_pos(V2(p.x + 24 * s, p.y))
        imgui.push_text_wrap_pos(w - 52.0 * s)
        bold(info["name"], theme.TEXT)
        imgui.pop_text_wrap_pos()
        if _close_x("##fx"):
            c.facility_sel = -1
            _end_panel()
            return
        dim(f"{info['type_label']}  |  source: {info['source'] or 'unknown'} {info['osm_id']}", wrap=True)
        imgui.spacing()
        cur = info["current"]
        if cur is not None:
            cm = cur["effective_cm"]
            t = tl.tier_of_cm(cm)
            text_c(_tier_col(t), f"{cur['storm']}: " + (f"{_cm(cm, fs.is_approx(cur['storm']))}  {TIER_LABEL[t]}" if cm >= 10 else "not flooded (below 10 cm)"))
            text_c(_source_colour(cur["source"]), cur["source"])
            if fs.is_approx(cur["storm"]):
                text_c(theme.WARN, "Preview from depth tiers: severe means 1.2 m or more, not a depth")
        ff = info["first_flooded"]
        text_c(theme.WARN if ff else theme.OK, f"First flooded in: {ff}" if ff else "Not flooded (>= 10 cm) in any modelled storm")
        mx, ms = info["max_cm"], info["max_storm"]
        if ms:
            dim(f"Deepest: {_cm(mx, fs.is_approx(ms))} in {ms}")
        imgui.spacing()
        has_t = any(p_["tile_cm"] is not None for p_ in info["per_storm"])
        ncol = 5 if has_t else 2
        if imgui.begin_table("##fst", ncol, int(imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.borders_inner_h)):
            try:
                TC = imgui.TableColumnFlags_
                imgui.table_setup_column("Storm", TC.width_stretch, 1.0)
                if has_t:
                    imgui.table_setup_column("Tiles", TC.width_fixed, 58.0 * s)
                    imgui.table_setup_column("Index", TC.width_fixed, 58.0 * s)
                    imgui.table_setup_column("Diff", TC.width_fixed, 52.0 * s)
                    imgui.table_setup_column("<=30 m", TC.width_fixed, 60.0 * s)
                else:
                    imgui.table_setup_column("Index", TC.width_fixed, 70.0 * s)
                imgui.table_headers_row()
                for p_ in info["per_storm"]:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    cur_s = p_["storm"] == c.storm_key()
                    sel_, _ = imgui.selectable(f"{p_['storm']}##f{p_['storm']}", cur_s, int(imgui.SelectableFlags_.span_all_columns))
                    if sel_:
                        ui.app.set_storm(p_["storm"])
                    t = tl.tier_of_cm(p_["effective_cm"])
                    if has_t:
                        imgui.table_next_column()
                        imgui.text_colored(V4(*_tier_col(t)), "-" if p_["tile_cm"] is None else f"{p_['tile_cm']} cm")
                        imgui.table_next_column()
                        imgui.text_colored(V4(*theme.TEXT_DIM), f"{p_['index_cm']} cm")
                        imgui.table_next_column()
                        cmp_ = p_["comparison"]
                        if cmp_ is not None:
                            imgui.text_colored(V4(*(theme.WARN if cmp_.disagree else theme.TEXT_DIM)), f"{cmp_.delta_cm:+d}" + (" !" if cmp_.disagree else ""))
                        else:
                            imgui.text_colored(V4(*theme.TEXT_MUTE), "-")
                        imgui.table_next_column()
                        imgui.text_colored(V4(*theme.TEXT_DIM), "-" if p_["near_cm"] is None else f"{p_['near_cm']} cm")
                    else:
                        imgui.table_next_column()
                        imgui.text_colored(V4(*_tier_col(t)), _cm(p_["index_cm"], fs.is_approx(p_["storm"])) if p_["index_cm"] >= 1 else "dry")
            finally:
                imgui.end_table()
        imgui.spacing()
        if has_t:
            dim("Tiles: the cell under the facility point. Index (grid): max over the campus outline on the flow grid. "
                "<=30 m: deepest tile cell within 30 m of the point. They can differ strongly for large sites.", wrap=True)
        else:
            dim("Index (grid): flow-grid maximum at the facility (a campus = max over its outline).", wrap=True)
        imgui.push_text_wrap_pos(w - 24.0 * s)
        imgui.text_colored(V4(*theme.WARN), "Model result, not validated against observed floods.")
        imgui.pop_text_wrap_pos()
        if imgui.button("Fly to"):
            c.fly_to(info["lng"], info["lat"], 400.0)
    finally:
        _end_panel()


# ------------------------------------------------------------------------------------------------ facilities window
def _facility_window(ui, c):
    if not c.show_fac:
        return
    L, s = ui.L, ui.s
    fw = min(380.0 * s, (L.sx1 - L.sx0) - 28.0 * s)
    imgui.set_next_window_pos(V2(L.sx1 - fw - 14.0 * s, L.sy0 + 14.0 * s), int(imgui.Cond_.appearing))
    imgui.set_next_window_size(V2(fw, min(560.0 * s, L.sy1 - L.sy0 - 60.0 * s)), int(imgui.Cond_.appearing))
    imgui.set_next_window_bg_alpha(0.97)
    vis, opened = imgui.begin("Critical facilities", True, WIN_FLAGS)
    try:
        if vis:
            _facility_body(ui, c)
    finally:
        imgui.end()
    if opened is False:
        c.show_fac = False


def _facility_body(ui, c):
    s = ui.s
    fs = c.facilities
    st = c.status["facilities"]
    if fs is None:
        if st.state in ("missing", "error"):
            text_c(theme.WARN, "Facilities are not available")
            dim(st.message, wrap=True)
        else:
            dim("Loading facilities...")
        return
    storm = c.storm_key()
    ch, v = imgui.checkbox("Show markers on the map  (M)", c.markers_on)
    if ch:
        c.markers_on = bool(v)
        c.persist()
    n_fl, n = fs.counts(storm)
    text_c(theme.WARN if n_fl else theme.OK, f"{n_fl} of {n} facilities flooded (>= 10 cm) in {storm}")
    src = fs.source(storm)
    text_c(_source_colour(src), src)
    if fs.is_approx(storm):
        text_c(theme.WARN, "Preview from depth tiers: severe means 1.2 m or more, not a depth")
    imgui.spacing()
    counts = {}
    for t in fs.types:
        counts[t] = counts.get(t, 0) + 1
    changed = False
    first = True
    for t in sorted(counts, key=lambda k: -counts[k]):
        label, rgb, _ = fac.style(t)
        text = f"{label} {counts[t]}"
        need = imgui.calc_text_size(text).x + 30.0 + imgui.get_style().item_spacing.x
        if not first and imgui.get_content_region_avail().x > need:
            imgui.same_line()
        first = False
        if chip_toggle(f"##ft{t}", text, t not in c.types_off, (*rgb, 1.0)):
            (c.types_off.discard if t in c.types_off else c.types_off.add)(t)
            changed = True
    if changed:
        c.persist()
    imgui.spacing()
    eff = fs.depth_cm(storm)
    order = fs.order(storm)
    avail = imgui.get_content_region_avail()
    if imgui.begin_table("##fl", 3, int(imgui.TableFlags_.scroll_y | imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit), V2(0, max(avail.y - 8.0 * s, 80.0))):
        try:
            TC = imgui.TableColumnFlags_
            imgui.table_setup_column("Facility", TC.width_stretch, 1.0)
            imgui.table_setup_column(storm, TC.width_fixed, 66.0 * s)
            imgui.table_setup_column("First", TC.width_fixed, 52.0 * s)
            imgui.table_setup_scroll_freeze(0, 1)
            imgui.table_headers_row()
            clip = imgui.ListClipper()
            clip.begin(len(order))
            while clip.step():
                for k in range(clip.display_start, clip.display_end):
                    i = int(order[k])
                    f = fs.items[i]
                    label, rgb, shape = fac.style(f.type)
                    cm = int(eff[i])
                    t = tl.tier_of_cm(cm)
                    imgui.table_next_row()
                    imgui.table_next_column()
                    clicked, _ = imgui.selectable(f"##fr{i}", i == c.facility_sel, int(imgui.SelectableFlags_.span_all_columns), V2(0, 0))
                    if clicked:
                        c.select_facility(i, fly=True)
                    imgui.same_line(0.0, 0.0)
                    p = imgui.get_cursor_screen_pos()
                    _shape(imgui.get_window_draw_list(), shape, p.x + 7 * s, p.y + imgui.get_text_line_height() * 0.5, 4.8 * s, (*rgb, 1.0), (0.04, 0.05, 0.1, 1.0))
                    imgui.set_cursor_screen_pos(V2(p.x + 20 * s, p.y))
                    imgui.text(f.name or f"({label})")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(f"{f.name or '(unnamed)'}\n{label}")
                    imgui.table_next_column()
                    imgui.text_colored(V4(*_tier_col(t)), _cm(cm, fs.is_approx(storm)) if cm >= 10 else "-")
                    imgui.table_next_column()
                    ff = fs.first_flooded(i)
                    imgui.text_colored(V4(*theme.TEXT_DIM), ff or "-")
            clip.end()
        finally:
            imgui.end_table()


# ------------------------------------------------------------------------------------------------ roads window
def _roads_window(ui, c):
    if not c.show_roads:
        return
    L, s = ui.L, ui.s
    rw = min(540.0 * s, (L.sx1 - L.sx0) - 28.0 * s)
    imgui.set_next_window_pos(V2(L.sx1 - rw - 14.0 * s - (394.0 * s if c.show_fac else 0.0), L.sy0 + 14.0 * s), int(imgui.Cond_.appearing))
    imgui.set_next_window_size(V2(rw, min(560.0 * s, L.sy1 - L.sy0 - 60.0 * s)), int(imgui.Cond_.appearing))
    imgui.set_next_window_bg_alpha(0.97)
    vis, opened = imgui.begin("Roads at risk", True, WIN_FLAGS)
    try:
        if vis:
            _roads_body(ui, c)
    finally:
        imgui.end()
    if opened is False:
        c.show_roads = False


COLS = ("name", "class", "flooded_pct", "max_cm", "len_m", "evac")


def _roads_body(ui, c):
    s = ui.s
    storm = c.storm_key()
    t = c.road_table(storm)
    if t is None:
        dim(f"Loading the road list for {storm}...")
        return
    if isinstance(t, str):
        text_c(theme.WARN, "No road list for this storm")
        dim(t, wrap=True)
        return
    text_c(theme.TEXT, f"{t.label}")
    dim(f"{t.n:,} roads with water | {t.flooded_km:,.1f} km flooded of {t.total_km:,.0f} km | source: "
        f"{'solver depth rasters' if t.source == 'rasters' else 'depth-tier preview (approximate)'}", wrap=True)
    dim("Values are maxima over the whole storm (flow-grid maximum), not the hour on screen.", wrap=True)
    imgui.spacing()
    imgui.set_next_item_width(160.0 * s)
    chg, txt = imgui.input_text_with_hint("##rtxt", "name or class", c.road_text)
    if chg:
        c.road_text = txt[:60]
    imgui.same_line()
    imgui.set_next_item_width(120.0 * s)
    chg, v = imgui.slider_float("min flooded %", c.road_min_pct, 0.0, 100.0, "%.0f%%")
    if chg:
        c.road_min_pct = float(v)
    imgui.same_line()
    chg, v = imgui.checkbox("Evacuation only", c.road_evac)
    if chg:
        c.road_evac = bool(v)
    key = (id(t), c.road_sort, c.road_asc, round(c.road_min_pct, 1), c.road_evac, c.road_text)
    if getattr(c, "_road_view_key", None) != key:
        c._road_view = t.view(c.road_sort, c.road_asc, c.road_min_pct, c.road_evac, c.road_text)
        c._road_view_key = key
    view = c._road_view
    dim(f"{len(view):,} shown", )
    if c.status["geometry"].state == "missing":
        text_c(theme.WARN, "Zoom to road is unavailable: " + c.status["geometry"].message)
    avail = imgui.get_content_region_avail()
    flags = (imgui.TableFlags_.scroll_y | imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.sortable
             | imgui.TableFlags_.resizable)
    if imgui.begin_table("##roads", 6, int(flags), V2(0, max(avail.y - 8.0 * s, 80.0))):
        try:
            TC = imgui.TableColumnFlags_
            imgui.table_setup_column("Road", TC.width_stretch, 1.0, 0)
            imgui.table_setup_column("Class", TC.width_fixed | TC.no_sort, 78.0 * s, 1)
            imgui.table_setup_column("Flooded", TC.width_fixed | TC.prefer_sort_descending | TC.default_sort, 66.0 * s, 2)
            imgui.table_setup_column("Max depth", TC.width_fixed | TC.prefer_sort_descending, 76.0 * s, 3)
            imgui.table_setup_column("Length", TC.width_fixed | TC.prefer_sort_descending, 64.0 * s, 4)
            imgui.table_setup_column("Evac", TC.width_fixed | TC.prefer_sort_descending, 46.0 * s, 5)
            imgui.table_setup_scroll_freeze(0, 1)
            imgui.table_headers_row()
            specs = imgui.table_get_sort_specs()
            if specs is not None and specs.specs_count > 0:
                sp = specs.get_specs(0)
                col = COLS[int(sp.column_index)] if 0 <= int(sp.column_index) < len(COLS) else "flooded_pct"
                asc = int(sp.sort_direction) == int(imgui.SortDirection_.ascending)
                if col in ("class",):
                    col = c.road_sort
                if (col, asc) != (c.road_sort, c.road_asc):
                    c.road_sort, c.road_asc = col, asc
            clip = imgui.ListClipper()
            clip.begin(len(view))
            while clip.step():
                for k in range(clip.display_start, clip.display_end):
                    _road_row(ui, c, t, int(view[k]))
            clip.end()
        finally:
            imgui.end_table()


def _road_row(ui, c, t, i):
    s = ui.s
    sel = c.road_sel is not None and c.road_sel["id"] == int(t.id[i])
    imgui.table_next_row()
    imgui.table_next_column()
    clicked, _ = imgui.selectable(f"##rr{i}", sel, int(imgui.SelectableFlags_.span_all_columns), V2(0, 0))
    if clicked:
        ok = c.select_road(t, i)
        if not ok and c.geometry is not None:
            ui.toasts.push("No geometry for this road", theme.WARN)
    if imgui.is_item_hovered():
        tip = f"{t.name(i)}  (OSM way {int(t.id[i])})\n{t.class_name(i)}; {t.len_m[i]:,.0f} m long; wet {t.wet_pct[i]:.0f}% / flooded {t.flooded_pct[i]:.0f}%"
        if t.evac_text(i):
            tip += "\n" + t.evac_text(i)
        imgui.set_tooltip(tip)
    imgui.same_line(0.0, 0.0)
    imgui.text(t.name(i))
    imgui.table_next_column()
    imgui.text_colored(V4(*theme.TEXT_DIM), t.class_name(i))
    imgui.table_next_column()
    pct = float(t.flooded_pct[i])
    imgui.text_colored(V4(*(theme.WARN if pct >= 50 else theme.TEXT)), f"{pct:.0f}%")
    imgui.table_next_column()
    mx = int(t.max_cm[i])
    imgui.text_colored(V4(*_tier_col(tl.tier_of_cm(mx))), _cm(mx, t.source == "tiers") if mx >= 1 else "-")
    imgui.table_next_column()
    L_ = float(t.len_m[i])
    imgui.text_colored(V4(*theme.TEXT_DIM), f"{L_ / 1000:.1f} km" if L_ >= 1000 else f"{L_:.0f} m")
    imgui.table_next_column()
    e = int(t.evac[i])
    imgui.text_colored(V4(*(theme.ACCENT2 if e else theme.TEXT_MUTE)), "tag" if e == 1 else "major" if e == 2 else "-")


# ------------------------------------------------------------------------------------------------ first-run hints
def _is_offscreen(app) -> bool:
    return "offscreen" in type(getattr(app, "canvas", None)).__name__.lower()


def _onboarding(ui, c):
    if c.onboarded:
        return
    if _is_offscreen(ui.app) and not (os.environ.get("BGV_FORCE_HINT") or c.hint_req):
        return
    if ui.frame_ct < 20 or ui.app.loading:
        return
    L, s = ui.L, ui.s
    dl = imgui.get_foreground_draw_list()
    dl.add_rect_filled(V2(L.sx0, L.sy0), V2(L.sx1, L.sy1), col32((0.0, 0.0, 0.05, 0.38)))
    x, y, w = getattr(c, "bar_rect", (L.sx0 + 200, L.sy0 + 10, 400))
    hb = getattr(c, "bar_h", 38.0 * s)
    items = [
        ("1", "Search any address", "Ctrl+F, then a number and a street", (x + w * 0.5, y + hb + 4 * s), (x + w * 0.5 - 150 * s, y + hb + 60 * s)),
        ("2", "Pick a storm and layers", "number keys 1-9, or the sidebar", (L.sx0 + 10 * s, L.sy0 + 220 * s), (L.sx0 + 36 * s, L.sy0 + 250 * s)),
        ("3", "Click a building", "depth, tier and history by hour", ((L.sx0 + L.sx1) * 0.5, (L.sy0 + L.sy1) * 0.62), ((L.sx0 + L.sx1) * 0.5 - 120 * s, (L.sy0 + L.sy1) * 0.62 + 20 * s)),
        ("4", "Facilities, roads, reports", "M, L and Ctrl+P - or the Find menu", (L.sx1 - 30 * s, L.sy0 + 40 * s), (L.sx1 - 340 * s, L.sy0 + 70 * s)),
    ]
    ph = 46.0 * s
    for num, title, sub, tip, box in items:
        bw = max(imgui.calc_text_size(title).x, imgui.calc_text_size(sub).x) + 62.0 * s
        bx = min(max(box[0], L.sx0 + 8 * s), L.sx1 - bw - 8 * s)
        by = min(max(box[1], L.sy0 + 8 * s), L.sy1 - ph - 8 * s)
        dl.add_line(V2(bx + 20 * s, by), V2(tip[0], tip[1]), col32((1, 1, 1, 0.55)), 1.5)
        dl.add_circle_filled(V2(tip[0], tip[1]), 4.0 * s, col32((1, 1, 1, 0.9)))
        dl.add_rect_filled(V2(bx, by), V2(bx + bw, by + ph), col32((0.05, 0.08, 0.16, 0.96)), 9.0)
        dl.add_rect(V2(bx, by), V2(bx + bw, by + ph), col32(theme.ACCENT), 9.0, 1.4)
        dl.add_circle_filled(V2(bx + 20 * s, by + ph * 0.5), 12.0 * s, col32(theme.ACCENT))
        nt = imgui.calc_text_size(num)
        dl.add_text(V2(bx + 20 * s - nt.x * 0.5, by + ph * 0.5 - nt.y * 0.5), col32((0.02, 0.04, 0.1, 1.0)), num)
        dl.add_text(V2(bx + 42 * s, by + 6 * s), col32(theme.TEXT), title)
        dl.add_text(V2(bx + 42 * s, by + 6 * s + imgui.get_text_line_height()), col32(theme.TEXT_DIM), sub)
    # dismiss button (a real window so it is clickable)
    bw, bh = 330.0 * s, 56.0 * s
    imgui.set_next_window_pos(V2((L.sx0 + L.sx1) * 0.5, L.sy1 - 20.0 * s), int(imgui.Cond_.always), V2(0.5, 1.0))
    imgui.set_next_window_size(V2(bw, 0.0))
    imgui.set_next_window_bg_alpha(0.97)
    imgui.set_next_window_focus() if ui.frame_ct == 20 else None
    vis, _ = imgui.begin("##hint", None, CARD_FLAGS & ~int(imgui.WindowFlags_.no_bring_to_front_on_focus))
    try:
        if vis:
            bold("Welcome to the DSOC Flood Map", theme.ACCENT2)
            dim("Press F1 any time for help and shortcuts.", wrap=True)
            if imgui.button("Got it", V2(100 * s, 0)):
                c.onboarded = True
                c.persist()
            imgui.same_line()
            if imgui.button("Open help", V2(100 * s, 0)):
                c.onboarded = True
                c.persist()
                ui.show_help = True
    finally:
        imgui.end()
