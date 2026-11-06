"""Flood-risk feature pack (one extension module, registered through bgv.ui.registry only):

  * Dry route finder (tool G): shortest route between two clicked points that avoids road pieces with solver depth >= threshold
    at the displayed hour, drawn with the normal route; lengths compared (bgv.data.routing).
  * Building exposure (X): filter by depth tier / land use / value for the displayed storm and hour, table, CSV / GeoJSON export.
  * Compare storms (V): thumbnails of every storm at the same view and hour (offscreen renders, cached), click to switch.
  * Depth-duration (tool U): hours >= 10 / 30 / 60 cm per storm for a clicked point or the selected building, and the
    'hours flooded' heat layer (bgv/gpu/layers/hoursflooded.py).
  * Share card (F9): annotated PNG (title, storm, hour, legend, scale bar, north arrow, honesty stamp, credits).
  * Session (Ctrl+S / Ctrl+O): .bgview files (bgv.data.session).
  * Accessibility: keyboard navigation of the panels (F6 cycles), high-contrast theme, font size independent of the UI scale.
  * UX arbiter: one click-capturing tool at a time (analysis tools vs registry tools), stale state cleared on a domain / storm
    switch, generated F1 cheat sheet (bgv.ui.keymap).
All work that reads tiles runs in worker threads; every number shown carries its source label.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from . import bridge, keymap, registry, theme
from ..data import duration as durmod, exposure as expmod, routing, session as sesmod

V2 = imgui.ImVec2
V4 = imgui.ImVec4
TABS = ("Route", "Exposure", "Duration", "Compare storms", "Share", "Session", "Access")
# Okabe-Ito colours (colour-blind safe, checked in tests/test_features.py with bgv.ui.cvd)
ROUTE_DRY = (0.0, 0.447, 0.698)          # blue
ROUTE_NORMAL = (0.835, 0.369, 0.0)       # vermillion
ROUTE_BLOCK = (0.8, 0.475, 0.655)        # reddish purple: blocked road pieces near the route
DUR_COLOURS = ((0.337, 0.706, 0.914), (0.902, 0.624, 0.0), (0.8, 0.475, 0.655))   # 10 / 30 / 60 cm
HC = dict(BG0="#000000", BG1="#000000", BG2="#101010", BG3="#202020", BORDER="#ffffff", ACCENT="#ffd400", ACCENT2="#ffe866",
          TEXT="#ffffff", TEXT_DIM="#e6e6e6", TEXT_MUTE="#c8c8c8", OK="#5cff9d", WARN="#ffd400", ERR="#ff7b7b")
NAV_WINDOWS = ("Flood tools", "##sidebar", "##timeline", "##atrisk")


def _docs(sub: str) -> Path:
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    return home / "Documents" / sub


class Features:
    def __init__(self):
        self.open = False
        self.tab = 0
        self.tab_request = None
        # route
        self.graph = None
        self.graph_src = ""
        self.graph_job = durmod.Background()
        self.route_job = durmod.Background()
        self.depth_cache = None
        self.route_pts: list = []          # [(lng, lat)] 0..2
        self.route_res = None
        self.route_xy = None               # cached projected polylines
        self.threshold_cm = 30
        # exposure
        self.expo_filter = expmod.ExposureFilter()
        self.expo_job = durmod.Background()
        self.expo_ids = None
        self.expo_msg = ""
        self.expo_lu = []
        self.expo_sel_lu: set = set()
        # duration
        self.dur_job = durmod.Background()
        self.dur_target = None             # ("pt", lng, lat) | ("bld", id)
        # compare storms
        self.thumbs: dict = {}             # storm -> (ImTextureRef, texture, view key)
        self.capture = None
        # share / session
        self.last_msg = ""
        self.pending_session = None
        # accessibility
        self.hc_applied = None
        self.font_applied = None
        self.nav_i = -1
        # arbiter
        self.seen_domain = None
        self.seen_storm = None
        self.prev_an_tool = "none"
        self.help_done = False
        t = os.environ.get("BGV_FEATURES_TAB", "")            # devshot / tests: open the window on a tab
        if t.isdigit():
            self.open, self.tab_request = True, int(t)


def F(ui) -> Features:
    f = getattr(ui, "_features", None)
    if f is None:
        f = ui._features = Features()
    return f


def _st(ui):
    return registry.state(ui.app, "features")


def open_tab(ui, i):
    f = F(ui)
    f.open, f.tab_request = True, i


def toggle_tab(ui, i):
    f = F(ui)
    if f.open and f.tab == i:
        f.open = False
    else:
        open_tab(ui, i)


def svc(app):
    from ..data import depthapi
    return depthapi.service(app)


def tile_label(app, storm=None):
    ts = svc(app).tileset(storm or app.storm.name)
    return ts.label if ts is not None else ""


def hour_now(app) -> int:
    return int(round(app.t))


# ================================================================================================ world <-> screen
def ll_to_world(fr, lng, lat):
    """Vectorised GridFrame.lnglat_to_world (the frame's own method takes scalars)."""
    from ..geo import R_MERC
    x = R_MERC * np.radians(np.asarray(lng, float))
    y = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(np.asarray(lat, float)) / 2))
    return fr.merc_to_world(x, y)


def project(app, X, Y, lift=3.0, Z=None):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if Z is None:
        Z = np.asarray(app.terrain.sampler.sample(X, Y), float) * float(app.exag) + lift
    P = np.stack([X, Y, Z, np.ones_like(X)], 1) @ np.asarray(getattr(app, "view_proj", None) if getattr(app, "view_proj", None) is not None else app.camera.view_proj(), float).T
    w = P[:, 3]
    ok = w > 1e-6
    cam = app.camera
    sx = (P[:, 0] / np.where(ok, w, 1) * 0.5 + 0.5) * cam.width
    sy = (0.5 - P[:, 1] / np.where(ok, w, 1) * 0.5) * cam.height
    return sx, sy, ok


def _polyline(dl, sx, sy, ok, col, width):
    """Draw the visible runs of a projected polyline; points closer than ~2 px on screen are merged (a county route has
    thousands of OSM nodes), so the per-frame Python work stays small."""
    sx, sy, ok = np.asarray(sx), np.asarray(sy), np.asarray(ok, bool)
    if sx.size < 2:
        return
    q = np.floor(sx / 2.0) * 1e5 + np.floor(sy / 2.0)
    keep = np.r_[True, q[1:] != q[:-1]] | ~ok | np.r_[~ok[1:], True] | np.r_[True, ~ok[:-1]]
    keep[-1] = True
    sx, sy, ok = sx[keep], sy[keep], ok[keep]
    cuts = np.flatnonzero(np.diff(ok.astype(np.int8)) != 0) + 1
    c_out, c_in = theme.col32((0, 0, 0, 0.8)), theme.col32(col)
    for seg in np.split(np.arange(sx.size), cuts):
        if seg.size > 1 and ok[seg[0]]:
            run = [V2(x, y) for x, y in zip(sx[seg].tolist(), sy[seg].tolist())]
            dl.add_polyline(run, c_out, width + 3.0, 0)
            dl.add_polyline(run, c_in, width, 0)


# ================================================================================================ routing
def _graph_source(app):
    root = Path(app.root)
    pbf = root / "kentucky-latest.osm.pbf"
    roads = root / "layers" / ("county" if app.domain.key == "county" else "") / "roads.geojson.gz"
    if not roads.is_file():
        roads = root / "layers" / "roads.geojson"
    return pbf, roads


def ensure_graph(ui):
    f, app = F(ui), ui.app
    if f.graph is not None or f.graph_job.busy:
        return
    if f.graph_job.result is not None and f.graph_job.key == app.domain.key:
        f.graph, f.graph_src = f.graph_job.result
        f.depth_cache = routing.DepthCache(f.graph, svc(app))
        return
    pbf, roads = _graph_source(app)
    fr = app.terrain.frame
    lo0, la0 = fr.world_to_lnglat(-fr.size_m[0] / 2, -fr.size_m[1] / 2)
    lo1, la1 = fr.world_to_lnglat(fr.size_m[0] / 2, fr.size_m[1] / 2)
    bbox = (min(lo0, lo1), min(la0, la1), max(lo0, lo1), max(la0, la1))
    from .. import config

    def job(progress):
        if pbf.is_file():
            g = routing.load_osm_graph(pbf, bbox, config.cache_dir() / "routing", progress)
            return g, f"OpenStreetMap ({pbf.name}): {g.n:,} junctions, {int(g.bridge.sum())} bridge / tunnel pieces"
        g = routing.load_graph(roads)
        return g, f"{roads.name} (simplified: junctions partly lost, many roads disconnected)"
    f.graph_job.start(app.domain.key, job)


def plan_route(ui):
    f, app = F(ui), ui.app
    if f.graph is None or len(f.route_pts) < 2:
        return
    storm, h, thr = app.storm.name, hour_now(app), f.threshold_cm / 100.0
    key = (storm, h, thr, tuple(f.route_pts))
    lab = tile_label(app, storm)
    g, cache, a, b = f.graph, f.depth_cache, f.route_pts[0], f.route_pts[1]
    f.route_job.key = None if f.route_job.key != key else f.route_job.key
    f.route_job.start(key, lambda p: routing.plan(g, cache, a, b, storm, h, thr, lab))


class RouteTool(registry.Tool):
    name = "route"
    title = "Dry route finder"
    shortcut = "G"
    hint = "Click the start, then the destination (Esc leaves)"

    def on_activate(self, ui):
        open_tab(ui, 0)
        ensure_graph(ui)

    def on_click(self, ui, x, y, pick):
        f = F(ui)
        if pick is None:
            return True
        ll = ui.app.terrain.frame.world_to_lnglat(float(pick[0]), float(pick[1]))
        if len(f.route_pts) >= 2:
            f.route_pts, f.route_res = [], None
        f.route_pts.append((float(ll[0]), float(ll[1])))
        plan_route(ui)
        return True


def route_tick(ui):
    f = F(ui)
    if f.graph is None and f.graph_job.result is not None:
        ensure_graph(ui)
    j = f.route_job
    if j.result is not None and not j.busy:
        if j.key is not None and j.key[0] == ui.app.storm.name:
            f.route_res = j.result
    if f.graph is not None and len(f.route_pts) == 2:
        want = (ui.app.storm.name, hour_now(ui.app), f.threshold_cm / 100.0, tuple(f.route_pts))
        if j.key != want and not j.busy and not ui.app.playing:      # re-plan when the time stops (a re-plan reads ~1.4 M tile samples)
            plan_route(ui)


def draw_route_overlay(ui):
    f, app = F(ui), ui.app
    if not f.route_pts:
        return
    dl = imgui.get_background_draw_list()
    fr = app.terrain.frame
    r = f.route_res
    if r is not None and f.graph is not None:
        g = f.graph
        ck = (id(r), id(app.terrain), float(app.exag))
        if f.__dict__.get("route_geo_key") != ck:                 # world X, Y, Z of both routes: once per result
            geo = []
            for route in (r.normal, r.dry):
                if route.found:
                    X, Y = ll_to_world(fr, g.lng[route.nodes], g.lat[route.nodes])
                    geo.append((X, Y, np.asarray(app.terrain.sampler.sample(X, Y), float) * float(app.exag) + 4.0))
                else:
                    geo.append(None)
            f.route_geo, f.route_geo_key = geo, ck
        for gg, col, wdt in zip(f.route_geo, (ROUTE_NORMAL, ROUTE_DRY), (4.0, 6.0)):
            if gg is not None:
                sx, sy, ok = project(app, gg[0], gg[1], Z=gg[2])
                _polyline(dl, sx, sy, ok, col + (1.0,), wdt * ui.s)
    for i, (lng, lat) in enumerate(f.route_pts):
        X, Y = fr.lnglat_to_world(lng, lat)
        sx, sy, ok = project(app, [X], [Y], 4.0)
        if ok[0]:
            c = V2(float(sx[0]), float(sy[0]))
            dl.add_circle_filled(c, 9 * ui.s, theme.col32((0, 0, 0, 0.85)))
            dl.add_circle_filled(c, 7 * ui.s, theme.col32((1, 1, 1, 1)))
            dl.add_text(V2(c.x - 4 * ui.s, c.y - 7 * ui.s), theme.col32((0, 0, 0, 1)), "AB"[i])


def tab_route(ui):
    f, app = F(ui), ui.app
    ensure_graph(ui)
    if f.graph is None:
        if f.graph_job.error:
            imgui.text_colored(V4(*theme.ERR), f"Road graph failed: {f.graph_job.error}")
        else:
            imgui.text_disabled(f"Building the road graph ... {100 * f.graph_job.progress:.0f} %  (first time about a minute; cached)")
    else:
        imgui.text_wrapped(f"Graph: {f.graph_src}")
    imgui.set_next_item_width(160 * ui.s)
    ch, v = imgui.slider_int("Avoid depth >= (cm)", int(f.threshold_cm), 5, 120)
    if ch:
        f.threshold_cm = int(v)
    on = ui.tool is not None and ui.tool.name == "route"
    if imgui.button("Stop picking (Esc)" if on else "Pick start and destination (G)"):
        ui.toggle_tool("route")
    imgui.same_line()
    if imgui.button("Clear route"):
        f.route_pts, f.route_res = [], None
    if svc(app).tileset(app.storm.name) is None:
        imgui.text_colored(V4(*theme.WARN), "No solver tiles for this storm: depths unknown, every road counts as passable.")
    r = f.route_res
    if f.route_job.busy:
        imgui.text_disabled("Routing ...")
    if r is not None:
        imgui.text_wrapped(r.summary())
        _chip(ROUTE_DRY, "dry route")
        imgui.same_line()
        _chip(ROUTE_NORMAL, "normal route")
        imgui.text_disabled(f"{r.storm}, hour {r.hour}; {r.blocked_count:,} road pieces blocked; start / end snapped {r.a_snap_m:.0f} / {r.b_snap_m:.0f} m")
        imgui.text_disabled(f"Depth: {r.label or 'no tiles'}")
    elif len(f.route_pts) == 1:
        imgui.text_disabled("Start set; click the destination.")
    imgui.push_text_wrap_pos(0.0)
    imgui.text_disabled("Honesty: " + routing.ROUTE_NOTE)
    imgui.pop_text_wrap_pos()


def _chip(col, text):
    p = imgui.get_cursor_screen_pos()
    s = imgui.get_text_line_height()
    imgui.get_window_draw_list().add_rect_filled(p, V2(p.x + s, p.y + s), theme.col32(tuple(col[:3]) + (1.0,)))
    imgui.dummy(V2(s, s))
    imgui.same_line()
    imgui.text(text)


# ================================================================================================ exposure
def expo_depths(app, storm, hour):
    """(int32 cm per building, label) for the displayed domain: tiles when the footprint index is ready, else the flow grid."""
    s = svc(app)
    if s.tileset(storm) is not None and s.index_for(storm) is not None:
        arr = s.building_depths(storm, hour, block=True)
        if arr is not None:
            return np.asarray(arr), s.tileset(storm).label + " (footprint maximum)"
    fl = bridge.flood(app)
    st = fl.get("buildings") if fl else None
    if st is None or fl.get("storm") != storm:
        return None, "statistics not ready"
    return np.round(np.asarray(st.depth_m, float) * 100).astype(np.int32), "flow-grid depth (20-25 m cells), current statistics hour"


def tab_exposure(ui):
    f, app = F(ui), ui.app
    b = bridge.buildings(app)
    if b is None:
        imgui.text_disabled("Buildings are not loaded.")
        return
    flt = f.expo_filter
    tiers = list(flt.tiers)
    for k, nm in enumerate(expmod.TIER_NAMES):
        if k:
            imgui.same_line()
        ch, v = imgui.checkbox(nm.capitalize() + "##xt", tiers[k])
        if ch:
            tiers[k] = v
    flt.tiers = tuple(tiers)
    if not f.expo_lu or f.expo_lu[0] is not b:
        f.expo_lu = (b, expmod.land_use_values(b.land_use))
    lus = f.expo_lu[1]
    if len(lus) == 1 and lus[0][0] == expmod.UNKNOWN:
        imgui.text_disabled("Land use: not in this building file (county OSM footprints carry none)")
    elif imgui.tree_node(f"Land use ({len(f.expo_sel_lu) or 'all'})"):
        for name, n in lus[:30]:
            ch, v = imgui.checkbox(f"{name} ({n})##lu", name in f.expo_sel_lu)
            if ch:
                (f.expo_sel_lu.add if v else f.expo_sel_lu.discard)(name)
        imgui.tree_pop()
    flt.land_uses = tuple(sorted(f.expo_sel_lu))
    known = int(np.sum(np.isfinite(b.sale_price) & (b.sale_price > 0)))
    if known:
        imgui.set_next_item_width(120 * ui.s)
        ch, v = imgui.input_float("Min value $", float(flt.min_value), 10000.0, 100000.0, "%.0f")
        if ch:
            flt.min_value = max(0.0, v)
        imgui.same_line()
        imgui.set_next_item_width(120 * ui.s)
        ch, v = imgui.input_float("Max value $", float(flt.max_value), 10000.0, 100000.0, "%.0f")
        if ch:
            flt.max_value = max(0.0, v)
        imgui.text_disabled(f"Value = last sale price (parcel data), known for {known:,} of {b.n:,} buildings")
    else:
        imgui.text_disabled("Value: no sale prices in this building file")
    storm, h = app.storm.name, hour_now(app)
    peak = _st(ui).get("expo_peak", False)
    ch, peak = imgui.checkbox("Use the storm's peak instead of the displayed hour", bool(peak))
    if ch:
        _st(ui)["expo_peak"] = bool(peak)
    which = "peak" if peak else h
    key = (app.domain.key, storm, which, id(b))
    j = f.expo_job
    if j.key != key and not j.busy:
        j.key = None
        j.start(key, lambda p: expo_depths(app, storm, which))
    if j.busy or j.result is None or j.result[0] is None:
        imgui.text_disabled("Computing building depths ..." if j.busy else (j.result[1] if j.result else ""))
        return
    depth, label = j.result
    if depth.shape[0] != b.n:
        imgui.text_colored(V4(*theme.WARN), "Depth array does not match the buildings (domain switching)")
        return
    ck = (id(depth), id(b), flt.key())
    if f.__dict__.get("expo_ck") != ck or f.expo_ids is None:          # filtering 25 k buildings is ~ms: once per change
        f.expo_ids = expmod.apply(depth, b.land_use, b.sale_price, flt)
        f.expo_sum = expmod.summarize(f.expo_ids, depth, b.sale_price)
        f.expo_ck = ck
    ids, sm = f.expo_ids, f.expo_sum
    imgui.text(f"{sm['count']:,} buildings  -  {storm}, {'peak' if peak else f'hour {h}'}")
    imgui.text_disabled("  ".join(f"{k}: {v:,}" for k, v in sm["by_tier"].items() if v))
    if sm["value_known"]:
        imgui.text_disabled(f"Known value total ${sm['value_sum']:,.0f} ({sm['value_known']:,} buildings)")
    imgui.text_disabled(f"Depth: {label}")
    if imgui.button("Export CSV"):
        _export(ui, ids, depth, label, storm, which, "csv")
    imgui.same_line()
    if imgui.button("Export GeoJSON"):
        _export(ui, ids, depth, label, storm, which, "geojson")
    if f.last_msg:
        imgui.text_disabled(f.last_msg[:160])
    flags = int(imgui.TableFlags_.row_bg | imgui.TableFlags_.scroll_y | imgui.TableFlags_.borders_inner_h)
    if imgui.begin_table("##expo", 4, flags, V2(0, 220 * ui.s)):
        for c in ("Address", "Depth", "Land use", "Value"):
            imgui.table_setup_column(c)
        imgui.table_headers_row()
        clip = imgui.ListClipper()
        clip.begin(int(ids.size))
        while clip.step():
            for i in ids[clip.display_start:clip.display_end]:
                imgui.table_next_row()
                imgui.table_next_column()
                addr = str(b.address[i]) or f"building {int(i)}"
                if imgui.selectable(f"{addr}##e{int(i)}", False, int(imgui.SelectableFlags_.span_all_columns))[0]:
                    bridge.select_building(app, int(i), fly=True)
                imgui.table_next_column()
                imgui.text(f"{int(depth[i])} cm")
                imgui.table_next_column()
                imgui.text(str(b.land_use[i]) or "-")
                imgui.table_next_column()
                p = float(b.sale_price[i])
                imgui.text(f"${p:,.0f}" if np.isfinite(p) and p > 0 else "-")
        imgui.end_table()


def _export(ui, ids, depth, label, storm, which, kind):
    f, app = F(ui), ui.app
    b = bridge.buildings(app)
    rws = expmod.rows(ids, b, depth, app.terrain.frame)
    meta = {"storm": storm, "hour": which, "domain": app.domain.key, "depth_source": label,
            "filter": repr(f.expo_filter.key()), "note": "model result, not validated", "written": time.strftime("%Y-%m-%d %H:%M")}
    p = _docs("BG_Flood_Exports") / f"exposure_{app.domain.key}_{storm}_{which}_{time.strftime('%Y%m%d_%H%M%S')}.{kind}"
    try:
        (expmod.write_csv if kind == "csv" else expmod.write_geojson)(p, rws, meta)
        f.last_msg = f"Saved {len(rws):,} buildings: {p}"
        ui.toasts.push(f.last_msg, theme.OK, 6.0)
    except Exception as e:
        f.last_msg = f"Export failed: {e}"


# ================================================================================================ depth-duration
class DurationTool(registry.Tool):
    name = "duration"
    title = "Depth-duration probe"
    shortcut = "U"
    hint = "Click a point to count the hours under water per storm (Esc leaves)"

    def on_activate(self, ui):
        open_tab(ui, 2)

    def on_click(self, ui, x, y, pick):
        if pick is None:
            return True
        ll = ui.app.terrain.frame.world_to_lnglat(float(pick[0]), float(pick[1]))
        set_duration_target(ui, ("pt", float(ll[0]), float(ll[1])))
        return True


def set_duration_target(ui, tgt):
    f, app = F(ui), ui.app
    f.dur_target = tgt
    s = svc(app)
    storms = list(app.storms_available)
    key = (app.domain.key, tgt)
    f.dur_job.key = None
    if tgt[0] == "pt":
        f.dur_job.start(key, lambda p: durmod.table(s, storms, lnglat=(tgt[1], tgt[2])))
    else:
        f.dur_job.start(key, lambda p: durmod.table(s, storms, bid=int(tgt[1])))


def tab_duration(ui):
    f, app = F(ui), ui.app
    on = ui.tool is not None and ui.tool.name == "duration"
    if imgui.button("Stop picking (Esc)" if on else "Pick a point (U)"):
        ui.toggle_tool("duration")
    sel = bridge.selection(app).get("building")
    imgui.same_line()
    imgui.begin_disabled(sel is None)
    if imgui.button("Selected building"):
        if svc(app).index_for(app.storm.name) is None:
            f.last_msg = "Tile footprint index not ready yet; try again in a moment."
        else:
            set_duration_target(ui, ("bld", int(sel)))
    imgui.end_disabled()
    t = f.dur_target
    if t is not None:
        imgui.text(f"Point {t[2]:.5f}, {t[1]:.5f}" if t[0] == "pt" else f"Building {t[1]}  {bridge.buildings(app).address[t[1]] if bridge.buildings(app) is not None else ''}")
    j = f.dur_job
    if j.busy:
        imgui.text_disabled("Reading the hourly tiles of every storm ...")
    elif j.error:
        imgui.text_colored(V4(*theme.ERR), j.error)
    elif j.result is not None:
        flags = int(imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h)
        if imgui.begin_table("##dur", 4, flags):
            imgui.table_setup_column("Storm")
            for k, thr in enumerate(durmod.THRESHOLDS_CM):
                imgui.table_setup_column(f"h >= {thr} cm")
            imgui.table_headers_row()
            for storm, d in j.result:
                imgui.table_next_row()
                imgui.table_next_column()
                if imgui.selectable(storm + "##ds", storm == app.storm.name, int(imgui.SelectableFlags_.span_all_columns))[0]:
                    app.set_storm(storm)
                for k, thr in enumerate(durmod.THRESHOLDS_CM):
                    imgui.table_next_column()
                    if d is None:
                        imgui.text_disabled("no tiles")
                    else:
                        imgui.text_colored(V4(*DUR_COLOURS[k], 1.0) if d[thr] else V4(*theme.TEXT_DIM), f"{d[thr]} h")
            imgui.end_table()
        imgui.text_disabled(("footprint maximum, " if t and t[0] == "bld" else "point sample, ") + durmod.DURATION_NOTE)
    imgui.separator()
    lay = app.get_layer("hoursflooded")
    if lay is None:
        imgui.text_disabled("Hours-flooded layer not available")
        return
    ch, v = imgui.checkbox("Show 'hours flooded' heat layer for this storm", bool(lay.enabled))
    if ch:
        lay.enabled = bool(v)
    if lay.enabled:
        if lay.status == "computing" or lay._loaded_for != app.storm.name and not lay.error:
            imgui.text_disabled(f"Computing from {len(svc(app).tileset(app.storm.name).hours) if svc(app).tileset(app.storm.name) else 0} hourly tiles ... {100 * lay.progress:.0f} %  (cached on disk afterwards)")
        if lay.error:
            imgui.text_colored(V4(*theme.WARN), lay.error[:120])
        from ..gpu.layers import flowpaths, hoursflooded
        imgui.text_disabled(hoursflooded.LABEL)
        dl = imgui.get_window_draw_list()
        p0 = imgui.get_cursor_screen_pos()
        wbar = imgui.get_content_region_avail().x
        for k in range(24):
            c = flowpaths.ramp_colour("hours", (k + 0.5) / 24)
            dl.add_rect_filled(V2(p0.x + wbar * k / 24, p0.y), V2(p0.x + wbar * (k + 1) / 24 + 1, p0.y + 10 * ui.s), theme.col32(c + (1.0,)))
        imgui.dummy(V2(wbar, 12 * ui.s))
        imgui.text_disabled("1 h                          12 h                         24 h")


# ================================================================================================ compare storms (small multiples)
def view_key(app):
    c = app.camera
    return (app.domain.key, round(float(c.target[0]), 0), round(float(c.target[1]), 0), round(float(c.dist), 0),
            round(float(c.yaw), 1), round(float(c.pitch), 1), hour_now(app))


def start_capture(ui):
    f, app = F(ui), ui.app
    if f.capture is not None or getattr(app, "imgui_renderer", None) is None:
        return
    f.capture = {"storms": list(app.storms_available), "i": 0, "orig": app.storm.name, "t": float(app.t), "wait": 0,
                 "playing": app.playing, "key": view_key(app), "t0": time.perf_counter()}
    app.playing = False


def capture_tick(ui):
    f, app = F(ui), ui.app
    c = f.capture
    if c is None:
        return
    if c["i"] >= len(c["storms"]) or time.perf_counter() - c["t0"] > 240:
        app.set_storm(c["orig"])
        app.t, app.playing = c["t"], c["playing"]
        f.capture = None
        return
    s = c["storms"][c["i"]]
    if app.storm.name != s:
        app.set_storm(s)
        c["wait"] = 0
        return
    app.t = c["t"]
    if not app.flow.ready_for(c["t"]):
        return
    c["wait"] += 1
    if c["wait"] < 5:
        return
    from . import screenshot
    from PIL import Image
    arr = screenshot.capture_rgba(app, include_ui=False)
    h, w = arr.shape[:2]
    tw = 320
    th = max(1, int(round(h * tw / w)))
    small = np.asarray(Image.fromarray(arr).resize((tw, th), Image.BILINEAR))
    _set_thumb(app, f, s, small, c["key"])
    c["i"] += 1


def _set_thumb(app, f, storm, rgba, key):
    import wgpu
    old = f.thumbs.get(storm)
    dev = app.ctx.device
    h, w = rgba.shape[:2]
    fmt = "rgba8unorm-srgb" if str(app.colour_format).endswith("srgb") else "rgba8unorm"
    tex = dev.create_texture(label=f"thumb-{storm}", size=(w, h, 1), format=fmt,
                             usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dev.queue.write_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)}, np.ascontiguousarray(rgba),
                            {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
    view = tex.create_view()
    be = app.imgui_renderer.backend
    ref = be.register_texture(view)
    f.thumbs[storm] = (ref, tex, key, view, (w, h))
    if old is not None:
        _free_thumb(app, old)


def _free_thumb(app, t):
    try:
        app.imgui_renderer.backend.unregister_texture(t[3])
    except Exception:
        pass
    try:
        t[1].destroy()
    except Exception:
        pass


def clear_thumbs(ui):
    f = F(ui)
    for t in f.thumbs.values():
        _free_thumb(ui.app, t)
    f.thumbs.clear()


def tab_compare(ui):
    f, app = F(ui), ui.app
    vk = view_key(app)
    c = f.capture
    if c is not None:
        imgui.text(f"Rendering storm {c['i'] + 1} of {len(c['storms'])} ({c['storms'][min(c['i'], len(c['storms']) - 1)]}) ...")
    else:
        stale = any(t[2] != vk for t in f.thumbs.values())
        if imgui.button("Render all storms at this view and hour" if not f.thumbs else ("Refresh (view or hour changed)" if stale else "Refresh")):
            start_capture(ui)
        imgui.text_disabled("Offscreen renders of the 3D view (no interface), cached until the view or hour changes.")
    if not f.thumbs:
        return
    avail = imgui.get_content_region_avail().x
    cols = 3 if avail > 600 * ui.s else 2
    tw = (avail - (cols - 1) * 8 * ui.s) / cols
    for i, s in enumerate(app.storms_available):
        t = f.thumbs.get(s)
        if i % cols:
            imgui.same_line()
        imgui.begin_group()
        if t is None:
            imgui.dummy(V2(tw, tw * 0.56))
        else:
            w, h = t[4]
            if imgui.image_button(f"##th{s}", t[0], V2(tw, tw * h / w)):
                app.set_storm(s)
        lab = s + ("  (current)" if s == app.storm.name else "") + ("  [old view]" if t is not None and t[2] != vk else "")
        imgui.text(lab)
        imgui.end_group()


# ================================================================================================ share card
def scale_and_north(app):
    """(metres per pixel at the view centre, screen angle of north in degrees clockwise from up) or (None, 0)."""
    cam = app.camera
    cx, cy = cam.width * 0.5, cam.height * 0.5
    a, b = cam.pick(cx - 50, cy), cam.pick(cx + 50, cy)
    mpp = None
    if a is not None and b is not None:
        mpp = math.hypot(b[0] - a[0], b[1] - a[1]) / 100.0
    t = cam.target
    sx, sy, ok = project(app, [t[0], t[0]], [t[1], t[1] + 100.0], 0.0)
    ang = math.degrees(math.atan2(sx[1] - sx[0], -(sy[1] - sy[0]))) if ok.all() else 0.0
    return mpp, ang


def share_card(ui):
    from . import screenshot, sharecard
    app = ui.app
    arr = screenshot.capture_rgba(app, include_ui=False)
    pw, _ = app.canvas.get_physical_size()
    mpp, north = scale_and_north(app)
    if mpp is not None:
        mpp = mpp * app.camera.width / max(pw, 1)          # logical -> physical pixels
    lab = tile_label(app)
    leg = [(f"{n}  {lo}-{hi} cm" if hi else f"{n}  >= {lo} cm", tuple(int(255 * c) for c in theme.TIER_RGB[k][:3]))
           for k, (n, lo, hi) in enumerate(zip(theme.TIER_NAMES, (10, 30, 60, 120), (30, 60, 120, None)))]
    info = {"title": f"Bowling Green / Warren County flood model - {app.domain.label}", "storm": app.storm.name,
            "hour": hour_now(app), "legend": leg, "legend_title": "Building / water depth tiers", "metres_per_px": mpp,
            "north_deg": north, "source": "depth colours: flow grid; " + (lab or "no solver tiles for this storm")}
    if getattr(app.storm, "synthetic", False):
        info["stamp"] = "DEMO DATA - synthetic, not a simulation result"
    card = sharecard.compose(arr, info)
    p = sharecard.save(card, screenshot.target_dir(app) / f"sharecard_{app.storm.name}_h{hour_now(app):02d}_{time.strftime('%Y%m%d_%H%M%S')}.png")
    F(ui).last_msg = f"Share card saved: {p}"
    ui.toasts.push(F(ui).last_msg, theme.OK, 6.0)
    return p


def tab_share(ui):
    f = F(ui)
    imgui.text_wrapped("A PNG of the 3D view with title, storm, hour, tier legend, scale bar, north arrow, the 'model result' stamp "
                       "and the data credits, saved next to the F12 screenshots.")
    if imgui.button("Save share card (F9)"):
        f.share_request = True
    if f.last_msg:
        imgui.text_disabled(f.last_msg[:200])


# ================================================================================================ session
def session_dir():
    return _docs("BG_Flood_Sessions")


def save_session(ui, path=None):
    f, app = F(ui), ui.app
    d = sesmod.collect(app, ui)
    p = sesmod.save(path or session_dir() / f"view_{app.domain.key}_{app.storm.name}_{time.strftime('%Y%m%d_%H%M%S')}.bgview", d)
    f.last_msg = f"Session saved: {p}"
    ui.toasts.push(f.last_msg, theme.OK, 5.0)
    return p


def open_session(ui, path):
    f, app = F(ui), ui.app
    try:
        d = sesmod.load(path)
    except Exception as e:
        f.last_msg = f"Could not open {path}: {e}"
        ui.toasts.push(f.last_msg, theme.ERR, 6.0)
        return False
    if d.get("domain") and d["domain"] != app.domain.key:
        f.pending_session = d
        app.set_domain(d["domain"])
        f.last_msg = f"Switching to {d['domain']} to open the session ..."
        return True
    _apply_session(ui, d)
    return True


def _apply_session(ui, d):
    f = F(ui)
    missed = sesmod.apply(ui.app, ui, d)
    f.last_msg = "Session restored" + (f" (not restored: {', '.join(missed)})" if missed else "")
    ui.toasts.push(f.last_msg, theme.OK if not missed else theme.WARN, 5.0)


def recent_sessions(n=8):
    try:
        return sorted(session_dir().glob("*.bgview"), key=lambda p: p.stat().st_mtime, reverse=True)[:n]
    except OSError:
        return []


def tab_session(ui):
    f = F(ui)
    if imgui.button("Save session (Ctrl+S)"):
        save_session(ui)
    imgui.text_disabled(f"Folder: {session_dir()}")
    imgui.text("Recent sessions (Ctrl+O opens the newest):")
    for p in recent_sessions():
        if imgui.selectable(p.name, False)[0]:
            open_session(ui, p)
    imgui.text_disabled("Saves camera, view (city / county), storm, hour, layers, bookmarks, at-risk filters and feature settings.")
    if f.last_msg:
        imgui.text_disabled(f.last_msg[:200])


# ================================================================================================ accessibility
def apply_access(ui):
    f, st = F(ui), _st(ui)
    hc = bool(st.get("high_contrast", False))
    want = (hc, theme.MODE)
    if f.hc_applied != want:
        if hc:
            d = HC
            for k in ("BG0", "BG1", "BG2", "BG3", "BORDER", "ACCENT", "ACCENT2", "TEXT", "TEXT_DIM", "TEXT_MUTE", "OK", "WARN", "ERR"):
                setattr(theme, k, theme.hexrgb(d[k]))
            theme.LIGHT_ON = False
            try:
                from . import widgets, panels_risk
                widgets._DIM = None
                panels_risk._DIM4 = None
            except Exception:
                pass
        elif f.hc_applied is not None and f.hc_applied[0]:
            theme.set_mode(theme.MODE)
        theme.apply_colours(imgui)
        if hc:
            sty = imgui.get_style()
            sty.frame_border_size = 1.0
            sty.window_border_size = 2.0
        elif f.hc_applied is not None and f.hc_applied[0]:
            sty = imgui.get_style()
            sty.frame_border_size = 0.0
            sty.window_border_size = 1.0
        f.hc_applied = want
    fs = float(st.get("font_scale", 1.0)) if isinstance(st.get("font_scale", 1.0), (int, float)) else 1.0
    fs = min(2.0, max(0.75, fs))
    if f.font_applied != fs:
        try:
            imgui.get_style().font_scale_dpi = fs
        except Exception:
            pass
        f.font_applied = fs
    io = imgui.get_io()
    nav = int(imgui.ConfigFlags_.nav_enable_keyboard)
    if st.get("keyboard_nav", True):
        io.config_flags |= nav
    else:
        io.config_flags &= ~nav


def cycle_focus(ui, step=1):
    f = F(ui)
    names = [n for n in NAV_WINDOWS if n != "Flood tools" or f.open]
    f.nav_i = (f.nav_i + step) % len(names)
    imgui.set_window_focus(names[f.nav_i])
    ui.toasts.push(f"Keyboard focus: {names[f.nav_i].strip('#')}  (arrows / Space / Tab, Esc returns to the map)", theme.ACCENT2, 3.0)


def tab_access(ui):
    st = _st(ui)
    ch, v = imgui.checkbox("High-contrast theme", bool(st.get("high_contrast", False)))
    if ch:
        st["high_contrast"] = bool(v)
    imgui.set_next_item_width(180 * ui.s)
    ch, v = imgui.slider_float("Font size", float(st.get("font_scale", 1.0)), 0.75, 2.0, "%.2fx")
    if ch:
        st["font_scale"] = float(v)
    imgui.text_disabled("Text only; the interface scale (View menu) stays as it is.")
    ch, v = imgui.checkbox("Keyboard navigation of panels (F6 cycles: tools, sidebar, timeline, at-risk list)", bool(st.get("keyboard_nav", True)))
    if ch:
        st["keyboard_nav"] = bool(v)
    imgui.text_disabled("In a panel: arrows move, Space / Enter activate, Tab next field, Esc returns to the map.")
    imgui.text_disabled("New legends (routes, durations, heat ramp) use Okabe-Ito / tested colours: protan, deutan and tritan safe.")


# ================================================================================================ the window
def draw_window(ui):
    f = F(ui)
    if not f.open:
        return
    L, s = ui.L, ui.s
    # keep the window inside the 3D view rectangle (never over the sidebar, timeline, at-risk list or menu bar); the user can
    # move / resize it inside that rectangle; a layout change (resize, UI scale, right panel) pulls it back in
    x0, y0, x1, y1 = L.sx0 + 8 * s, L.sy0 + 8 * s, L.sx1 - 8 * s, L.sy1 - 8 * s
    vw, vh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    want_w = min(460.0 * s, vw)
    want_h = min(560.0 * s, vh - 52 * s) if vh > 160 * s else vh
    r = f.__dict__.get("win_rect")
    lay = (round(x0), round(y0), round(x1), round(y1))
    if r is None or f.__dict__.get("win_layout") != lay or r[0] < x0 - 1 or r[1] < y0 - 1 or r[0] + r[2] > x1 + 1 or r[1] + r[3] > y1 + 1:
        w = min(r[2], vw) if r is not None and f.__dict__.get("win_layout") == lay else want_w
        h = min(r[3], vh) if r is not None and f.__dict__.get("win_layout") == lay else want_h
        imgui.set_next_window_pos(V2(max(x0, x1 - w), min(max(y0, y0 + 44 * s), max(y0, y1 - h))), imgui.Cond_.always)
        imgui.set_next_window_size(V2(w, h), imgui.Cond_.always)
        f.win_layout = lay
    imgui.set_next_window_size_constraints(V2(min(240 * s, vw), min(160 * s, vh)), V2(vw, vh))
    vis, opened = imgui.begin("Flood tools", True, int(imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings))
    try:
        p, sz = imgui.get_window_pos(), imgui.get_window_size()
        f.win_rect = (p.x, p.y, sz.x, sz.y)
        if not opened:
            f.open = False
        if vis and imgui.begin_tab_bar("##ftabs"):
            for i, name in enumerate(TABS):
                flags = int(imgui.TabItemFlags_.set_selected) if f.tab_request == i else 0
                sel, _ = imgui.begin_tab_item(name, None, flags)
                if sel:
                    f.tab = i
                    [tab_route, tab_exposure, tab_duration, tab_compare, tab_share, tab_session, tab_access][i](ui)
                    imgui.end_tab_item()
            f.tab_request = None
            imgui.end_tab_bar()
    finally:
        imgui.end()


def arbiter(ui):
    """Every frame: exclusive click tools, stale-state reset, accessibility, keyboard focus cycling, queued work."""
    f, app = F(ui), ui.app
    if not f.help_done:
        keymap.install_help()
        f.help_done = True
        _env_setup(ui)
    apply_access(ui)
    # one click-capturing tool at a time: the analysis tools (Analysis menu) and the registry tools
    an = app.get_layer("analysis")
    at = getattr(an, "tool", "none") if an is not None else "none"
    if at != "none" and ui.tool is not None:
        if f.prev_an_tool == "none":                 # the analysis tool was just switched on: it wins
            ui.activate_tool(None)
        else:                                        # a registry tool was just switched on: leave the analysis tool
            an.tool = "none"
            at = "none"
    f.prev_an_tool = at
    # stale state after a city <-> county or storm switch
    dk = (app.domain.key, id(app.domain))
    if f.seen_domain is not None and dk != f.seen_domain:
        reset_for_domain(ui)
    f.seen_domain = dk
    sk = app.storm.name if app.storm is not None else None
    if f.seen_storm is not None and sk != f.seen_storm and f.capture is None:
        f.route_res = None                           # recomputed for the new storm by route_tick
        f.expo_ids = None
    f.seen_storm = sk
    if f.pending_session is not None and not app.loading and app.domain.key == f.pending_session.get("domain"):
        d, f.pending_session = f.pending_session, None
        _apply_session(ui, d)
    route_tick(ui)
    capture_tick(ui)
    if getattr(f, "share_request", False):
        f.share_request = False
        try:
            share_card(ui)
        except Exception as e:
            ui.toasts.push(f"Share card failed: {e}", theme.ERR, 6.0)
    if _st(ui).get("keyboard_nav", True) and imgui.is_key_pressed(imgui.Key.f6, False):
        cycle_focus(ui, -1 if imgui.get_io().key_shift else 1)
    if f.route_pts:
        draw_route_overlay(ui)


def _env_setup(ui):
    """Benchmarks / screenshots: BGV_FEATURES_ROUTE=lng,lat,lng,lat  BGV_FEATURES_HEAT=1  BGV_FEATURES_HC=1."""
    f = F(ui)
    r = os.environ.get("BGV_FEATURES_ROUTE", "")
    try:
        v = [float(x) for x in r.split(",")] if r else []
        if len(v) == 4:
            f.route_pts = [(v[0], v[1]), (v[2], v[3])]
            ensure_graph(ui)
            for _ in range(3000):                            # scripted mode: wait for the graph (cached: about 1 s)
                if not f.graph_job.busy:
                    break
                time.sleep(0.05)
            ensure_graph(ui)
            if f.graph is not None:
                app = ui.app
                f.route_res = routing.plan(f.graph, f.depth_cache, f.route_pts[0], f.route_pts[1], app.storm.name, hour_now(app),
                                           f.threshold_cm / 100.0, tile_label(app))
                f.route_job.key = (app.storm.name, hour_now(app), f.threshold_cm / 100.0, tuple(f.route_pts))
    except ValueError:
        pass
    if os.environ.get("BGV_FEATURES_HEAT") == "1":
        lay = ui.app.get_layer("hoursflooded")
        if lay is not None:
            from .. import config
            durmod.cached_hours_flooded(svc(ui.app), ui.app.storm.name, ui.app.terrain.frame, config.cache_dir() / "hoursflooded")
            lay.enabled = True
    if os.environ.get("BGV_FEATURES_HC") == "1":
        _st(ui)["high_contrast"] = True


def reset_for_domain(ui):
    f = F(ui)
    ui.activate_tool(None)
    f.graph, f.depth_cache, f.route_pts, f.route_res = None, None, [], None
    f.graph_job = durmod.Background()
    f.expo_ids, f.expo_job, f.expo_lu = None, durmod.Background(), []
    f.dur_target, f.dur_job = None, durmod.Background()
    f.capture = None
    clear_thumbs(ui)
    try:                                              # search cards hold building / road / facility ids of the old domain
        from . import panels_search
        c = getattr(ui, panels_search.CTL_ATTR, None)
        if c is not None:
            c.result, c.facility_sel, c.road_sel = None, -1, None
    except Exception:
        pass


def on_key(ui, ev) -> bool:
    if ev.get("event_type") != "key_down":
        return False
    k = ev.get("key", "")
    kl = k.lower() if len(k) == 1 else k
    mods = set(ev.get("modifiers") or ())
    ctrl = bool(mods & {"Control", "Meta"})
    if ctrl and not (mods & {"Alt", "Shift"}):
        if kl == "s":
            save_session(ui)
            return True
        if kl == "o":
            r = recent_sessions(1)
            if r:
                open_session(ui, r[0])
            else:
                ui.toasts.push("No saved sessions yet (Ctrl+S saves one)", theme.WARN, 4.0)
            return True
    if k == "F9" and not mods:
        F(ui).share_request = True
        return True
    return False


# ================================================================================================ registration
registry.register_tool(RouteTool())
registry.register_tool(DurationTool())
registry.register_hotkey("x", lambda ui: toggle_tab(ui, 1), description="Building exposure filter and export")
registry.register_hotkey("v", lambda ui: toggle_tab(ui, 3), description="Compare storms (thumbnails of every storm)")
registry.register_key_hook("feature keys", on_key)
for _k, _m, _d in (("s", ("Control",), "Save the session (.bgview)"), ("o", ("Control",), "Open the newest saved session"),
                   ("F9", (), "Save an annotated share card (PNG)")):
    keymap.declare(_k, _m, owner="features", description=_d)
keymap.declare("F6", (), owner="features", description="Keyboard focus to the next panel (Shift+F6: previous)")
registry.register_panel("features arbiter", arbiter, order=5)
registry.register_panel("features window", draw_window, order=110)
for _i, _t in enumerate(TABS):
    registry.register_menu_item("Tools", f"Flood tools: {_t}", lambda ui, i=_i: open_tab(ui, i),
                                shortcut={1: "X", 3: "V", 4: "F9"}.get(_i, ""), order=60 + _i)
registry.register_menu_item("File", "Save session ...", lambda ui: save_session(ui), shortcut="Ctrl+S", order=40)
registry.register_menu_item("File", "Open newest session", lambda ui: on_key(ui, {"event_type": "key_down", "key": "o",
                                                                                     "modifiers": ("Control",)}), shortcut="Ctrl+O", order=41)
registry.register_menu_item("File", "Save share card (PNG)", lambda ui: setattr(F(ui), "share_request", True), shortcut="F9", order=42)
registry.register_menu_item("View", "High-contrast theme", lambda ui: _st(ui).__setitem__("high_contrast", not _st(ui).get("high_contrast", False)),
                            checked=lambda ui: bool(_st(ui).get("high_contrast", False)), order=120)
