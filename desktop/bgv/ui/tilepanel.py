"""Interface pieces for the native solver tiles: per-building rows of the selection card, the statistics comparison
(tile-derived vs flow-grid-derived flooded buildings) and the start-up hook that prepares the footprint index.

Nothing here changes an existing number: the flow-grid values stay where they were (relabelled "grid" when tiles are shown beside
them); the tile values are extra lines with their own label.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from . import bridge, theme, units
from .widgets import bold, dim, text_c

V2 = imgui.ImVec2


def service(app):
    from ..data import depthapi
    return depthapi.service(app)


def tiles_active(ui) -> bool:
    """True when the tile read-outs should show for the displayed storm."""
    try:
        if not ui.st.use_tiles:
            return False
        svc = service(ui.app)
        return bool(svc.enabled and svc.tileset(ui.app.storm.name) is not None)
    except Exception:
        return False


def tick(ui):
    """Called once per frame: when the scene is ready and the storm has tiles, build the footprint index in the background."""
    app = ui.app
    svc = service(app)
    if not (ui.st.use_tiles and svc.enabled) or svc.index_state != "idle":
        return
    if bridge.scene(app) is None or svc.tileset(app.storm.name) is None:
        return
    dom, frame = app.domain, app.terrain.frame

    def path():
        return dom.buildings_geojson if dom.key == "city" else dom.scene_paths(frame)[0]
    svc.start_index(path, app.storm.name)


def text_of(smp) -> str:
    if smp is None:
        return "-"
    if smp.status == "nodata":
        return "no data (outside the modelled area)"
    if smp.status == "unavailable":
        return smp.note or "not available"
    if smp.status == "dry":
        return "no water (0 cm)"
    return f"{smp.cm:.0f} cm  ({smp.ft:.2f} ft)"


def _kv(k: str, v: str, color=None):
    dim(k)
    imgui.same_line(imgui.get_font_size() * 6.6)
    if color is None:
        imgui.text(v)
    else:
        text_c(color, v)


# ------------------------------------------------------------------------------------------------ building card
def building_rows(ui, bid: int) -> bool:
    """Solver depth of a building (footprint maximum of the tile, peak + displayed hour) for the selection card.
    False (nothing drawn) when the storm has no tiles or they are off."""
    if not tiles_active(ui):
        return False
    app = ui.app
    hour = int(round(app.t))
    svc = service(app)

    def get():
        return (app.building_depth(bid, hour="peak", source="tiles"), app.building_depth(bid, hour=hour, source="tiles"))
    pk, hr = ui.cached(("tbd", app.storm.name, int(bid), hour, svc.index_state, id(svc.index)), get)
    col = theme.TEXT if hr.status in ("depth",) else (theme.TEXT_DIM if hr.status != "dry" else theme.OK)
    _kv(f"Solver depth", text_of(hr), col)
    imgui.same_line()
    dim(f" hour {hr.hour}" if hr.hour is not None else "")
    if pk.status != "unavailable":
        _kv("Solver peak", text_of(pk), theme.TEXT)
    ref = hr if hr.status != "unavailable" else pk
    imgui.push_text_wrap_pos(0.0)
    dim(f"{ref.label}: highest cell under the footprint" + (f"  |  {ref.scope} run" if ref.scope == "county" and app.domain.key == "city" else ""),
        wrap=True)
    imgui.pop_text_wrap_pos()
    return True


# ------------------------------------------------------------------------------------------------ statistics comparison
def compare_numbers(fl: dict, cm: np.ndarray, in_grid: np.ndarray) -> dict:
    """Pure function (unit-tested): the grid-derived and tile-derived flooded-building counts over the buildings that have a flow-grid
    pixel AND a tile value, and how they overlap.  `fl` = the statistics snapshot (fl['buildings'] with .tier / .flooded)."""
    bs = fl["buildings"]
    use = np.asarray(in_grid, bool) & (cm >= 0)
    g_fl = np.asarray(bs.flooded, bool)[use]
    t_cm = cm[use]
    t_fl = t_cm >= 10
    g_tier = np.asarray(bs.tier)[use]
    tc = np.array([int(((t_cm >= lo) & (t_cm < hi)).sum()) for lo, hi in ((10, 30), (30, 60), (60, 120), (120, 10 ** 9))])
    gc = np.array([int((g_tier == k).sum()) for k in (1, 2, 3, 4)])
    return {
        "n": int(use.sum()), "nodata": int(np.asarray(in_grid, bool).sum() - use.sum()),
        "grid_flooded": int(g_fl.sum()), "tile_flooded": int(t_fl.sum()),
        "both": int((g_fl & t_fl).sum()), "grid_only": int((g_fl & ~t_fl).sum()), "tile_only": int((~g_fl & t_fl).sum()),
        "grid_tiers": gc, "tile_tiers": tc,
        "grid_max_m": float(np.asarray(bs.depth_m)[use].max()) if use.any() else 0.0,
        "tile_max_m": float(t_cm.max()) / 100.0 if use.any() else 0.0,
    }


def stats_compare(ui, fl: dict):
    """The 'Compare with solver tiles' block of the Statistics section."""
    st, app = ui.st, ui.app
    if not tiles_active(ui):
        return
    ch, v = imgui.checkbox("Compare flooded buildings: tiles vs flow grid", bool(st.tile_stats))
    if ch:
        st.tile_stats = bool(v)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Counts flooded buildings a second way: the highest solver-tile cell under each footprint (6.25 m cells)\n"
                          "instead of the flow-grid cells. The numbers above are not changed.")
    if not st.tile_stats:
        return
    svc = service(app)
    b = bridge.buildings(app)
    if b is None or fl.get("buildings") is None:
        return
    hour = int(fl.get("hour", round(app.t)))
    cm = svc.building_depths(app.storm.name, hour)
    if cm is None:
        if svc.index_state == "error":
            text_c(theme.ERR, f"Footprint index failed: {svc.index_msg[:80]}")
        else:
            dim("computing the tile footprints (first time: a few seconds)...")
        return
    if len(cm) != b.n:
        text_c(theme.ERR, "Tile footprint index does not match the buildings (different building file).")
        return
    r = ui.cached(("tcmp", fl.get("version"), app.storm.name, hour, id(cm)), lambda: compare_numbers(fl, cm, b.in_grid))
    ts = svc.tileset(app.storm.name)
    dim(f"hour {hour}, {r['n']:,} buildings with both a grid cell and a tile value", wrap=True)
    names = [n[:4] for n in units.TIER_NAMES]
    flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.borders_inner_h
    if imgui.begin_table("##tcmp", 6, flags):
        try:
            imgui.table_setup_column("", imgui.TableColumnFlags_.width_stretch, 1.6)
            for n in ("Total", *names):
                imgui.table_setup_column(n, imgui.TableColumnFlags_.width_stretch, 1.0)
            imgui.table_headers_row()
            for lab, fl_n, tiers, col in (("Flow grid", r["grid_flooded"], r["grid_tiers"], theme.WARN),
                                          ("Tiles", r["tile_flooded"], r["tile_tiers"], theme.ACCENT2)):
                imgui.table_next_row()
                imgui.table_next_column()
                text_c(col, lab)
                imgui.table_next_column()
                bold(f"{fl_n:,}", col)
                for k in range(4):
                    imgui.table_next_column()
                    imgui.text(f"{int(tiers[k]):,}")
        finally:
            imgui.end_table()
    dim(f"flooded in both {r['both']:,}   grid only {r['grid_only']:,}   tiles only {r['tile_only']:,}", wrap=True)
    dim(f"max depth: grid {r['grid_max_m']:.2f} m, tiles {r['tile_max_m']:.2f} m   |   no tile value: {r['nodata']:,}", wrap=True)
    imgui.push_text_wrap_pos(0.0)
    dim("Differences are expected: tile cells are 6.25 m, flow-grid cells 20-25 m, and each building takes the maximum over the cells "
        "its footprint touches." + (f" {ts.scope.capitalize()} solver run." if ts is not None and ts.scope == "county"
                                    and app.domain.key == "city" else ""), wrap=True)
    imgui.pop_text_wrap_pos()
