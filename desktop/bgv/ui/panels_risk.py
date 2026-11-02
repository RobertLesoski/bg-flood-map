"""Right-hand panels: selected-building card and the at-risk building list (virtual list, filters, search, CSV export)."""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from . import atrisk, bridge, theme, units
from .theme import col32
from .widgets import bold, chip_toggle, dim, font_text, sparkline_depth, text_c

V2 = imgui.ImVec2
V4 = imgui.ImVec4

CARD_H = 292.0


def _card_visible(ui) -> bool:
    sel = bridge.selection(ui.app)
    return bool(ui.st.show_selected and (sel.get("building") is not None or sel.get("road") is not None)
                and bridge.buildings(ui.app) is not None)


def draw_selected(ui):
    if not _card_visible(ui):
        return
    L, app = ui.L, ui.app
    b = bridge.buildings(app)
    sel = bridge.selection(app)
    if sel.get("building") is None:
        return _road_card(ui, int(sel["road"]))
    bid = int(sel["building"])
    if bid < 0 or bid >= b.n:
        return
    fl = bridge.flood(app)
    x = L.W - L.right_w - L.margin
    y = L.sy0 + L.margin
    compact = L.right_h < 760.0 * ui.s
    if not ui.begin_panel("##selected", x, y, L.right_w, 0.0, padding=(14.0 * ui.s, 10.0 * ui.s), auto_height=True):
        ui.end_panel()
        return
    try:
        bs = fl.get("buildings") if fl else None
        approx = bool(fl and fl.get("source") == "tiers")
        depth = float(bs.depth_m[bid]) if bs is not None else 0.0
        tier = int(bs.tier[bid]) if bs is not None else 0
        speed = float(bs.speed_ms[bid]) if bs is not None else 0.0
        addr = str(b.address[bid]).strip() or f"Building #{bid}"
        # title row
        imgui.push_text_wrap_pos(L.right_w - 70.0 * ui.s)
        bold(addr, theme.TEXT)
        imgui.pop_text_wrap_pos()
        imgui.same_line(L.right_w - 56.0 * ui.s)
        if imgui.small_button("Close"):
            bridge.clear_selection(app)
        # tier chip + depth
        p = imgui.get_cursor_screen_pos()
        dl = imgui.get_window_draw_list()
        if 1 <= tier <= 4:
            col = theme.tier_color(tier)
            lab = f"{units.TIER_NAMES[tier - 1]}  {units.tier_range_text(tier)}"
        else:
            col = theme.OK
            lab = "Not flooded at this hour"
        w = imgui.calc_text_size(lab).x + 18.0
        h2 = imgui.calc_text_size(lab).y + 6.0
        dl.add_rect_filled(p, V2(p.x + w, p.y + h2), col32(theme.with_alpha(col, 0.2)), h2 * 0.5)
        dl.add_rect(p, V2(p.x + w, p.y + h2), col32(col), h2 * 0.5, 1.0)
        dl.add_text(V2(p.x + 9, p.y + 3), col32(col), lab)
        imgui.dummy(V2(w, h2))
        imgui.same_line()
        dim(f"as of hour {int(fl['hour'])}" if fl else "")
        imgui.spacing()
        # key numbers
        _kv("Water depth", units.depth_text(depth, tier=tier, approx=approx))
        if speed > 0.01:
            _kv("Flow speed", units.speed_text(speed))
        if bs is not None and bool(bs.flooded[bid]):
            rk = ui.cached(("rank", fl.get("version"), app.storm.name, bid), lambda: (lambda w: int(w[0]) + 1 if w.size else 0)(np.flatnonzero(bs.order == bid)))
            if rk:
                _kv("Rank", f"#{rk:,} of {int(bs.flooded_count):,} flooded (deepest first)")
        hm = float(b.height_m[bid])
        am = float(b.area_m2[bid])
        if compact:                                   # short window: one info line, the at-risk list keeps its room
            imgui.push_text_wrap_pos(0.0)
            dim(f"{units.land_use_label(str(b.land_use[bid]))}  |  {units.fmt_money(b.sale_price[bid])}  |  "
                f"{hm:.0f} m tall  |  {am:,.0f} m²", wrap=True)
            imgui.pop_text_wrap_pos()
        else:
            _kv("Land use", units.land_use_label(str(b.land_use[bid])))
            _kv("Sale price", units.fmt_money(b.sale_price[bid]))
            _kv("Height", f"{hm:.1f} m  ({units.m_to_ft(hm):.0f} ft)")
            _kv("Footprint", f"{am:,.0f} m²  ({am * units.M2_TO_FT2:,.0f} ft²)")
        # depth over time
        tl = bridge.timeline(app) or {}
        series = ui.cached(("series", app.storm.name, bid, tl.get("done", 0)), lambda: bridge.depth_series(app, bid))
        imgui.spacing()
        dim("Depth by hour" + ("   (tiers: approximate)" if approx else ""))
        p = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        ph = (38.0 if compact else 60.0) * ui.s
        if series is not None:
            vals = np.array(series, np.float64)
            top = sparkline_depth(imgui.get_window_draw_list(), (p.x, p.y), (avail.x, ph), vals, app.t,
                                  col if tier else theme.ACCENT)
            fin = vals[np.isfinite(vals)]
            peak = float(fin.max()) if fin.size else 0.0
            first = next((i for i, v in enumerate(vals) if np.isfinite(v) and v >= 0.1), None)
            imgui.dummy(V2(avail.x, ph))
            if first is not None:
                dim(f"first flooded hour {first}  |  peak {units.depth_short(peak, tier=4 if (approx and peak >= 1.2) else 1, approx=approx)}"
                    + ("" if np.isfinite(vals).all() else "  (computing...)"))
            else:
                dim("never flooded in the computed hours" + ("" if np.isfinite(vals).all() else " (computing...)"))
        else:
            imgui.dummy(V2(avail.x, ph))
            dim("depth history is being computed...")
        if imgui.button("Fly to  (F)"):
            bridge.fly_to_building(app, bid)
        imgui.same_line()
        if imgui.button("Deselect  (Esc)"):
            bridge.clear_selection(app)
        ui.card_h = float(imgui.get_window_height())
    finally:
        ui.end_panel()


def _kv(k: str, v: str):
    dim(k)
    imgui.same_line(imgui.get_font_size() * 6.6)        # scales with the interface size
    imgui.text(v)


def _road_card(ui, rid: int):
    """Small card for a selected road (class, length, flooded share at the displayed hour)."""
    L, app = ui.L, ui.app
    d = bridge.describe_road(app, rid)
    if d is None:
        return
    x = L.W - L.right_w - L.margin
    y = L.sy0 + L.margin
    if not ui.begin_panel("##selected", x, y, L.right_w, 0.0, padding=(14.0 * ui.s, 10.0 * ui.s), auto_height=True):
        ui.end_panel()
        return
    try:
        bold(f"Road: {str(d.get('class', 'road')).replace('_', ' ')}", theme.TEXT)
        imgui.same_line(L.right_w - 56.0 * ui.s)
        if imgui.small_button("Close"):
            bridge.clear_selection(app)
        _kv("Length", f"{d['length_m']:,.0f} m  ({units.m_to_ft(d['length_m']):,.0f} ft)")
        pct = float(d.get("flooded_pct", 0.0))
        imgui.text_colored(V4(*(theme.TIER_RGB[3] if pct > 0 else theme.OK)),
                           f"{pct:.0f}% of this road is flooded" + (f" (hour {d['hour']})" if d.get("hour") is not None else ""))
        ui.card_h = float(imgui.get_window_height())
    finally:
        ui.end_panel()


# ------------------------------------------------------------------------------------------------ at-risk list
def draw_atrisk(ui):
    L, app, st = ui.L, ui.app, ui.st
    if not ui.atrisk_shown():
        return
    x = L.W - L.right_w - L.margin
    y0 = L.sy0 + L.margin
    top_used = 0.0
    if _card_visible(ui):
        top_used = min(ui.card_h or CARD_H * ui.s, L.right_h * 0.7) + L.margin
    y = y0 + top_used
    h = L.right_h - top_used
    if h < 120.0:
        return
    if not ui.begin_panel("##atrisk", x, y, L.right_w, h, padding=(12.0 * ui.s, 10.0 * ui.s)):
        ui.end_panel()
        return
    try:
        fl = bridge.flood(app)
        b = bridge.buildings(app)
        ids = ui.risk_list()
        bold("At-risk buildings", theme.ACCENT2)
        if fl is not None and fl.get("ready") and ids is not None:
            imgui.same_line()
            dim(f"  {len(ids):,} shown  |  hour {fl['hour']}")
        elif fl is not None and not fl.get("ready"):
            imgui.same_line()
            dim("  preparing...")
        if ids is None or b is None:
            imgui.spacing()
            dim("Building data is loading...", wrap=True)
            return
        _filters(ui)
        sel = bridge.selection(app).get("building")
        approx = bool(fl.get("source") == "tiers")
        bs = fl["buildings"]
        avail = imgui.get_content_region_avail()
        flags = imgui.TableFlags_.scroll_y | imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit
        if imgui.begin_table("##risk", 4, flags, V2(0, max(avail.y - 36.0 * ui.s, 60.0))):
            try:
                TC = imgui.TableColumnFlags_
                imgui.table_setup_column("Address", TC.width_stretch, 1.0)
                imgui.table_setup_column("Depth", TC.width_fixed, 70.0 * ui.s)
                imgui.table_setup_column("Use", TC.width_fixed, 34.0 * ui.s)
                imgui.table_setup_column("Price", TC.width_fixed, 58.0 * ui.s)
                imgui.table_setup_scroll_freeze(0, 1)
                imgui.table_headers_row()
                clip = imgui.ListClipper()
                clip.begin(len(ids))
                sel_pos = None
                if sel is not None and ui.scroll_to_selected:
                    w = np.flatnonzero(ids == sel)
                    sel_pos = int(w[0]) if w.size else None
                while clip.step():
                    for r in range(clip.display_start, clip.display_end):
                        _row(ui, r, int(ids[r]), bs, b, approx, sel)
                if sel_pos is not None:
                    row_h = imgui.get_text_line_height_with_spacing() + 3.0
                    imgui.set_scroll_y(max(0.0, sel_pos * row_h - imgui.get_window_height() * 0.35))
                    ui.scroll_to_selected = False
                elif ui.scroll_to_selected:
                    ui.scroll_to_selected = False
                clip.end()
            finally:
                imgui.end_table()
        if imgui.button("Export CSV"):
            quick_export(ui)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Saves the list as shown (filters applied) to Documents\\BG_Flood_Exports")
        imgui.same_line()
        dim("click a row to select + fly there")
    finally:
        ui.end_panel()


def _filters(ui):
    st, f = ui.st, ui.risk_filter
    tiers = list(f.tiers)
    changed = False
    for k in range(4):
        if k:
            imgui.same_line()
        if chip_toggle(f"##tier{k}", units.TIER_NAMES[k], bool(tiers[k]), theme.TIER_RGB[k],
                       tip=units.tier_range_text(k + 1)):
            tiers[k] = not tiers[k]
            changed = True
    imgui.set_next_item_width(-1)
    ch, txt = imgui.input_text_with_hint("##risk_search", "Search address or land use", f.search)
    if ch:
        f.search = txt[:80]
        changed = True
    if changed:
        f.tiers = tuple(tiers)
        ui.sync_risk_filter()


_DIM4 = None
_SEL_FLAGS = int(imgui.SelectableFlags_.span_all_columns | imgui.SelectableFlags_.allow_overlap)


def _row(ui, r: int, bid: int, bs, b, approx: bool, sel):
    global _DIM4
    if _DIM4 is None:
        _DIM4 = V4(*theme.TEXT_DIM)
    depth = float(bs.depth_m[bid])
    tier = int(bs.tier[bid])
    addr = str(b.address[bid]).strip() or f"Building #{bid}"
    imgui.table_next_row()
    imgui.table_next_column()
    clicked, _ = imgui.selectable(f"##r{bid}", sel is not None and bid == sel, _SEL_FLAGS, V2(0, 0))
    if clicked:
        bridge.select_building(ui.app, bid, fly=True)
    hov = imgui.is_item_hovered()
    imgui.same_line(0.0, 0.0)
    p = imgui.get_cursor_screen_pos()
    col = theme.tier_color(tier)
    imgui.get_window_draw_list().add_rect_filled(V2(p.x, p.y + 2), V2(p.x + 4, p.y + imgui.get_text_line_height() - 1),
                                                 col32(col), 2.0)
    imgui.set_cursor_screen_pos(V2(p.x + 10, p.y))
    imgui.text(addr)
    if hov:
        imgui.set_tooltip(f"{addr}\n{units.depth_text(depth, tier=tier, approx=approx)}  -  "
                          f"{units.TIER_NAMES[tier - 1] if 1 <= tier <= 4 else ''}\n{units.land_use_label(str(b.land_use[bid]))}")
    imgui.table_next_column()
    imgui.text_colored(V4(*col), units.depth_short(depth, tier=tier, approx=approx))
    imgui.table_next_column()
    imgui.text_colored(_DIM4, str(b.land_use[bid])[:3])
    imgui.table_next_column()
    imgui.text_colored(_DIM4, units.fmt_money(b.sale_price[bid]))


# ------------------------------------------------------------------------------------------------ export
def _export(ui, path) -> bool:
    app = ui.app
    fl = bridge.flood(app)
    b = bridge.buildings(app)
    ids = ui.risk_list()
    if fl is None or b is None or ids is None or fl.get("buildings") is None:
        ui.toasts.push("Nothing to export yet - the statistics are still loading", theme.WARN)
        return False
    try:
        n = atrisk.export_csv(path, ids, b, fl["buildings"], storm=str(fl["storm"]), hour=int(fl["hour"]),
                              source=str(fl.get("source", "")), frame=app.terrain.frame)
    except Exception as e:
        ui.toasts.push(f"Export failed: {e}", theme.ERR, 6.0)
        return False
    ui.toasts.push(f"Exported {n:,} buildings to {path}", theme.OK, 6.0)
    return True


def quick_export(ui):
    fl = bridge.flood(ui.app) or {}
    name = atrisk.default_export_name(str(fl.get("storm", "storm")), int(fl.get("hour", 0)))
    return _export(ui, atrisk.default_export_dir() / name)


def export_dialog(ui):
    """File > Export...: a native Save dialog (tkinter, main thread, blocks the UI while open); falls back to the
    default folder when tkinter is unavailable."""
    fl = bridge.flood(ui.app) or {}
    name = atrisk.default_export_name(str(fl.get("storm", "storm")), int(fl.get("hour", 0)))
    folder = atrisk.default_export_dir()
    path = None
    try:
        import tkinter
        from tkinter import filedialog
        folder.mkdir(parents=True, exist_ok=True)
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.asksaveasfilename(parent=root, title="Export at-risk buildings", initialdir=str(folder),
                                                initialfile=name, defaultextension=".csv",
                                                filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        finally:
            root.destroy()
        if not path:
            return False
    except Exception:
        path = str(folder / name)
    return _export(ui, path)
