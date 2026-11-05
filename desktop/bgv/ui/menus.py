"""Top bar: menus on the left, storm label / hour / source badge / FPS on the right."""
from __future__ import annotations

import os

import numpy as np
from imgui_bundle import imgui

from . import bridge, theme, units
from .cameranav import PRESETS
from .state import PRESETS as QUALITY_PRESETS
from .theme import col32
from .widgets import pill, pill_size

V2 = imgui.ImVec2


def source_info(storm):
    """(label, is_full_precision)"""
    if getattr(storm, "synthetic", False):
        return "DEMO DATA - synthetic, not a simulation result", False
    if storm.source == "tiers":
        return "Preview from depth tiers", False
    return "Full-precision (solver depth rasters)", True


def domain_tooltip(text: str, width_em: float = 30.0):
    """A wrapped tooltip (the reasons a domain is unavailable are long sentences)."""
    imgui.begin_tooltip()
    imgui.push_text_wrap_pos(imgui.get_font_size() * width_em)
    imgui.text_unformatted(text)
    imgui.pop_text_wrap_pos()
    imgui.end_tooltip()


def domain_items(app):
    """[(info, current, enabled)] for the Domain menu / combo; the explanation of an unavailable entry is `info.reason`."""
    out = []
    for info in app.domain_infos():
        cur = info.key == app.domain.key
        out.append((info, cur, bool(info.available)))
    return out


def _toggle(label: str, shortcut: str, value: bool):
    ch, v = imgui.menu_item(label, shortcut, bool(value))
    return v if ch else value


def draw(ui):
    app, st = ui.app, ui.st
    imgui.push_style_var(imgui.StyleVar_.frame_padding, V2(10.0 * ui.s, 8.0 * ui.s))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, V2(10.0 * ui.s, 6.0 * ui.s))
    try:
        if not imgui.begin_main_menu_bar():
            return
        try:
            ui.top_h = float(imgui.get_window_height())
            bar_h = ui.top_h
            if imgui.begin_menu("File"):
                if imgui.menu_item("Export at-risk list (CSV)", "", False)[0]:
                    from . import panels_risk
                    panels_risk.export_dialog(ui)
                if imgui.menu_item("Save screenshot", "F12", False)[0]:
                    ui.shot_request = "ui"
                if imgui.menu_item("Save screenshot (no interface)", "Shift+F12", False)[0]:
                    ui.shot_request = "scene"
                if imgui.menu_item("Open screenshot folder", "", False)[0]:
                    _open_folder(ui, "shots")
                if imgui.menu_item("Open exports folder", "", False)[0]:
                    _open_folder(ui, "exports")
                imgui.separator()
                if imgui.menu_item("Exit", "", False)[0]:
                    try:
                        app.canvas.close()
                    except Exception:
                        pass
                imgui.end_menu()
            if imgui.begin_menu("View"):
                st.show_sidebar = _toggle("Sidebar", "", st.show_sidebar)
                st.show_timeline = _toggle("Timeline", "", st.show_timeline)
                ui.set_atrisk(_toggle("At-risk list", "", ui.atrisk_shown()))
                st.show_selected = _toggle("Selected building card", "", st.show_selected)
                st.show_legends = _toggle("Legends", "", st.show_legends)
                st.show_compass = _toggle("Compass", "", st.show_compass)
                st.show_scalebar = _toggle("Scale bar", "", st.show_scalebar)
                st.show_probe = _toggle("Probe tooltip", "", st.show_probe)
                st.show_perf = _toggle("Performance overlay", "F3", st.show_perf)
                imgui.separator()
                st.imperial = _toggle("Imperial units on scale bar / legend", "", st.imperial)
                if imgui.begin_menu("Interface size"):
                    for pct in (80, 90, 100, 110, 125, 150, 175):
                        if imgui.menu_item(f"{pct}%", "", abs(st.ui_scale * 100 - pct) < 1)[0]:
                            st.ui_scale = pct / 100.0
                    imgui.end_menu()
                imgui.separator()
                if imgui.menu_item("Hide interface", "H", False)[0]:
                    st.hidden = True
                if imgui.menu_item("Fullscreen", "F11", st.fullscreen)[0]:
                    ui.toggle_fullscreen()
                imgui.end_menu()
            if imgui.begin_menu("Domain"):
                for info, cur, enabled in domain_items(app):
                    clicked, _ = imgui.menu_item(info.label, "", cur, enabled)
                    if info.reason and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                        domain_tooltip(info.reason)
                    if clicked and not cur:
                        app.set_domain(info.key)
                imgui.separator()
                ask = bool(app.settings.get("domain_chooser", True))
                ch, ask2 = imgui.menu_item("Ask which view at start-up", "", ask)
                if ch:
                    app.settings["domain_chooser"] = bool(ask2)
                imgui.text_disabled("Terrain, buildings, roads and statistics are rebuilt")
                imgui.end_menu()
            if imgui.begin_menu("Camera"):
                for name in app.domain.presets:
                    if imgui.menu_item(name, "Home" if name == "Overview" else "", False)[0]:
                        ui.fly_preset(name)
                if imgui.menu_item("Reset view", "", False)[0]:
                    ui.nav.cancel()
                    app.camera.reset()
                if imgui.menu_item("Look north", "", False)[0]:
                    ui.look_north()
                sel = bridge.selection(app).get("building")
                if imgui.menu_item("Fly to selected building", "F", False, sel is not None)[0]:
                    bridge.fly_to_building(app, int(sel))
                imgui.end_menu()
            from . import panels_analysis          # ANALYSIS hook: the Analysis menu
            panels_analysis.menu(ui)
            if imgui.begin_menu("Help"):
                if imgui.menu_item("Controls and shortcuts", "F1", False)[0]:
                    ui.show_help = True
                if imgui.menu_item("About DSOC Flood Map", "", False)[0]:
                    ui.show_about = True
                if imgui.menu_item("Data source details", "", False)[0]:
                    ui.show_source = True
                imgui.end_menu()
            x_menu_end = imgui.get_cursor_screen_pos().x
            _status(ui, bar_h, x_menu_end)
        finally:
            imgui.end_main_menu_bar()
    finally:
        imgui.pop_style_var(2)


def _status(ui, bar_h: float, x0: float):
    """Right part of the bar, drawn with the window draw list (no widgets, so it never steals clicks)."""
    app = ui.app
    dl = imgui.get_window_draw_list()
    pos = imgui.get_window_pos()
    W = imgui.get_io().display_size.x
    st = app.storm
    ts_h = imgui.calc_text_size("Ag").y
    cy = pos.y + bar_h * 0.5
    ty = cy - ts_h * 0.5
    label = getattr(st, "label", st.name) or st.name
    label = f"{app.domain.label}   |   {label}"
    hr = units.snapshot_hour(app.t)
    when = f"Hour {hr} of 24   ·   t = {units.hour_label(app.t)}"
    # right-aligned: fps pill, source badge
    fps_txt = f"{app.fps:4.0f} fps"
    fm = np.array(app.frame_ms, np.float64) if len(app.frame_ms) else np.array([16.7])
    good = app.fps >= 55 or app.fps == 0
    fw, fh = pill_size(fps_txt, (8.0, 2.0))
    x = W - fw - 12.0
    pill(dl, x, cy - fh * 0.5, fps_txt, theme.OK if good else theme.WARN,
         theme.with_alpha(theme.OK if good else theme.WARN, 0.14), (8.0, 2.0))
    src_label, full = source_info(st)
    demo = bool(getattr(st, "synthetic", False))
    badge = "DEMO DATA" if demo else ("Full precision" if full else "Preview from depth tiers")
    bw, bh = pill_size(badge, (9.0, 2.0))
    x -= bw + 10.0
    col = theme.OK if full else theme.WARN
    pill(dl, x, cy - bh * 0.5, badge, col, theme.with_alpha(col, 0.15), (9.0, 2.0), border=theme.with_alpha(col, 0.6))
    # hit area for the badge -> source details
    imgui.set_cursor_screen_pos(V2(x, pos.y))
    if imgui.invisible_button("##badge", V2(bw, bar_h)):
        ui.show_source = not ui.show_source
    if imgui.is_item_hovered():
        imgui.set_tooltip(src_label + "\nClick for details")
    # centre-left: storm label and hour
    wl = imgui.calc_text_size(label).x
    ww = imgui.calc_text_size(when).x
    avail = x - 24.0 - x0
    cx = x0 + 18.0
    if avail > wl + ww + 40:
        dl.add_text(V2(cx, ty), col32(theme.ACCENT2), label)
        dl.add_line(V2(cx + wl + 14.0, cy - 8.0), V2(cx + wl + 14.0, cy + 8.0), col32(theme.BORDER), 1.0)
        dl.add_text(V2(cx + wl + 28.0, ty), col32(theme.TEXT), when)
    elif avail > ww:
        dl.add_text(V2(cx, ty), col32(theme.TEXT), when)


def _open_folder(ui, which: str):
    from . import screenshot
    from .atrisk import default_export_dir
    folder = screenshot.default_dir() if which == "shots" else default_export_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))                      # Windows: opens Explorer on the folder
    except Exception as e:
        ui.toasts.push(f"Cannot open {folder}: {e}", theme.ERR)
