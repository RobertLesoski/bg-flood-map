"""Help and About window: searchable help topics, keyboard-shortcut cheat sheet, per-storm data sources
(renders STORM_DATA_SOURCES.md from the repository root when present, else a placeholder), and About with the data-file status
table and the log-file path.  It replaces the two small dialogs (`ui.show_help` opens the Help tab, `ui.show_about` the About tab);
the old flags keep working.  Pure drawing - content lives in `bgv.search.helpdoc`.
"""
from __future__ import annotations

import os
import sys
import time

from imgui_bundle import imgui

from ..search import helpdoc
from . import theme
from .theme import col32
from .widgets import bold, dim, font_text, text_c

V2 = imgui.ImVec2
V4 = imgui.ImVec4
TABS = ("Help", "Shortcuts", "Data sources", "About")
TITLE = "DSOC Flood Map - Bowling Green and Warren County, KY"

CREDITS = [
    "Flood depths: custom 2D GPU local-inertial solver (Bates et al. 2010) run on TACC Vista (NVIDIA H100)",
    "Rainfall: NOAA Atlas 14, Warren County KY (design storms) and historical events",
    "Terrain: USGS / KyFromAbove LiDAR DEM",
    "Buildings, roads, facilities: OpenStreetMap contributors; parcel data: Warren County PVA",
    "Flow velocity: derived from water-surface slope with Manning's equation (not solver output)",
]


class HelpState:
    def __init__(self):
        self.tab = "Help"
        self.want_tab: str | None = None
        self.query = ""
        self.blocks = None
        self.sources_msg = ""
        self.sources_path = ""
        self.sources_t = 0.0
        self.topics = helpdoc.topics()
        self.files = None
        self.files_key = None
        self.files_t = 0.0
        self.opened = {"help": False, "about": False}


def _state(ui) -> HelpState:
    h = getattr(ui, "_help_state", None)
    if h is None:
        h = HelpState()
        ui._help_state = h
    return h


def draw(ui, ctl=None):
    h = _state(ui)
    was = h.opened["help"] or h.opened["about"]
    now_open = bool(ui.show_help or ui.show_about)
    if now_open and not was:                       # just opened: choose the tab, raise above the overlay panels
        h.want_tab = "About" if (ui.show_about and not ui.show_help) else (h.tab if h.tab else "Help")
        imgui.set_next_window_focus()
    if ui.show_help and not h.opened["help"] and h.opened["about"]:
        h.want_tab = "Help"
    if ui.show_about and not h.opened["about"] and h.opened["help"]:
        h.want_tab = "About"
    h.opened["help"], h.opened["about"] = bool(ui.show_help), bool(ui.show_about)
    if not now_open:
        return
    io = imgui.get_io()
    s = ui.s
    imgui.set_next_window_pos(V2(io.display_size.x * 0.5, io.display_size.y * 0.5), int(imgui.Cond_.appearing), V2(0.5, 0.5))
    imgui.set_next_window_size(V2(760.0 * s, 620.0 * s), int(imgui.Cond_.appearing))
    imgui.set_next_window_bg_alpha(1.0)
    vis, opened = imgui.begin("Help and About", True, int(imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings))
    try:
        if vis:
            _body(ui, h, ctl)
    finally:
        imgui.end()
    if opened is False:
        ui.show_help = ui.show_about = False


def _body(ui, h: HelpState, ctl):
    s = ui.s
    imgui.set_next_item_width(-1)
    ch, q = imgui.input_text_with_hint("##hq", "Search help and shortcuts", h.query)
    if ch:
        h.query = q[:80]
    if h.query.strip():
        _search_results(ui, h)
        return
    if imgui.begin_tab_bar("##helptabs"):
        try:
            for name in TABS:
                flags = int(imgui.TabItemFlags_.set_selected) if h.want_tab == name else 0
                vis, _ = imgui.begin_tab_item(name, None, flags)
                if vis:
                    h.tab = name
                    imgui.begin_child("##tab_" + name, V2(0, 0))
                    try:
                        {"Help": _tab_help, "Shortcuts": _tab_shortcuts, "Data sources": _tab_sources, "About": _tab_about}[name](ui, h, ctl)
                    finally:
                        imgui.end_child()
                    imgui.end_tab_item()
            h.want_tab = None
        finally:
            imgui.end_tab_bar()


# ------------------------------------------------------------------------------------------------ rendering helpers
def render_blocks(ui, blocks, key: str = "b"):
    s = ui.s
    for n, b in enumerate(blocks):
        k = b.kind
        if k in ("h1", "h2"):
            imgui.spacing()
            font_text(theme.Fonts.bold, 0.0, b.text, theme.ACCENT2)
        elif k == "h3":
            bold(b.text, theme.TEXT)
        elif k == "p":
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(V4(*theme.TEXT), b.text)
            imgui.pop_text_wrap_pos()
        elif k == "bullet":
            imgui.bullet()
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(V4(*theme.TEXT), b.text)
            imgui.pop_text_wrap_pos()
        elif k == "code":
            imgui.push_style_color(imgui.Col_.text, V4(*theme.TEXT_DIM))
            imgui.text_unformatted(b.text)
            imgui.pop_style_color()
        elif k == "hr":
            imgui.separator()
        elif k == "table" and b.rows:
            ncol = max(len(r) for r in b.rows)
            if imgui.begin_table(f"##{key}t{n}", ncol, int(imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.sizing_stretch_prop)):
                try:
                    for ri, r in enumerate(b.rows):
                        imgui.table_next_row()
                        for ci in range(ncol):
                            imgui.table_next_column()
                            imgui.push_text_wrap_pos(0.0)
                            txt = r[ci] if ci < len(r) else ""
                            if ri == 0:
                                imgui.text_colored(V4(*theme.ACCENT2), txt)
                            else:
                                imgui.text_colored(V4(*theme.TEXT), txt)
                            imgui.pop_text_wrap_pos()
                finally:
                    imgui.end_table()


def _search_results(ui, h):
    shown = False
    imgui.begin_child("##hres", V2(0, 0))
    try:
        for t, blocks in helpdoc.search(h.topics, h.query):
            shown = True
            bold(t.title, theme.ACCENT2)
            render_blocks(ui, [b for b in blocks if b.kind != "h2"], "s" + t.key)
            imgui.spacing()
        sc = helpdoc.search_shortcuts(h.query)
        if sc:
            shown = True
            bold("Shortcuts", theme.ACCENT2)
            for g, k, d in sc:
                imgui.text_colored(V4(*theme.TEXT), k)
                imgui.same_line(210.0 * ui.s)
                imgui.text_colored(V4(*theme.TEXT_DIM), d)
        blocks, path, msg = _sources(h)
        if blocks:
            words = h.query.lower().split()
            hit = [b for b in blocks if all(w in (b.text + " ".join(" ".join(r) for r in b.rows)).lower() for w in words)]
            if hit:
                shown = True
                bold("Storm data sources", theme.ACCENT2)
                render_blocks(ui, hit, "ss")
        if not shown:
            dim("Nothing found. Try fewer words, e.g. 'facilities', 'report', 'tiles', 'F11'.")
    finally:
        imgui.end_child()


# ------------------------------------------------------------------------------------------------ tabs
def _tab_help(ui, h, ctl):
    for t in h.topics:
        render_blocks(ui, t.blocks, t.key)
        imgui.spacing()
    if imgui.button("Show the first-run tips again"):
        try:
            if ctl is not None:
                ctl.onboarded = False
                ctl.hint_req = True
                ctl.persist()
            ui.show_help = ui.show_about = False
        except Exception:
            pass


def _tab_shortcuts(ui, h, ctl):
    s = ui.s
    for g, items in helpdoc.SHORTCUTS:
        imgui.spacing()
        bold(g, theme.ACCENT2)
        if imgui.begin_table(f"##sc{g}", 2, int(imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit)):
            try:
                imgui.table_setup_column("k", imgui.TableColumnFlags_.width_fixed, 210.0 * s)
                imgui.table_setup_column("d", imgui.TableColumnFlags_.width_stretch, 1.0)
                for k, d in items:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text_colored(V4(*theme.TEXT), k)
                    imgui.table_next_column()
                    imgui.push_text_wrap_pos(0.0)
                    imgui.text_colored(V4(*theme.TEXT_DIM), d)
                    imgui.pop_text_wrap_pos()
            finally:
                imgui.end_table()


def _sources(h):
    now = time.time()
    if h.blocks is None and not h.sources_msg or now - h.sources_t > 5.0:          # re-check the file every 5 s (it is written later)
        h.sources_t = now
        blocks, path, msg = helpdoc.storm_sources()
        h.blocks, h.sources_path, h.sources_msg = blocks, str(path), msg
    return h.blocks, h.sources_path, h.sources_msg


def _storm_descriptions(ui):
    """Full text of every storm's description + the honesty line (layers/storm_blurbs.json)."""
    from ..data import storm_blurbs
    app = ui.app
    names = list(getattr(app, "storms_available", []) or [])
    if not names or not any(storm_blurbs.blurb(app.root, n) for n in names):
        return
    bold("Storm descriptions", theme.ACCENT2)
    imgui.push_text_wrap_pos(0.0)
    for n in names:
        bl = storm_blurbs.blurb(app.root, n)
        if not bl:
            continue
        imgui.text_colored(V4(*theme.TEXT), f"{app.storm_label(n)}: {bl['long']}")
        dim(bl["honesty"], wrap=True)
        imgui.spacing()
    imgui.pop_text_wrap_pos()
    imgui.separator()


def _tab_sources(ui, h, ctl):
    try:
        _storm_descriptions(ui)
    except Exception as e:                                           # noqa: BLE001
        dim(f"storm descriptions unavailable: {e}", wrap=True)
    blocks, path, msg = _sources(h)
    if blocks is None:
        bold("Per-storm data sources", theme.ACCENT2)
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(V4(*theme.TEXT), msg)
        imgui.pop_text_wrap_pos()
        dim(f"Looking for: {path}", wrap=True)
        try:
            app = ui.app
            st = app.storm
            imgui.spacing()
            bold(f"Storm on screen: {getattr(st, 'label', st.name)}", theme.TEXT)
            note = str(st.meta.get("source_note", "(no source note in meta.json)"))
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(V4(*theme.TEXT), note)
            imgui.pop_text_wrap_pos()
        except Exception:
            pass
        return
    dim(f"from {path}", wrap=True)
    render_blocks(ui, blocks, "src")


def _log_path_text():
    try:
        from .. import applog
        p = applog.log_path()
        return str(p) if p else "(no log file could be created)"
    except Exception:
        return "(unknown)"


def _tab_about(ui, h, ctl):
    s = ui.s
    app = ui.app
    font_text(theme.Fonts.bold, 22.0, TITLE, theme.ACCENT2)
    dim("Desktop GPU viewer (wgpu / Vulkan)")
    imgui.spacing()
    imgui.push_text_wrap_pos(0.0)
    imgui.text_wrapped("Interactive 3D flood visualisation for Bowling Green and Warren County, Kentucky: simulated water depth, derived "
                       "flow velocity, buildings and roads coloured by flooding, address search, critical facilities, roads at risk "
                       "and one-click risk reports. Switch between the city and the county in the Domain menu.")
    imgui.pop_text_wrap_pos()
    imgui.spacing()
    bold("Data and credits", theme.ACCENT2)
    for c_ in CREDITS:
        imgui.bullet()
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(V4(*theme.TEXT), c_)
        imgui.pop_text_wrap_pos()
    imgui.spacing()
    bold("Honesty notes", theme.WARN)
    dim("Model results, not validated against observed floods. Velocity is derived from the water-surface slope (Manning), not solver "
        "output. Depth-tier previews are approximate: the severe tier is only known as >= 1.2 m. County depth tiles are marked "
        "'verification pending'. Indicative only; not an official flood map.", wrap=True)
    imgui.spacing()
    bold("Where things are", theme.ACCENT2)
    log = _log_path_text()
    _kv("Log file", log)
    imgui.same_line()
    if imgui.small_button("Copy##log"):
        imgui.set_clipboard_text(log)
    imgui.same_line()
    if imgui.small_button("Open folder##log"):
        try:
            os.startfile(os.path.dirname(log))
        except Exception as e:
            ui.toasts.push(f"Cannot open the log folder: {e}", theme.ERR)
    try:
        from .. import config
        _kv("Data folder", str(app.root))
        _kv("Cache folder", str(config.cache_dir()))
        _kv("Settings file", str(config.settings_path()))
    except Exception:
        pass
    from . import screenshot
    from ..search import report
    _kv("Screenshots", str(screenshot.target_dir(app)))
    _kv("Risk reports", str((app.settings or {}).get("report_dir") or report.default_dir()))
    _kv("Python", sys.version.split()[0])
    imgui.spacing()
    bold("Data files for this view", theme.ACCENT2)
    key = (app.domain.key, str(app.root))
    if h.files is None or h.files_key != key or time.time() - h.files_t > 3.0:
        h.files = helpdoc.data_files(app.root, app.domain.key, app.settings)
        h.files_key, h.files_t = key, time.time()
    for label, path, found, note in h.files:
        text_c(theme.OK if found else theme.WARN, "found  " if found else "MISSING")
        imgui.same_line()
        imgui.text_colored(V4(*theme.TEXT), label)
        imgui.same_line()
        dim(path, wrap=False)
        if note:
            dim("          " + note)
    imgui.spacing()
    bold("Robert Lesoski", theme.TEXT)
    dim("Undergraduate researcher, Disaster Science Operations Center (DSOC) | AI Research Lab (AIR), Western Kentucky University",
        wrap=True)
    dim("Principal investigator: Dr. Manmeet Singh", wrap=True)
    imgui.spacing()
    if imgui.button("Close"):
        ui.show_help = ui.show_about = False


def _kv(k, v):
    imgui.text_colored(V4(*theme.TEXT_DIM), f"{k}:")
    imgui.same_line()
    imgui.text_colored(V4(*theme.TEXT), v)
