"""Floating dialogs: About, Help (all shortcuts), Source details (meta source_note)."""
from __future__ import annotations

from imgui_bundle import imgui

from . import bridge, theme
from .menus import source_info
from .widgets import bold, dim, font_text, text_c

V2 = imgui.ImVec2
V4 = imgui.ImVec4

TITLE = "DSOC Flood Map - Bowling Green, KY"

SHORTCUTS = [
    ("Mouse", None),
    ("Left drag", "Pan the map"),
    ("Right / middle drag", "Orbit (rotate and tilt)"),
    ("Mouse wheel", "Zoom toward the cursor"),
    ("Click a building", "Select it (card + highlight)"),
    ("Click the compass", "Face north"),
    ("Click / drag in the timeline chart", "Seek in time"),
    ("Keyboard", None),
    ("W A S D  /  arrows", "Move the camera (Shift = faster)"),
    ("Q  E", "Rotate left / right"),
    ("R  F", "Tilt (F flies to the selection when one exists)"),
    ("+  -", "Zoom in / out"),
    ("Space", "Play / pause"),
    ("[  ]", "Step one hour back / forward"),
    ("F", "Fly to the selected building"),
    ("Home", "Overview (reset view)"),
    ("Ctrl+F  or  /", "Focus the address search from anywhere"),
    ("In a text box", "Every key goes to the box (no hotkeys fire); Esc leaves the box first"),
    ("Esc", "Deselect / close dialogs"),
    ("H", "Hide / show the interface"),
    ("F1", "This help"),
    ("F3", "Performance overlay"),
    ("F11", "Fullscreen"),
    ("F12", "Screenshot to Documents\\BG_Flood_Screenshots  (Shift+F12: without interface)"),
]

CREDITS = [
    "Flood depths: custom 2D GPU local-inertial solver (Bates et al. 2010) run on TACC Vista (NVIDIA H100), 2 m grid",
    "Rainfall: NOAA Atlas 14, Warren County KY (5, 10, 25, 50, 100-year 24-hour storms)",
    "Terrain: USGS / KyFromAbove Phase 2 LiDAR DEM",
    "Buildings and roads: OpenStreetMap contributors; parcel data: Warren County PVA",
    "Flood zones: FEMA NFHL (web viewer)",
    "Flow velocity: derived from water-surface slope with Manning's equation (not solver output)",
]


def _track_opened(ui):
    """Keys of the dialogs that were opened this frame: they are raised above the overlay panels (which never take focus)."""
    prev = getattr(ui, "_dlg_open", {})
    now = {"help": bool(ui.show_help), "about": bool(ui.show_about), "source": bool(ui.show_source)}
    ui._dlg_open = now
    ui._dlg_raise = {k for k, v in now.items() if v and not prev.get(k, False)}


def _raise(ui, key):
    if key in getattr(ui, "_dlg_raise", ()):
        imgui.set_next_window_focus()


def draw(ui):
    _track_opened(ui)
    try:                                              # SEARCH hook: the searchable Help / About window replaces the two dialogs
        from . import help as help_window, panels_search
        help_window.draw(ui, panels_search.controller(ui))
    except Exception as e:
        ui.app.report_error("ui.help", e)
        _help(ui)
        _about(ui)
    _source(ui)


def _centered(ui, name, w, h):
    io = imgui.get_io()
    imgui.set_next_window_pos(V2(io.display_size.x * 0.5, io.display_size.y * 0.5), imgui.Cond_.appearing, V2(0.5, 0.5))
    imgui.set_next_window_size(V2(w * ui.s, h * ui.s), imgui.Cond_.appearing)


def _help(ui):
    if not ui.show_help:
        return
    _centered(ui, "help", 560, 560)
    _raise(ui, "help")
    imgui.set_next_window_bg_alpha(1.0)
    vis, opened = imgui.begin("Controls and shortcuts", True, imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings)
    try:
        if vis:
            if imgui.begin_table("##keys", 2, imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit):
                imgui.table_setup_column("k", imgui.TableColumnFlags_.width_fixed, 210.0 * ui.s)
                imgui.table_setup_column("d", imgui.TableColumnFlags_.width_stretch, 1.0)
                for k, d in SHORTCUTS:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    if d is None:
                        imgui.spacing()
                        bold(k, theme.ACCENT2)
                        imgui.table_next_column()
                        continue
                    imgui.text_colored(V4(*theme.TEXT), k)
                    imgui.table_next_column()
                    imgui.push_text_wrap_pos(0.0)
                    imgui.text_colored(V4(*theme.TEXT_DIM), d)
                    imgui.pop_text_wrap_pos()
                imgui.end_table()
    finally:
        imgui.end()
    if opened is False:
        ui.show_help = False


def _about(ui):
    if not ui.show_about:
        return
    _centered(ui, "about", 600, 500)
    _raise(ui, "about")
    imgui.set_next_window_bg_alpha(1.0)
    vis, opened = imgui.begin("About", True, imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings)
    try:
        if vis:
            font_text(theme.Fonts.bold, 22.0, TITLE, theme.ACCENT2)
            dim("Desktop GPU viewer (wgpu / Vulkan)")
            imgui.spacing()
            imgui.push_text_wrap_pos(0.0)
            imgui.text_wrapped("Interactive 3D flood visualisation for Bowling Green and Warren County, Kentucky: simulated "
                               "water depth, derived flow velocity, buildings and roads coloured by flooding, and live "
                               "statistics for each storm and hour.  Switch between the city and the whole county in the "
                               "Domain menu.")
            imgui.pop_text_wrap_pos()
            imgui.spacing()
            bold("Data and credits", theme.ACCENT2)
            for c in CREDITS:
                imgui.bullet()
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(V4(*theme.TEXT), c)
                imgui.pop_text_wrap_pos()
            imgui.spacing()
            bold("Honesty notes", theme.WARN)
            imgui.push_text_wrap_pos(0.0)
            dim("Velocity is derived from the water-surface slope (Manning), not solver output. Depth-tier previews are "
                "approximate: the severe tier is only known as >= 1.2 m. Indicative only; not an official flood map.", wrap=True)
            imgui.pop_text_wrap_pos()
            imgui.spacing()
            bold("Robert Lesoski", theme.TEXT)
            dim("Undergraduate researcher, Disaster Science Operations Center (DSOC) | AI Research Lab (AIR), "
                "Western Kentucky University", wrap=True)
            dim("Principal investigator: Dr. Manmeet Singh", wrap=True)
            imgui.spacing()
            if imgui.button("Close"):
                ui.show_about = False
    finally:
        imgui.end()
    if opened is False:
        ui.show_about = False


def _source(ui):
    if not ui.show_source:
        return
    app = ui.app
    st = app.storm
    L = ui.L
    w = 440.0 * ui.s
    imgui.set_next_window_pos(V2(L.W - w - 14.0 * ui.s, L.H - L.foot - 8.0 * ui.s), imgui.Cond_.appearing, V2(0.0, 1.0))
    imgui.set_next_window_pos(V2(L.W - 14.0 * ui.s, L.H - L.foot - 10.0 * ui.s), imgui.Cond_.always, V2(1.0, 1.0))
    imgui.set_next_window_size_constraints(V2(w, 0.0), V2(w, 520.0 * ui.s))
    imgui.set_next_window_bg_alpha(1.0)
    _raise(ui, "source")
    vis, opened = imgui.begin("Data source details", True, imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings
                              | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move)
    try:
        if vis:
            label, full = source_info(st)
            text_c(theme.OK if full else theme.WARN, label)
            dim(f"{getattr(st, 'label', st.name)}  |  storm {st.name}  |  hours {st.hours[0]}-{st.hours[-1]}", wrap=True)
            imgui.spacing()
            note = str(st.meta.get("source_note", "(no source note in meta.json)"))
            imgui.push_text_wrap_pos(w - 24.0 * ui.s)
            imgui.text_colored(V4(*theme.TEXT), note)
            imgui.pop_text_wrap_pos()
            vm = st.meta.get("velocity_method")
            if vm:
                imgui.spacing()
                dim("Velocity method: " + str(vm), wrap=True)
            rs = st.meta.get("run_summary") or {}
            if rs:
                imgui.spacing()
                dim(f"Solver {rs.get('solver', '?')}, total rain {float(rs.get('total_rain_mm', 0)):.0f} mm, "
                    f"CN {rs.get('cn_pervious', '?')}/{rs.get('cn_impervious', '?')}", wrap=True)
    finally:
        imgui.end()
    if opened is False:
        ui.show_source = False
