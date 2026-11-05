"""Cinematic panel (Bookmarks | Tour | Time-lapse | Export), the Cinematic menu, tour captions + controls, the export progress
bar and the hotkeys.  Registered with the UI hooks by `register()` (called from CinematicLayer.init, idempotent).

Hotkeys: Ctrl+B save a bookmark, Ctrl+1..9 jump to bookmark n, Alt+1..9 overwrite bookmark n with the current view,
T start / stop the tour, Space pause / resume a running tour, N / P next / previous place, Esc stops the tour,
L play / pause the time-lapse, C show / hide the Cinematic panel, Ctrl+E open the export tab.
"""
from __future__ import annotations

import math

from imgui_bundle import imgui

from .. import tools_cinematic as tc
from . import hooks, theme, units
from .theme import col32
from .widgets import bold, dim, hbar, pill, sparkline_depth, text_c

V2 = imgui.ImVec2
V4 = imgui.ImVec4
TABS = ("Bookmarks", "Tour", "Time-lapse", "Export")
SHORTCUT_ROWS = [
    ("Cinematic tools", None),
    ("Ctrl+B", "Save the current view as a bookmark"),
    ("Ctrl+1 .. 9", "Fly to bookmark 1 .. 9 (Alt+1 .. 9 overwrites it with the current view)"),
    ("T", "Start / stop the automatic tour (Space pauses, N / P next / previous place, Esc stops)"),
    ("L", "Play / pause the time-lapse"),
    ("C", "Show / hide the Cinematic panel (Ctrl+E: export tab)"),
]


def register():
    hooks.register_panel("cinematic panel", draw_panel)
    hooks.register_overlay("cinematic overlay", draw_overlay)
    hooks.register_menu("Cinematic", draw_menu)
    hooks.register_key("cinematic keys", on_key)
    hooks.register_shortcuts("cinematic", SHORTCUT_ROWS)


def layer(ui):
    lay = ui.app.get_layer("cinematic")
    return lay if (lay is not None and lay.app is not None) else None


# ------------------------------------------------------------------------------------------------ keys
def on_key(ui, ev) -> bool:
    lay = layer(ui)
    if lay is None or ev.get("event_type") != "key_down":
        return False
    k = ev.get("key", "")
    kl = k.lower() if len(k) == 1 else k
    mods = set(ev.get("modifiers") or ())
    ctrl, alt = bool(mods & {"Control", "Meta"}), "Alt" in mods
    tour_on = lay.tour.active
    if ctrl and kl == "b":
        lay.save_bookmark()
        return True
    if ctrl and len(k) == 1 and k in "123456789":
        lay.jump(int(k) - 1)
        return True
    if alt and len(k) == 1 and k in "123456789":
        i = int(k) - 1
        if i < len(lay.bookmarks()):
            lay.save_bookmark(None, i)
            return True
        return False
    if ctrl and kl == "e":
        lay.panel_open, lay.tab = True, 3
        lay._tab_request = 3
        return True
    if ctrl or alt:
        return False
    if kl == "t":
        lay.tour_toggle()
        return True
    if kl == "l":
        lay.toggle_lapse()
        return True
    if kl == "c":
        lay.panel_open = not lay.panel_open
        return True
    if tour_on:
        if k == " ":
            lay.tour.paused = not lay.tour.paused
            return True
        if kl == "n":
            lay.tour_skip(1)
            return True
        if kl == "p":
            lay.tour_skip(-1)
            return True
        if k == "Escape":
            lay.stop_tour()
            return True
    elif lay.flight.active and k == "Escape":
        lay.flight.cancel()
        return True
    if lay.lapse.playing and k == " ":                 # Space pauses whichever motion runs
        lay.lapse.playing = False
        return True
    return False


# ------------------------------------------------------------------------------------------------ menu
def draw_menu(ui):
    lay = layer(ui)
    if lay is None:
        return
    if imgui.begin_menu("Cinematic"):
        if imgui.menu_item("Show Cinematic panel", "C", lay.panel_open)[0]:
            lay.panel_open = not lay.panel_open
        imgui.separator()
        if imgui.menu_item("Save current view as bookmark", "Ctrl+B", False)[0]:
            lay.save_bookmark()
        if imgui.begin_menu("Go to bookmark", len(lay.bookmarks()) > 0):
            for i, b in enumerate(lay.bookmarks()):
                if imgui.menu_item(b.name, f"Ctrl+{i + 1}" if i < 9 else "", False)[0]:
                    lay.jump(i)
            imgui.end_menu()
        imgui.separator()
        if imgui.menu_item("Stop tour" if lay.tour.active else "Start tour of key places", "T", False)[0]:
            lay.tour_toggle()
        if imgui.menu_item("Play / pause time-lapse", "L", lay.lapse.playing)[0]:
            lay.toggle_lapse()
        imgui.separator()
        if imgui.menu_item("Export clip ...", "Ctrl+E", False)[0]:
            lay.panel_open, lay.tab, lay._tab_request = True, 3, 3
        imgui.end_menu()


# ------------------------------------------------------------------------------------------------ panel
def draw_panel(ui):
    lay = layer(ui)
    if lay is None or not lay.panel_open:
        return
    L, s = ui.L, ui.s
    w = 400.0 * s
    imgui.set_next_window_pos(V2(max(L.sx1 - w - 14.0 * s, L.sx0 + 10.0), L.sy0 + 12.0 * s), int(imgui.Cond_.first_use_ever))
    imgui.set_next_window_size(V2(w, 560.0 * s), int(imgui.Cond_.first_use_ever))
    imgui.set_next_window_bg_alpha(0.96)
    flags = int(imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_collapse)
    vis, opened = imgui.begin("Cinematic tools", True, flags)
    try:
        if not opened:
            lay.panel_open = False
        if vis:
            req = getattr(lay, "_tab_request", None)
            if imgui.begin_tab_bar("##cinetabs"):
                for i, name in enumerate(TABS):
                    tf = int(imgui.TabItemFlags_.set_selected) if req == i else 0
                    sel, _ = imgui.begin_tab_item(name, None, tf)
                    if sel:
                        lay.tab = i
                        try:
                            (_tab_bookmarks, _tab_tour, _tab_lapse, _tab_export)[i](ui, lay)
                        finally:
                            imgui.end_tab_item()
                imgui.end_tab_bar()
            lay._tab_request = None
    finally:
        imgui.end()


def _depth_now(ui, lay, lng, lat, t):
    """(text, series) of the depth at a place as of hour t from the depth source (None series while it is being read)."""
    src = lay.depth_source()
    if src.kind == "none":
        return "no depth data", None
    s = lay.depth_series(lng, lat)
    if s is None:
        return "reading ...", None
    from ..data.depthtiles import mix_hours
    d = mix_hours(s, t)
    return ("-" if d != d else f"{d:.2f} m"), s


def _tab_bookmarks(ui, lay):
    app = ui.app
    s = ui.s
    imgui.spacing()
    if imgui.button("Save current view   (Ctrl+B)"):
        lay.save_bookmark()
    imgui.same_line()
    ch, v = imgui.checkbox("restore hour + storm", bool(lay.restore_time))
    if ch:
        lay.restore_time = bool(v)
    imgui.spacing()
    items = lay.bookmarks()
    if not items:
        dim("No bookmarks for this area yet. Frame a view, then press Ctrl+B.", wrap=True)
    frame = app.terrain.frame
    delete = None
    flags = int(imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit)
    if items and imgui.begin_table("##bms", 4, flags):
        imgui.table_setup_column("#", imgui.TableColumnFlags_.width_fixed, 22.0 * s)
        imgui.table_setup_column("name", imgui.TableColumnFlags_.width_stretch, 1.0)
        imgui.table_setup_column("depth", imgui.TableColumnFlags_.width_fixed, 84.0 * s)
        imgui.table_setup_column("act", imgui.TableColumnFlags_.width_fixed, 110.0 * s)
        for i, b in enumerate(items):
            imgui.push_id(i)
            imgui.table_next_row()
            imgui.table_next_column()
            dim(str(i + 1) if i < 9 else " ")
            imgui.table_next_column()
            imgui.set_next_item_width(-1)
            ch, nm = imgui.input_text("##name", b.name)
            if ch and nm.strip():
                b.name = nm
            if imgui.is_item_hovered():
                extra = f"  (saved at hour {b.t:.1f}, {b.storm})" if b.t is not None else ""
                imgui.set_tooltip(f"Edit the name here.{extra}")
            imgui.table_next_column()
            try:
                lng, lat = frame.world_to_lnglat(b.view[0], b.view[1])
                txt, _ = _depth_now(ui, lay, lng, lat, app.t)
                dim(txt)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Water depth at the view centre as of hour {app.t:.1f}\n{lay.depth_source().label}")
            except Exception:
                dim("-")
            imgui.table_next_column()
            if imgui.small_button("Go"):
                lay.jump(i)
            imgui.same_line()
            if imgui.small_button("Set"):
                lay.save_bookmark(None, i)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Overwrite this bookmark with the current view")
            imgui.same_line()
            if imgui.small_button("X"):
                delete = i
            imgui.pop_id()
        imgui.end_table()
    if delete is not None:
        lay.store.delete(lay.domain_key, delete)
    if items:
        imgui.spacing()
        dim("Depth column: " + lay.depth_source().label, wrap=True)
        dim("Ctrl+1..9 flies to a bookmark; Alt+1..9 overwrites it.", wrap=True)
    if lay.flight.active:
        imgui.spacing()
        text_c(theme.ACCENT2, "flying ... (any camera input stops it)")


def _tab_tour(ui, lay):
    app = ui.app
    o, rt = lay.tour_opts, lay.tour
    imgui.spacing()
    dim("Route: hospitals, schools, fire, police and shelters named in infrastructure.geojson, plus the most flooded named roads.",
        wrap=True)
    imgui.spacing()
    for n, (key, label) in enumerate((("hospital", "Hospitals"), ("school", "Schools"), ("fire_station", "Fire"),
                                      ("police", "Police"), ("assembly_point", "Shelters"), ("clinic", "Clinics"))):
        on = key in o.types
        if n % 3:
            imgui.same_line(130.0 * ui.s * (n % 3))
        imgui.push_id(key)
        ch, v = imgui.checkbox(label, on)
        imgui.pop_id()
        if ch:
            (o.types.append(key) if v else o.types.remove(key))
    imgui.set_next_item_width(-170.0 * ui.s)
    ch, v = imgui.slider_int("places", o.max_facilities, 1, 30)
    o.max_facilities = int(v) if ch else o.max_facilities
    imgui.set_next_item_width(-170.0 * ui.s)
    ch, v = imgui.slider_int("flooded roads", o.max_roads, 0, 10)
    o.max_roads = int(v) if ch else o.max_roads
    imgui.set_next_item_width(-170.0 * ui.s)
    ch, v = imgui.slider_float("seconds per place", o.dwell_s, 3.0, 15.0, "%.0f")
    o.dwell_s = float(v) if ch else o.dwell_s
    imgui.set_next_item_width(-170.0 * ui.s)
    ch, v = imgui.slider_float("tour speed", o.speed, 0.25, 3.0, "x%.2f")
    o.speed = float(v) if ch else o.speed
    ch, v = imgui.checkbox("only places the storm floods (>= 0.1 m)", o.only_wet)
    o.only_wet = bool(v) if ch else o.only_wet
    ch, v = imgui.checkbox("escalate storms (small to large)", o.escalate)
    o.escalate = bool(v) if ch else o.escalate
    if o.escalate:
        imgui.same_line()
        imgui.set_next_item_width(70.0 * ui.s)
        ch, v = imgui.slider_int("##en", o.escalate_n, 2, 6)
        o.escalate_n = int(v) if ch else o.escalate_n
        dim("Storms in order: " + ", ".join(app.storm_label(x) for x in lay.tour_storms()), wrap=True)
    imgui.text("Hour during the tour:")
    for i, (key, label) in enumerate((("sweep", "rises 0 to 24"), ("hold", "hold current hour"))):
        imgui.same_line()
        if imgui.radio_button(label + f"##tm{i}", o.time_mode == key):
            o.time_mode = key
    ch, v = imgui.checkbox("hide the interface while touring (H shows it)", o.hide_ui)
    o.hide_ui = bool(v) if ch else o.hide_ui
    imgui.spacing()
    if rt.building:
        text_c(theme.WARN, "choosing places and reading depths ...")
    elif imgui.button("Build route" if rt.path is None else "Rebuild route"):
        lay.build_tour(False)
    imgui.same_line()
    if rt.active:
        if imgui.button("Stop tour (T)"):
            lay.stop_tour()
    elif imgui.button("Start tour (T)"):
        lay.tour_toggle()
    if rt.error:
        imgui.push_text_wrap_pos(0.0)
        text_c(theme.ERR, rt.error[:200])
        imgui.pop_text_wrap_pos()
    if rt.path is not None:
        p = rt.path
        imgui.spacing()
        m, sec = divmod(int(round(p.duration / max(rt.opts.speed, 0.01))), 60)
        dim(f"{len(p.stops)} places, {m}:{sec:02d} at x{rt.opts.speed:.2f}   ({rt.info.get('facilities', 0)} facilities, "
            f"{rt.info.get('roads', 0)} roads; {rt.info.get('unnamed', 0)} unnamed places skipped)", wrap=True)
        dim("Depth read-outs: " + str(rt.info.get("source", "")), wrap=True)
        imgui.begin_child("##stops", V2(0, 130.0 * ui.s), imgui.ChildFlags_.borders)
        for i, st in enumerate(p.stops):
            cur = rt.active and rt.state is not None and rt.state.caption_stop == i
            pk = "" if st.peak_m is None else f"   peak {st.peak_m:.2f} m"
            text_c(theme.ACCENT2 if cur else theme.TEXT, f"{i + 1}. {st.name}")
            imgui.same_line()
            dim(f"{st.label}{pk}")
        imgui.end_child()


def _tab_lapse(ui, lay):
    app = ui.app
    lp = lay.lapse
    imgui.spacing()
    label = "Pause (L)" if lp.playing else "Play (L)"
    if imgui.button(label, V2(96.0 * ui.s, 0)):
        lay.toggle_lapse()
    imgui.same_line()
    dim(f"Hour {units.snapshot_hour(app.t)} of 24   t = {units.hour_label(app.t)}   ({lp.rate:.2f} hours per second)")
    imgui.set_next_item_width(-1)
    ch, t = imgui.slider_float("##scrub", app.t, 0.0, 24.0, "t = %.2f h", imgui.SliderFlags_.always_clamp)
    if ch:
        app.t = float(t)
        lp.playing = False
    imgui.spacing()
    dim("Speed (hours per second)")
    imgui.set_next_item_width(-1)
    ch, v = imgui.slider_float("##rate", tc.slider_from_rate(lp.rate), 0.0, 1.0, "")
    if ch:
        lp.rate = tc.rate_from_slider(v)
    for r in tc.RATE_PRESETS:
        if imgui.small_button(f"{r:g}##r{r}"):
            lp.rate = float(r)
        imgui.same_line()
    imgui.new_line()
    dim("Range")
    imgui.set_next_item_width(-1)
    ch, a, b = imgui.drag_float_range2("##rng", lp.t_in, lp.t_out, 0.1, 0.0, 24.0, "from %.1f h", "to %.1f h")
    if ch:
        lp.t_in, lp.t_out = float(a), float(b)
    for i, (key, lab) in enumerate((("loop", "loop"), ("pingpong", "ping-pong"), ("once", "once"))):
        if imgui.radio_button(lab + f"##lm{i}", lp.mode == key):
            lp.mode = key
        imgui.same_line()
    imgui.new_line()
    imgui.spacing()
    dim("Between two hourly flow frames the water is blended on the GPU (depth and velocity codes are interpolated, then decoded), "
        "so any speed shows continuous water. Building colours update per snapshot hour.", wrap=True)
    # ---- flood timeline at the view centre
    imgui.spacing()
    bold("Water depth over time at the view centre", theme.ACCENT2)
    c = app.camera
    try:
        lng, lat = app.terrain.frame.world_to_lnglat(float(c.target[0]), float(c.target[1]))
        lng, lat = round(lng, 4), round(lat, 4)
        src = lay.depth_source()
        s = None if src.kind == "none" else lay.depth_series(lng, lat)
    except Exception:
        s, src = None, None
    p0 = imgui.get_cursor_screen_pos()
    sz = (imgui.get_content_region_avail().x, 70.0 * ui.s)
    if s is not None:
        vals = [0.0 if v != v else float(v) for v in s]
        sparkline_depth(imgui.get_window_draw_list(), (p0.x, p0.y), sz, vals, app.t, theme.ACCENT)
        imgui.dummy(V2(*sz))
        from ..data.depthtiles import mix_hours
        dim(f"{mix_hours(s, app.t):.2f} m now, peak {max(vals):.2f} m   |   {src.label}", wrap=True)
    else:
        imgui.dummy(V2(*sz))
        dim("reading depth ..." if src is not None and src.kind != "none" else "no depth data for this storm", wrap=True)


def _tab_export(ui, lay):
    from .. import export_cinematic as ex
    from ..gpu.layers.cinematic import FPS_CHOICES, SIZES
    app = ui.app
    e = lay.exp_opts
    job = lay.export
    busy = lay.export_active()
    imgui.spacing()
    imgui.begin_disabled(busy)
    kinds = (("time", "Time-lapse"), ("orbit", "Orbit"), ("flight", "Flight A to B"), ("tour", "Tour"))
    for i, (key, lab) in enumerate(kinds):
        if i:
            imgui.same_line()
        if imgui.radio_button(lab + f"##k{i}", e.kind == key):
            e.kind = key
    if e.kind == "time":
        dim("The current camera, the hour running over the time-lapse range (Time-lapse tab).", wrap=True)
    elif e.kind == "orbit":
        dim("The camera circles the view centre while the hour runs over the time-lapse range.", wrap=True)
    elif e.kind == "flight":
        dim("The camera flies from one bookmark to another while the hour runs.", wrap=True)
    else:
        dim("The route of the Tour tab, with its storms and hours, captions and depth read-outs.", wrap=True)
    imgui.spacing()
    if imgui.begin_combo("resolution", f"{SIZES[e.size_idx][0]} x {SIZES[e.size_idx][1]}"):
        for i, (w, h) in enumerate(SIZES):
            if imgui.selectable(f"{w} x {h}" + ("   (4K)" if w == 3840 else ""), i == e.size_idx)[0]:
                e.size_idx = i
        imgui.end_combo()
    if imgui.begin_combo("frame rate", f"{e.fps} fps"):
        for f in FPS_CHOICES:
            if imgui.selectable(f"{f} fps", f == e.fps)[0]:
                e.fps = f
        imgui.end_combo()
    if e.kind != "tour":
        ch, v = imgui.slider_float("seconds", e.seconds, 2.0, 120.0, "%.0f")
        e.seconds = float(v) if ch else e.seconds
        ch, v = imgui.slider_float("time-lapse loops", e.loops, 1.0, 4.0, "%.0f")
        e.loops = float(round(v)) if ch else e.loops
    if e.kind == "orbit":
        ch, v = imgui.slider_float("orbit degrees", e.orbit_deg, -360.0, 360.0, "%.0f")
        e.orbit_deg = float(v) if ch else e.orbit_deg
    if e.kind == "flight":
        names = [b.name for b in lay.bookmarks()]
        if len(names) < 2:
            text_c(theme.WARN, "Save at least two bookmarks first.")
        else:
            e.flight_a, e.flight_b = min(e.flight_a, len(names) - 1), min(e.flight_b, len(names) - 1)
            for attr, lab in (("flight_a", "from"), ("flight_b", "to")):
                cur = getattr(e, attr)
                if imgui.begin_combo(lab, names[cur]):
                    for i, n in enumerate(names):
                        if imgui.selectable(f"{i + 1}. {n}", i == cur)[0]:
                            setattr(e, attr, i)
                    imgui.end_combo()
    ff = ex.find_ffmpeg()
    ch, v = imgui.checkbox("make an MP4 when done", bool(e.mp4))
    e.mp4 = bool(v) if ch else e.mp4
    imgui.same_line()
    ch, v = imgui.checkbox("keep the PNG frames", bool(e.keep_png))
    e.keep_png = bool(v) if ch else e.keep_png
    imgui.end_disabled()
    if ff:
        dim("ffmpeg found: " + ff, wrap=True)
    else:
        text_c(theme.WARN, "ffmpeg not found on PATH:")
        dim("the PNG frames are written, plus make_mp4.bat / make_mp4.sh to build the MP4 where ffmpeg is installed.", wrap=True)
    dim(f"Folder: {ex.default_dir(app)}", wrap=True)
    shot, why = lay.make_shot()
    if shot is not None:
        n = tc.clip_frames(shot.duration, e.fps)
        dim(f"{n} frames, {shot.duration:.1f} s   |   the stamp says: model unvalidated, velocities derived, data source.")
    else:
        text_c(theme.WARN, why)
    imgui.spacing()
    if busy:
        _progress_block(ui, lay)
    else:
        imgui.begin_disabled(shot is None and not (e.kind == "tour" and lay.tour.path is None))
        if imgui.button("Start export", V2(150.0 * ui.s, 0)):
            if shot is None and e.kind == "tour":
                lay.build_tour(False)
            else:
                lay.start_export()
        imgui.end_disabled()
        if job is not None and job.state in ("done", "cancelled", "error"):
            imgui.spacing()
            imgui.push_text_wrap_pos(0.0)
            text_c(theme.ERR if job.state == "error" else theme.OK, job.message)
            imgui.pop_text_wrap_pos()
            if job.state != "error" and imgui.small_button("Open folder"):
                try:
                    import os
                    os.startfile(str(job.spec.out_dir))
                except Exception as ex_:
                    ui.toasts.push(f"Cannot open the folder: {ex_}", theme.ERR)


def _progress_block(ui, lay):
    job = lay.export
    hbar(job.progress, theme.ACCENT, 8.0)
    i = max(job.i, 0)
    eta = job.eta_s()
    pre = f"   (warm-up {job.spec.preroll + job.i}/{job.spec.preroll})" if job.i < 0 else ""
    dim(f"{job.message}: frame {min(i, job.n)}/{job.n}{pre}" + (f"   about {eta:.0f} s left" if eta else ""))
    if imgui.button("Cancel export", V2(150.0 * ui.s, 0)):
        job.cancel()
        lay.note(job.message)


# ------------------------------------------------------------------------------------------------ overlay
def draw_overlay(ui):
    lay = layer(ui)
    if lay is None:
        return
    if lay.export_active():
        lay.export_tick()                      # one frame per drawn UI frame (the window stays alive and shows the clip)
    _export_bar(ui, lay)
    _tour_controls(ui, lay)
    _caption(ui, lay)


def _export_bar(ui, lay):
    job = lay.export
    if job is None or not lay.export_active():
        return
    L, s = ui.L, ui.s
    w = 460.0 * s
    if not ui.begin_panel("##exportbar", (L.sx0 + L.sx1) * 0.5, L.sy0 + 14.0 * s, w, 0.0, auto_height=True, pivot=(0.5, 0.0),
                          bg_alpha=0.93, padding=(14.0 * s, 10.0 * s)):
        ui.end_panel()
        return
    try:
        _progress_block(ui, lay)
        dim(f"{job.W} x {job.H} @ {job.spec.fps:g} fps into {job.spec.out_dir.name}")
    finally:
        ui.end_panel()


def _tour_controls(ui, lay):
    rt = lay.tour
    if not rt.active or rt.path is None:
        return
    L, s = ui.L, ui.s
    w = 520.0 * s
    if not ui.begin_panel("##tourbar", (L.sx0 + L.sx1) * 0.5, L.sy1 - 14.0 * s, w, 0.0, auto_height=True, pivot=(0.5, 1.0),
                          bg_alpha=0.9, padding=(12.0 * s, 8.0 * s)):
        ui.end_panel()
        return
    try:
        if imgui.small_button("|<  Prev"):
            lay.tour_skip(-1)
        imgui.same_line()
        if imgui.small_button("Resume" if rt.paused else "Pause "):
            rt.paused = not rt.paused
        imgui.same_line()
        if imgui.small_button("Next  >|"):
            lay.tour_skip(1)
        imgui.same_line()
        if imgui.small_button("Stop"):
            lay.stop_tour()
        imgui.same_line()
        st = rt.state
        idx = (st.caption_stop if st is not None and st.caption_stop is not None else (st.stop if st is not None else 0))
        dim(f"place {min(idx + 1, len(rt.path.stops))} of {len(rt.path.stops)}   {app_label(ui)}"
            + ("   paused" if rt.paused else "") + ("   loading storm data ..." if rt.buffering else ""))
        hbar(rt.T / max(rt.path.duration, 1e-6), theme.ACCENT, 5.0)
    finally:
        ui.end_panel()


def app_label(ui) -> str:
    a = ui.app
    return f"{a.storm_label(a.storm.name)}  hour {units.snapshot_hour(a.t)}"


def _caption(ui, lay):
    cap = lay.tour_caption()
    if cap is None or cap["alpha"] <= 0.01:
        return
    L, s = ui.L, ui.s
    a = float(cap["alpha"])
    dl = imgui.get_foreground_draw_list()
    fb = theme.Fonts.bold
    title, sub, extra = cap["title"], cap["subtitle"], cap.get("extra", "")
    if fb is not None:
        imgui.push_font(fb, 30.0 * s)
    tw = imgui.calc_text_size(title)
    th = tw.y
    if fb is not None:
        imgui.pop_font()
    sw, ew = imgui.calc_text_size(sub), imgui.calc_text_size(extra)
    pad = 14.0 * s
    w = max(tw.x, sw.x, ew.x) + 2 * pad + 8.0 * s
    h = th + sw.y + ew.y + 2 * pad
    cx = (L.sx0 + L.sx1) * 0.5
    y1 = L.sy1 - (86.0 * s if lay.tour.active else 30.0 * s) - (0 if not ui.st.hidden else 0)
    x0 = cx - w * 0.5
    dl.add_rect_filled(V2(x0, y1 - h), V2(x0 + w, y1), col32((0.03, 0.05, 0.10, 0.80 * a)), 10.0 * s)
    dl.add_rect_filled(V2(x0, y1 - h), V2(x0 + 5.0 * s, y1), col32(theme.with_alpha(theme.ACCENT, a)), 3.0)
    x, y = x0 + pad + 6.0 * s, y1 - h + pad * 0.8
    if fb is not None:
        imgui.push_font(fb, 30.0 * s)
    dl.add_text(V2(x, y), col32((1, 1, 1, a)), title)
    if fb is not None:
        imgui.pop_font()
    y += th
    dl.add_text(V2(x, y), col32(theme.with_alpha(theme.ACCENT2, a)), sub)
    y += sw.y
    dl.add_text(V2(x, y), col32(theme.with_alpha(theme.TEXT, 0.9 * a)), extra)
