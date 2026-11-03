"""UiManager: owns the UI state, lays out every panel and draws them once per frame (called from App._gui through the
OverlayLayer).  Every panel runs inside `safe()`: an exception shows a red line in that panel and is logged once; the
render loop never sees it.

Layout (logical pixels, scaled by state.ui_scale):

    +--------------------------------------------------------------+  top bar: menus | storm | hour | source | fps
    | sidebar |  3D scene (legend bottom-left, compass top-left,    |
    | (left)  |  scale bar bottom-right)                 | right:   |  right: selected building card + at-risk list
    |         |                                          |          |
    |         +------------------------------------------+----------+
    |         |  timeline bar (transport + hour slider + chart)     |
    +---------+------------------------------------------+----------+
    | footer: honesty line (velocity is derived) + source badge     |
    +--------------------------------------------------------------+
"""
from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass

from imgui_bundle import imgui

from . import bridge, theme, units
from .atrisk import AddressIndex, AtRiskFilter, filter_ids
from .cameranav import CameraNav
from .probe import ProbeWorker
from .state import UiState
from .widgets import Toasts

V2 = imgui.ImVec2

PANEL_FLAGS = int(imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
                  | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_saved_settings
                  | imgui.WindowFlags_.no_bring_to_front_on_focus)
ALWAYS = int(imgui.Cond_.always)
AUTO_RESIZE = int(imgui.WindowFlags_.always_auto_resize)


@dataclass
class Layout:
    W: float = 1600.0
    H: float = 900.0
    top: float = 32.0
    foot: float = 26.0
    side_w: float = 340.0
    tl_h: float = 176.0
    right_w: float = 372.0
    margin: float = 8.0
    right_on: bool = True
    # scene rectangle (the part of the window showing the 3D view)
    sx0: float = 0.0
    sy0: float = 0.0
    sx1: float = 0.0
    sy1: float = 0.0

    @property
    def side_h(self) -> float:
        return self.H - self.top - self.foot

    @property
    def right_h(self) -> float:
        return self.sy1 - self.sy0 - 2 * self.margin


def compute_layout(W: float, H: float, st: UiState, top: float, s: float, right_visible: bool) -> Layout:
    """Pure function of the window size and the UI state (unit-tested)."""
    L = Layout(W=W, H=H, top=top, foot=26.0 * s)
    L.side_w = min(st.sidebar_w * s, W * 0.32) if st.show_sidebar else 0.0
    L.tl_h = min(max(st.timeline_h * s, 120.0), H * 0.32) if st.show_timeline else 0.0
    L.right_w = min(st.right_w * s, W * 0.30)
    L.right_on = bool(right_visible) and (W - L.side_w - L.right_w) > 340.0 * s
    L.sx0, L.sy0 = L.side_w, top
    L.sx1 = W - (L.right_w + 2 * L.margin if L.right_on else 0.0)
    L.sy1 = H - L.foot - L.tl_h
    return L


def _osm_credit(app, layout, scale):
    """BASEMAP agent: the OpenStreetMap credit (licence) while the map background shows; no-op otherwise."""
    from ..gpu.layers.basemap import draw_attribution
    draw_attribution(app, layout, scale)


class UiManager:
    def __init__(self, app, state: UiState | None = None):
        self.app = app
        self.st = state or UiState()
        self.L = Layout()
        self.toasts = Toasts()
        self.nav = CameraNav()
        self.probe_worker = ProbeWorker()
        self.error_logged: set[str] = set()
        self.panel_errors: dict[str, str] = {}
        self.panel_ms: dict[str, float] = {}
        self._memo: dict = {}
        self.lite = False
        self.legend_small_open = False
        self.narrow_list = False
        self._skipped = False
        self.top_h = 32.0
        self.card_h = 0.0
        self.s = 1.0
        # dialogs
        self.show_about = False
        self.show_help = False
        self.show_source = False
        self._open_about = False
        self._open_help = False
        # at-risk
        self.risk_filter = AtRiskFilter(tuple(self.st.risk_tiers), self.st.risk_search, self.st.risk_min_depth)
        self._addr_index: AddressIndex | None = None
        self._addr_for = None
        self._risk_key = None
        self.risk_ids = None
        self.scroll_to_selected = False
        # probe / pointer
        self.mouse = None                 # (x, y) logical px over the 3D view, else None
        self.ground_pick = None
        self.hover_building = None
        self._probe_key = None
        self._probe_t = 0.0
        self._pick_t = 0.0
        self._ground_t = 0.0
        self._ground_pending = False
        self._pick_pending = None
        self._press = None
        self.probe = None
        # misc
        self.shot_request = None          # None | 'ui' | 'scene'
        self.fullscreen = None
        self._first = True
        self._implot_ok = None
        self._scene_hint = 0.0
        self.frame_ct = 0
        self._prev_sel = None
        self._applied_scale = None

    # ------------------------------------------------------------------------------------------ helpers
    def px(self, v: float) -> float:
        return v * self.s

    def cached(self, key, fn):
        """Tiny memo keyed by `key` (only the last value per key kind is kept, so it never grows)."""
        c = self._memo.get(key[0])
        if c is not None and c[0] == key:
            return c[1]
        v = fn()
        self._memo[key[0]] = (key, v)
        return v

    def safe(self, name: str, fn, *a):
        """Run a panel; on error show a red line and log once.  Imgui begin/end pairing is the panel's own job, so
        panels use begin_panel()/end_panel() in try/finally and this wrapper only catches what escapes."""
        t0 = time.perf_counter()
        try:
            fn(self, *a)
            self.panel_errors.pop(name, None)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.panel_errors[name] = msg
            if name not in self.error_logged:
                self.error_logged.add(name)
                try:
                    self.app.report_error(f"ui.{name}", e)
                    if not getattr(self.app, "quiet", False):
                        traceback.print_exc()
                except Exception:
                    pass
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.panel_ms[name] = 0.9 * self.panel_ms.get(name, ms) + 0.1 * ms         # smoothed, shown in the F3 overlay

    def begin_panel(self, name: str, x: float, y: float, w: float, h: float, *, extra_flags: int = 0, bg_alpha=None,
                    padding=None, rounding=None, border: bool = True, auto_height: bool = False, pivot=None) -> bool:
        """Fixed-position panel.  auto_height: width fixed, height fits the content (`h` ignored); pivot (px, py) makes
        (x, y) the anchor point of the window (e.g. (0, 1) = bottom-left corner)."""
        imgui.set_next_window_pos(V2(x, y), ALWAYS, V2(*(pivot or (0.0, 0.0))))
        if auto_height:
            imgui.set_next_window_size_constraints(V2(w, 0.0), V2(w, 1.0e5))
            extra_flags |= AUTO_RESIZE
        else:
            imgui.set_next_window_size(V2(w, h))
        if bg_alpha is not None:
            imgui.set_next_window_bg_alpha(bg_alpha)
        pushed = 0
        if padding is not None:
            imgui.push_style_var(imgui.StyleVar_.window_padding, V2(*padding))
            pushed += 1
        if rounding is not None:
            imgui.push_style_var(imgui.StyleVar_.window_rounding, float(rounding))
            pushed += 1
        if not border:
            imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
            pushed += 1
        vis, _ = imgui.begin(name, None, PANEL_FLAGS | int(extra_flags))
        self._panel_stack = getattr(self, "_panel_stack", [])
        self._panel_stack.append(pushed)
        return vis

    def end_panel(self):
        imgui.end()
        n = self._panel_stack.pop() if getattr(self, "_panel_stack", None) else 0
        if n:
            imgui.pop_style_var(n)

    # ------------------------------------------------------------------------------------------ input
    def on_key(self, ev) -> bool:
        """Hotkeys from App._on_key (key_down only).  Returns True when consumed."""
        if ev.get("event_type") != "key_down":
            return False
        k = ev.get("key", "")
        kl = k.lower() if len(k) == 1 else k
        app = self.app
        mods = ev.get("modifiers") or ()
        if k == "F12":
            self.shot_request = "scene" if "Shift" in mods else "ui"
            return True
        if k == "F11":
            self.toggle_fullscreen()
            return True
        if k == "F3":
            self.st.show_perf = not self.st.show_perf
            return True
        if k == "F1":
            self.show_help = not self.show_help
            return True
        if kl == "h":
            self.st.hidden = not self.st.hidden
            return True
        if k in "123456789" and len(k) == 1 and not (set(mods) & {"Control", "Alt", "Meta"}):    # 1..9 pick a storm
            names = list(app.storms_available)
            if int(k) <= len(names):
                app.set_storm(names[int(k) - 1])
                return True
            return False
        if k == "Escape":
            if self.show_about or self.show_help:
                self.show_about = self.show_help = False
                return True
            if self.st.hidden:
                self.st.hidden = False
                return True
            sel = bridge.selection(app)
            if sel.get("building") is not None or sel.get("road") is not None:
                bridge.clear_selection(app)
                return True
            return False
        if k == "[":
            app.t = max(0.0, math.ceil(app.t - 1.0 - 1e-6))
            return True
        if k == "]":
            app.t = min(24.0, math.floor(app.t + 1.0 + 1e-6))
            return True
        if kl == "f":
            sel = bridge.selection(app)
            if sel.get("building") is not None:
                bridge.fly_to_building(app, int(sel["building"]))
                return True
            return False                                   # no selection: F tilts the camera as before
        if k == "Home":
            self.fly_preset("Overview")
            return True
        return False

    def over_ui(self, x: float, y: float) -> bool:
        """Pointer (logical px) outside the 3D view rectangle of the last drawn frame -> it is over the interface."""
        if self.frame_ct == 0 or self.st.hidden:
            return False
        L = self.L
        return not (L.sx0 <= x < L.sx1 and L.sy0 <= y < L.sy1)

    def toggle_fullscreen(self):
        from .windowing import Fullscreen
        if self.fullscreen is None:
            self.fullscreen = Fullscreen()
        ok = self.fullscreen.toggle(self.app.canvas)
        self.st.fullscreen = self.fullscreen.on
        if not ok:
            self.toasts.push("Fullscreen is not available here", theme.WARN)

    def fly_preset(self, name: str):
        app = self.app
        try:
            self.nav.fly(app.camera, *app.domain.preset_view(app.terrain.frame, name), seconds=1.1)
        except Exception as e:
            self.toasts.push(f"Camera preset failed: {e}", theme.ERR)

    def look_north(self):
        cam = self.app.camera
        self.nav.fly(cam, float(cam.target[0]), float(cam.target[1]), float(cam.dist), 0.0, float(cam.pitch), seconds=0.6)

    # ------------------------------------------------------------------------------------------ per frame
    CLICK_SLOP_PX = 4.0           # a press + release that moved more than this is a drag, not a click

    # adaptive refresh: when the app runs below ~58 fps AND the interface is a measurable part of the frame, the UI code
    # runs every second frame and the other frames replay its draw data (hysteresis: back to full rate above ~72 fps)
    LITE_ON_MS, LITE_OFF_MS, LITE_MIN_UI_MS = 17.2, 13.8, 2.0

    def lite_skip(self) -> bool:
        if not self.lite:
            self._skipped = False
            return False
        self._skipped = not self._skipped
        return self._skipped

    def _update_lite(self):
        fm = self.app.frame_ms
        if len(fm) < 40 or self.st.hidden:
            return
        mean = sum(list(fm)[-30:]) / 30.0
        ui_ms = sum(self.panel_ms.values())
        if not self.lite and mean > self.LITE_ON_MS and ui_ms > self.LITE_MIN_UI_MS:
            self.lite = True
        elif self.lite and mean < self.LITE_OFF_MS:
            self.lite = False

    def update(self, dt: float):
        """Called from the layer's update() (every frame, even when the UI is hidden)."""
        try:
            self._update_lite()
            self.nav.update(self.app.camera, dt)
        except Exception as e:
            self.nav.cancel()
            self.app.report_error("ui.nav", e)

    # narrow windows (< 1180 logical px wide): the right-hand at-risk list would squeeze the 3D view, so it only appears
    # when the user asks for it (menu / sidebar checkbox) - the selected-building card still shows on a selection
    NARROW_W = 1180.0

    def is_narrow(self) -> bool:
        return self.L.W / max(self.s, 0.1) < self.NARROW_W

    def atrisk_shown(self) -> bool:
        return bool(self.st.show_atrisk and (not self.is_narrow() or self.narrow_list))

    def set_atrisk(self, on: bool):
        if self.is_narrow():
            self.narrow_list = bool(on)
            if on:
                self.st.show_atrisk = True
        else:
            self.st.show_atrisk = bool(on)

    def sync_risk_filter(self):
        f = self.risk_filter
        self.st.risk_tiers = [bool(x) for x in f.tiers]
        self.st.risk_search = f.search
        self.st.risk_min_depth = float(f.min_depth_m)

    def risk_list(self):
        """Filtered, deepest-first building ids of the current snapshot (cached by snapshot version + filter)."""
        fl = bridge.flood(self.app)
        b = bridge.buildings(self.app)
        if fl is None or b is None or fl.get("buildings") is None:
            return None
        key = (fl.get("version"), fl.get("storm"), id(b), self.risk_filter.key())
        if key != self._risk_key:
            if self._addr_for is not b:
                self._addr_index = AddressIndex(b)
                self._addr_for = b
            self.risk_ids = filter_ids(fl["buildings"], b, self.risk_filter, self._addr_index)
            self._risk_key = key
        return self.risk_ids

    def frame(self):
        """Draw the whole UI (inside imgui new_frame / render)."""
        from . import dialogs, menus, panels_overlay, panels_risk, panels_side, panels_time
        app = self.app
        st = self.st
        io = imgui.get_io()
        self.frame_ct += 1
        if self._first:
            self._first = False
            bridge.ensure_hub(app)
            self._init_once()
        if self._applied_scale != st.ui_scale:
            theme.set_scale(imgui, st.ui_scale)
            self._applied_scale = st.ui_scale
        self.s = float(st.ui_scale)
        W, H = float(io.display_size.x), float(io.display_size.y)
        self.safe("pointer", UiManager._pointer, io)
        if st.hidden:
            self.top_h = 0.0
            self.L = compute_layout(W, H, st, 0.0, self.s, False)
            self.L.foot = 0.0
            self.L.sx0, self.L.sy0, self.L.sx1, self.L.sy1 = 0.0, 0.0, W, H
            self.safe("probe", panels_overlay.draw_probe)
            self.safe("osm credit", lambda ui: _osm_credit(ui.app, ui.L, ui.s))
            self.toasts.draw(W * 0.5, H - 24)
            self._draw_hidden_hint(W, H)
            self._finish_frame()
            return
        self.safe("menus", menus.draw)
        right_visible = bool(self.atrisk_shown() or (st.show_selected and bridge.selection(app).get("building") is not None))
        self.L = compute_layout(W, H, st, self.top_h, self.s, right_visible)
        self.safe("sidebar", panels_side.draw_sidebar) if st.show_sidebar else None
        self.safe("timeline", panels_time.draw_timeline) if st.show_timeline else None
        if self.L.right_on:
            self.safe("selected", panels_risk.draw_selected)
            self.safe("atrisk", panels_risk.draw_atrisk)
        self.safe("footer", panels_overlay.draw_footer)
        self.safe("legends", panels_overlay.draw_legends)
        self.safe("compass", panels_overlay.draw_compass)
        self.safe("scalebar", panels_overlay.draw_scalebar)
        self.safe("osm credit", lambda ui: _osm_credit(ui.app, ui.L, ui.s))     # BASEMAP: licence credit while the map shows
        self.safe("perf", panels_overlay.draw_perf)
        self.safe("probe", panels_overlay.draw_probe)
        self.safe("dialogs", dialogs.draw)
        self.toasts.draw((self.L.sx0 + self.L.sx1) * 0.5, self.L.sy1 - 14.0 * self.s)
        self._finish_frame()

    def _finish_frame(self):
        sel = bridge.selection(self.app).get("building")
        if sel != self._prev_sel:
            self._prev_sel = sel
            self.scroll_to_selected = sel is not None

    def _init_once(self):
        from . import panels_side, windowing
        try:
            self.app.quality = panels_side.APP_QUALITY.get(self.st.quality_preset, getattr(self.app, "quality", None))
        except Exception:
            pass
        try:
            windowing.set_min_size(self.app.canvas)
        except Exception:
            pass
        if not self.st.seen_help:
            self.toasts.push("Press F1 for the controls   |   H hides the interface   |   F12 saves a screenshot", theme.ACCENT2, 7.0)
            self.st.seen_help = True

    def _draw_hidden_hint(self, W, H):
        if self.frame_ct % 2 == 0 or True:
            dl = imgui.get_foreground_draw_list()
            s = "Interface hidden - press H"
            ts = imgui.calc_text_size(s)
            a = 0.45
            dl.add_text(V2(W - ts.x - 14, H - ts.y - 10), theme.col32((1, 1, 1, a)), s)

    # ------------------------------------------------------------------------------------------ pointer / probe
    def _pointer(self, io):
        """Mouse over the 3D view: ground pick + building under the cursor + click selection (all from imgui's io,
        so it works whichever canvas event order is used)."""
        app = self.app
        mp = io.mouse_pos
        over_view = bool(mp.x > -1e4 and not io.want_capture_mouse)
        left = imgui.is_mouse_down(0)
        any_down = left or imgui.is_mouse_down(1) or imgui.is_mouse_down(2)
        # ---- click detection (press + release without moving) selects a building
        if imgui.is_mouse_clicked(0) and over_view:
            self._press = (mp.x, mp.y, time.perf_counter())
        if imgui.is_mouse_released(0) and self._press is not None:
            x0, y0, t0 = self._press
            self._press = None
            if over_view and math.hypot(mp.x - x0, mp.y - y0) <= self.CLICK_SLOP_PX and time.perf_counter() - t0 < 0.6:
                self._click(mp.x, mp.y)
        if self.nav.active and any_down:
            self.nav.cancel()
        if not over_view or any_down or self.nav.active:
            self.mouse = None
            self.probe_active = False
            self._set_hover(None)
            return
        moved = self.mouse is None or abs(mp.x - self.mouse[0]) + abs(mp.y - self.mouse[1]) > 0.5
        now = time.perf_counter()
        self.mouse = (float(mp.x), float(mp.y))
        if moved:
            self._ground_pending = True
            self._pick_pending = now
        t_changed = abs(app.t - self._probe_t) > 0.01 and app.playing
        # the ground ray-cast is ~1 ms of numpy: at most 30 per second while the mouse moves (the tooltip follows the
        # cursor every frame, its numbers lag by at most one 33 ms step); it always catches up when the mouse stops
        if (self._ground_pending or t_changed or self.ground_pick is None) and now - self._ground_t >= 0.033:
            self._ground_pending = False
            self._ground_t = now
            pk = app.camera.pick(mp.x, mp.y)
            self.ground_pick = pk
            if pk is not None:
                key = (round(pk[0], 1), round(pk[1], 1), round(app.t, 2), app.storm.name)
                if key != self._probe_key:
                    self._probe_key = key
                    self._probe_t = app.t
                    self.probe_worker.request(app.storm, app.t, pk[0], pk[1], app.terrain.sampler, key)
        # building under the cursor: debounced (the ray march costs a few ms), so a moving mouse stays cheap
        pend = self._pick_pending
        if pend is not None and now - pend > 0.06 and now - self._pick_t > 0.10:
            self._pick_pending = None
            self._pick_t = now
            self._set_hover(bridge.hover_building(app, mp.x, mp.y))

    def _set_hover(self, bid):
        self.hover_building = bid
        bridge.selection(self.app)["hover_building"] = bid

    def _click(self, x: float, y: float):
        bridge.click_select(self.app, x, y)
        if bridge.selection(self.app).get("building") is not None:
            self.scroll_to_selected = True
