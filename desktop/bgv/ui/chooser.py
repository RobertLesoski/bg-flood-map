"""The City / County chooser on the loading screen.

At start-up (no `--domain` flag) the loading screen shows two big buttons, "City: Bowling Green" and "County: Warren County".
The domain remembered in the settings (default City) is built at once in the background - so a start without a click is as
fast as ever and only ONE domain is ever built - and:

    * a click on the other button restarts the build for that domain (`App.set_domain`) and continues as soon as it is ready;
    * a click on the building domain's button continues without waiting;
    * when BOTH domains have data, the loading screen does not close before a short countdown (3 s from the first frame of the
      chooser) has run out, so there is time to choose; "Cancel" stops the countdown and waits for a click.  When only the
      city has data (the county is a greyed-out button with a tooltip saying why) there is nothing to choose and nothing
      to wait for.
    * the choice is remembered (`settings["domain"]`, written when the app closes); `--domain` skips the chooser;
      `settings["domain_chooser"] = false` (Domain menu: "Ask at start-up") turns it off.

`DomainChooser` is the pure state machine (unit-tested); `draw(app)` draws it inside the imgui loading screen.
"""
from __future__ import annotations

COUNTDOWN_S = 3.0


class DomainChooser:
    def __init__(self, infos, building: str, *, countdown: float = COUNTDOWN_S, auto: bool = True):
        self.infos = list(infos)                 # [bgv.domain.DomainInfo]
        self.building = building                 # the domain whose data is being prepared right now
        self.countdown = float(countdown)
        self.auto = bool(auto)
        self.t0: float | None = None             # when the chooser was first drawn
        self.cancelled = False
        self.chosen = False
        self.request: str | None = None          # a domain the user clicked that is not the one being built (App consumes it)

    # ------------------------------------------------------------------------------------------ queries
    def info(self, key: str):
        return next((i for i in self.infos if i.key == key), None)

    @property
    def n_available(self) -> int:
        return sum(1 for i in self.infos if i.available)

    @property
    def waits(self) -> bool:
        """True when the loading screen must wait for the countdown / a click (there is something to choose)."""
        return self.auto and self.n_available >= 2

    def start(self, now: float):
        if self.t0 is None:
            self.t0 = now

    def remaining(self, now: float) -> float:
        if self.t0 is None:
            return self.countdown
        return max(0.0, self.countdown - (now - self.t0))

    def counting(self, now: float) -> bool:
        return self.waits and not self.chosen and not self.cancelled and self.remaining(now) > 0.0

    def may_finish(self, now: float) -> bool:
        """May the loading screen close (the building domain is ready)?"""
        if self.chosen or not self.waits:
            return True
        if self.cancelled:
            return False
        if self.t0 is None:
            return False                         # never finish before the user could see the chooser
        return now - self.t0 >= self.countdown

    # ------------------------------------------------------------------------------------------ input
    def click(self, key: str) -> str:
        """The user clicked a button.  'continue' (the building domain), 'switch' (another one: `request` is set),
        'unavailable' (greyed out: nothing happens)."""
        info = self.info(key)
        if info is None or not info.available:
            return "unavailable"
        self.chosen = True
        if key != self.building:
            self.request = key
            return "switch"
        return "continue"

    def cancel(self):
        """Stop the countdown: the loading screen then waits for a click."""
        if not self.chosen:
            self.cancelled = True

    def take_request(self) -> str | None:
        r, self.request = self.request, None
        return r

    def status_text(self, now: float, label: str) -> str:
        if self.chosen:
            return f"Opening {label} ..."
        if not self.waits:
            return ""
        if self.cancelled:
            return "Auto-start cancelled - click City or County to continue"
        if self.t0 is None or self.remaining(now) > 0.0:
            return f"Continuing with {label} in {self.remaining(now):.0f} s"
        return f"Continuing with {label} ..."


BUTTONS = (("city", "City", "Bowling Green"), ("county", "County", "Warren County"))


def draw(app) -> None:
    """The chooser inside the imgui loading screen (called by App._loading_gui, inside an imgui frame)."""
    import time
    from imgui_bundle import imgui
    V2 = imgui.ImVec2
    ch: DomainChooser | None = getattr(app, "chooser", None)
    if ch is None:
        return
    now = time.perf_counter()
    ch.start(now)
    io = imgui.get_io()
    W, H = io.display_size.x, io.display_size.y
    bw = min(230.0, max(W * 0.30, 120.0))
    bh = 70.0
    gap = 18.0
    total = 2 * bw + gap
    x0 = (W - total) * 0.5
    y0 = H * 0.5 + 54.0
    imgui.set_next_window_pos((x0 - 12.0, y0 - 8.0))
    imgui.set_next_window_size((total + 24.0, bh + 92.0))
    flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_background
             | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_scrollbar)
    imgui.begin("##domain_chooser", None, flags)
    try:
        dl = imgui.get_window_draw_list()
        col = lambda r, g, b, a=1.0: imgui.get_color_u32((r, g, b, a))     # noqa: E731
        for i, (key, title, sub) in enumerate(BUTTONS):
            info = ch.info(key)
            avail = bool(info is not None and info.available)
            x = x0 + i * (bw + gap)
            imgui.set_cursor_screen_pos((x, y0))
            clicked = imgui.invisible_button(f"##choose_{key}", (bw, bh))
            hov = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
            building = key == ch.building
            if avail:
                bg = (0.20, 0.42, 0.68, 1.0) if building else (0.13, 0.19, 0.28, 1.0)
                if hov:
                    bg = tuple(min(c + 0.07, 1.0) for c in bg[:3]) + (1.0,)
                border = (0.45, 0.78, 1.0, 1.0) if building else (0.30, 0.38, 0.50, 1.0)
                tcol, scol = (1, 1, 1, 1), (0.78, 0.86, 0.97, 1)
            else:
                bg, border = (0.10, 0.12, 0.16, 1.0), (0.20, 0.24, 0.30, 1.0)
                tcol, scol = (0.45, 0.48, 0.54, 1), (0.36, 0.39, 0.45, 1)
            p0 = V2(x, y0)
            p1 = V2(x + bw, y0 + bh)
            dl.add_rect_filled(p0, p1, col(*bg), 10.0)
            dl.add_rect(p0, p1, col(*border), 10.0, 2.0 if building and avail else 1.0)
            label = sub + ("  (DEMO DATA)" if info is not None and info.demo else "")
            tw = imgui.calc_text_size(title).x
            dl.add_text(V2(x + (bw - tw) * 0.5, y0 + 12.0), col(*tcol), title)
            sw = imgui.calc_text_size(label).x
            dl.add_text(V2(x + max((bw - sw) * 0.5, 6.0), y0 + 38.0), col(*scol), label)
            if hov and info is not None and (info.reason or not avail):
                from .menus import domain_tooltip
                domain_tooltip(info.reason or "not available")
            if clicked and avail:
                ch.click(key)
        # status line + cancel
        imgui.set_cursor_screen_pos((x0, y0 + bh + 14.0))
        label = next((s for k, t, s in BUTTONS if k == ch.building), ch.building)
        msg = ch.status_text(now, label)
        if msg:
            imgui.text_colored((0.62, 0.74, 0.90, 1.0), msg)
        if ch.counting(now):
            imgui.same_line()
            if imgui.small_button("Cancel"):
                ch.cancel()
        imgui.dummy(V2(1.0, 1.0))
    finally:
        imgui.end()
