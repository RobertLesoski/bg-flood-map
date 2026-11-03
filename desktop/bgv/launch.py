"""Window start-up: the window and a loading screen come first (well under 1.5 s), everything else loads behind it.

Timeline (this laptop, cached data):  0.0 s process start -> ~0.5 s window visible with a splash painted by GDI (no GPU needed)
-> the GPU adapter / device is created in a worker thread (about 1.4 s, mostly inside the Vulkan driver, GIL released)
while a second thread opens the storm and prepares the terrain (CPU only) -> the App is created in `defer` mode, the canvas
is configured and the imgui loading screen takes over -> the App builds terrain / flow textures / every layer in small
steps (`App.step_init`, at most ~10 ms of work per frame, so the window keeps repainting and stays responsive) -> the
normal frame loop.  Heavy per-session caches (buildings, roads, statistics) were already built by worker threads with a
progress bar inside the sidebar; flow hours stream in the background as before.

`start(args)` -> Launcher;  `Launcher.app` is None until the GPU is ready.  `Launcher.times` records the milestones.
"""
from __future__ import annotations

import sys
import threading
import time


class Splash:
    """A dark client area with the product name and a progress bar, drawn with plain GDI straight into the window (before any
    GPU object exists).  Windows only; every failure silently disables it."""

    def __init__(self, canvas):
        self.hwnd = None
        self.ok = False
        self._last = 0.0
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes
            import glfw
            self.hwnd = int(glfw.get_win32_window(canvas._window))
            u, g = ctypes.windll.user32, ctypes.windll.gdi32
            u.GetDC.argtypes, u.GetDC.restype = [wintypes.HWND], wintypes.HDC
            u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
            u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
            u.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH]
            u.DrawTextW.argtypes = [wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(wintypes.RECT), wintypes.UINT]
            g.CreateSolidBrush.argtypes, g.CreateSolidBrush.restype = [wintypes.COLORREF], wintypes.HBRUSH
            g.DeleteObject.argtypes = [wintypes.HGDIOBJ]
            g.SelectObject.argtypes, g.SelectObject.restype = [wintypes.HDC, wintypes.HGDIOBJ], wintypes.HGDIOBJ
            g.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
            g.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
            g.CreateFontW.restype = wintypes.HFONT
            self._u, self._g, self._ct, self._wt = u, g, ctypes, wintypes
            self.ok = True
        except Exception:
            self.ok = False

    def paint(self, frac: float, msg: str = "Loading", force: bool = False):
        if not self.ok:
            return
        now = time.perf_counter()
        if not force and now - self._last < 0.05:
            return
        self._last = now
        try:
            ct, wt, u, g = self._ct, self._wt, self._u, self._g
            hdc = u.GetDC(self.hwnd)
            if not hdc:
                return
            try:
                r = wt.RECT()
                u.GetClientRect(self.hwnd, ct.byref(r))
                w, h = r.right - r.left, r.bottom - r.top
                if w < 4 or h < 4:
                    return
                rgb = lambda R, G, B: R | (G << 8) | (B << 16)             # COLORREF 0x00BBGGRR
                bg = g.CreateSolidBrush(rgb(29, 34, 48))          # == the clear colour of App._draw_loading in the sRGB swap chain
                u.FillRect(hdc, ct.byref(r), bg)
                g.DeleteObject(bg)
                bw, bh = min(420, int(w * 0.6)), 8
                x0, y0 = (w - bw) // 2, h // 2
                track = g.CreateSolidBrush(rgb(40, 48, 66))
                fill = g.CreateSolidBrush(rgb(77, 160, 235))
                u.FillRect(hdc, ct.byref(wt.RECT(x0, y0, x0 + bw, y0 + bh)), track)
                u.FillRect(hdc, ct.byref(wt.RECT(x0, y0, x0 + int(bw * max(0.02, min(frac, 1.0))), y0 + bh)), fill)
                g.DeleteObject(track)
                g.DeleteObject(fill)
                font = g.CreateFontW(-22, 0, 0, 0, 600, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI")
                small = g.CreateFontW(-15, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 5, 0, "Segoe UI")
                g.SetBkMode(hdc, 1)
                old = g.SelectObject(hdc, font)
                g.SetTextColor(hdc, rgb(158, 204, 255))
                u.DrawTextW(hdc, "BG Flood Viewer", -1, ct.byref(wt.RECT(0, y0 - 44, w, y0 - 10)), 0x1 | 0x20 | 0x8000)
                g.SelectObject(hdc, small)
                g.SetTextColor(hdc, rgb(130, 142, 160))
                u.DrawTextW(hdc, "Loading - " + msg, -1, ct.byref(wt.RECT(0, y0 + 18, w, y0 + 44)), 0x1 | 0x20 | 0x8000)
                g.SelectObject(hdc, old)
                g.DeleteObject(font)
                g.DeleteObject(small)
            finally:
                u.ReleaseDC(self.hwnd, hdc)
        except Exception:
            self.ok = False


class Launcher:
    """Owns the start-up state machine; its `tick` is the canvas draw callback until the App has finished loading."""

    def __init__(self, args, canvas, loop, t_proc0: float, selftest: bool = False):
        from . import config
        self.args, self.canvas, self.loop = args, canvas, loop
        self.t0 = t_proc0
        self.selftest = selftest
        self.times: dict[str, float] = {"window": time.perf_counter() - t_proc0}
        self.app = None
        self.error: BaseException | None = None
        self.settings = {} if selftest else config.load_settings()
        from . import domain as domainmod
        domainmod.apply_cli(self.settings, args)               # --domain / --county-dem ... (remembered when the app saves)
        self.splash = Splash(canvas)
        self.splash.paint(0.02, "Starting", force=True)
        self._ctx = None
        self._ctx_err = None
        self._state = "gpu"
        self._t_splash = 0.0
        self.on_loaded = []                          # callables(app) run once, right after the App finished loading
        # create the process-wide wgpu instance HERE, before any thread: the canvas surface (created on this thread by
        # get_wgpu_context below) and the adapter requested by the GPU thread must belong to the same instance, otherwise
        # wgpu-core panics with "Surface does not exist" (lazy creation races between the two threads)
        try:
            import wgpu.backends.wgpu_native._api as _api
            _api.get_wgpu_instance()
        except Exception:
            pass
        root = config.data_root()
        from .startup import Prefetch
        self.domain, self.domain_notice = domainmod.resolve_or_city(self.settings.get("domain") or "city", root, self.settings)
        self.prefetch = Prefetch(root, args.storm, self.settings, domain=self.domain)
        # City / County chooser on the loading screen: skipped by --domain, by the selftest and by the setting
        # `domain_chooser: false`.  Only the remembered domain is built (above); a click restarts the build for the other.
        self.chooser = None
        if not selftest and not getattr(args, "domain", None) and self.settings.get("domain_chooser", True):
            from .ui.chooser import DomainChooser
            self.chooser = DomainChooser(domainmod.domain_infos(root, self.settings), self.domain.key)
        self._gpu_thread = threading.Thread(target=self._make_ctx, name="bgv-gpu-init", daemon=True)
        self._gpu_thread.start()
        # rendercanvas cancels every draw until the canvas has a context (`_draw_and_present`: "context is None"), so `tick`
        # would never run: create the (still unconfigured, present() is then a no-op) context now.  INTERACT agent fix.
        canvas.get_wgpu_context()
        canvas.request_draw(self.tick)

    # ------------------------------------------------------------------ worker
    def _make_ctx(self):
        try:
            from .gpu.context import GpuContext
            self._ctx = GpuContext.create(verbose=not getattr(self.args, "quiet", False))
        except BaseException as e:                   # reported on the main thread
            self._ctx_err = e

    # ------------------------------------------------------------------ main thread
    def tick(self):
        try:
            if self._state == "gpu":
                self._tick_gpu()
            elif self._state == "app":
                self._tick_app()
        except BaseException as e:
            self.error = e
            self._state = "failed"
            try:
                self.canvas.close()
            except Exception:
                pass

    def _tick_gpu(self):
        if self._gpu_thread.is_alive():
            el = time.perf_counter() - self.t0
            self.splash.paint(min(0.30, 0.04 + 0.10 * el), "Starting the graphics card")
            time.sleep(0.008)
            return
        if self._ctx_err is not None:
            raise self._ctx_err
        self.times["gpu"] = time.perf_counter() - self.t0
        from . import config, tools
        from .app import App
        ctx = self._ctx
        cctx = self.canvas.get_wgpu_context()
        fmt = cctx.get_preferred_format(ctx.adapter)
        a = self.args
        self.app = App(ctx, config.data_root(), storm=a.storm, msaa=a.msaa, settings=self.settings, colour_format=fmt,
                       defer=True, prefetch=self.prefetch, domain=self.domain, chooser=self.chooser)
        self.app.attach_canvas(self.canvas, ui=True)
        self.times["app_created"] = time.perf_counter() - self.t0
        self._state = "app"
        self.canvas.request_draw(self.tick)

    def _tick_app(self):
        app = self.app
        app.draw_frame()
        if app.loading:
            if "first_loading_frame" not in self.times:
                self.times["first_loading_frame"] = time.perf_counter() - self.t0
            return
        # finished
        from . import tools
        tools.apply_view_args(app, self.args)
        self.times["loaded"] = time.perf_counter() - self.t0
        self._state = "done"
        self.canvas.request_draw(app.draw_frame)
        for fn in self.on_loaded:
            fn(app)
        if not getattr(self.args, "quiet", False):
            t = self.times
            print("[bgv] start-up: window %.2f s, GPU %.2f s, loaded %.2f s" % (t["window"], t.get("gpu", 0), t["loaded"]),
                  flush=True)


def start(args, title: str = "BG Flood Viewer", selftest: bool = False) -> tuple["Launcher", object, object]:
    """Create the window at once and return (launcher, canvas, loop); the caller runs `loop.run()`."""
    t_proc0 = getattr(sys.modules.get("__main__"), "T_PROCESS_START", time.perf_counter())
    from rendercanvas.auto import RenderCanvas, loop
    w, h = 1600, 900
    if getattr(args, "size", None):
        ws, _, hs = args.size.lower().partition("x")
        w, h = max(int(ws), 16), max(int(hs), 16)
    canvas = RenderCanvas(size=(w, h), title=title, update_mode="continuous", max_fps=1000, vsync=not args.no_vsync)
    return Launcher(args, canvas, loop, t_proc0, selftest), canvas, loop
