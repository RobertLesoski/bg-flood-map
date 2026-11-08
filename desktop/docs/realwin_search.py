"""Real-window check of the address search box with REAL OS input (Windows SendInput, not canvas.submit_event).

    python docs/realwin_search.py [--domain county] [--out shot.png]

Opens the viewer, clicks the search box with the OS mouse, types an address plus keys bound to hotkeys, then reports
the box text and whether any hotkey state changed.  Captures ONLY the app window rectangle.  Exit code 0 = pass.
"""
from __future__ import annotations

import ctypes
import json
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

U = ctypes.windll.user32
TEXT = "1309 Scottsville Rd"
HOTKEYS = "tlbgxvum5 [],.h"          # bound to tour / time-lapse / storms / play / hours / hide UI ...


def _vk(ch):
    r = U.VkKeyScanW(ord(ch))
    return r & 0xFF, bool(r & 0x100)


def _key(vk, up=False):
    U.keybd_event(vk, U.MapVirtualKeyW(vk, 0), 2 if up else 0, 0)


def type_text(s, delay=0.03):
    for ch in s:
        _guard()
        vk, shift = _vk(ch)
        if shift:
            _key(0x10)
        _key(vk)
        _key(vk, True)
        if shift:
            _key(0x10, True)
        time.sleep(delay)


def click(x, y):
    _guard(x, y)
    U.SetCursorPos(int(x), int(y))
    time.sleep(0.15)
    _guard(x, y)
    U.mouse_event(2, 0, 0, 0, 0)
    time.sleep(0.05)
    U.mouse_event(4, 0, 0, 0, 0)


def find_window(title):
    """The largest visible top-level window of THIS process (the title may be changed by the app)."""
    import os
    pid, best = os.getpid(), [0, 0]
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(h, _):
        p = wintypes.DWORD()
        U.GetWindowThreadProcessId(h, ctypes.byref(p))
        if p.value == pid and U.IsWindowVisible(h):
            l, t, r, b = window_rect(h)
            if (r - l) * (b - t) > best[1]:
                best[:] = [h, (r - l) * (b - t)]
        return True
    U.EnumWindows(proto(cb), 0)
    return best[0]


class NotOurs(RuntimeError):
    pass


_H = [0]


def _guard(x=None, y=None):
    """Never send input anywhere but the viewer: the foreground window (and the window under the click) must be ours."""
    h = _H[0]
    if U.GetForegroundWindow() != h:
        raise NotOurs("viewer is not the foreground window: input aborted")
    if x is not None:
        U.WindowFromPoint.argtypes = [wintypes.POINT]
        w = U.GetAncestor(U.WindowFromPoint(wintypes.POINT(int(x), int(y))), 2)
        if w != h:
            raise NotOurs("another window covers the click point: input aborted")


def focus(h):
    """Bring the window to the front and keep it topmost for the check (only our own window is changed)."""
    _H[0] = h
    U.ShowWindow(h, 9)
    U.SetWindowPos(h, -1, 0, 0, 0, 0, 0x0001 | 0x0002)      # HWND_TOPMOST, no move / size
    k32 = ctypes.windll.kernel32
    for _ in range(5):                                       # Windows' focus lock: borrow the foreground thread's input state
        fg = U.GetForegroundWindow()
        t_fg = U.GetWindowThreadProcessId(fg, None)
        t_me = k32.GetCurrentThreadId()
        U.AttachThreadInput(t_me, t_fg, True)
        U.BringWindowToTop(h)
        U.SetForegroundWindow(h)
        U.AttachThreadInput(t_me, t_fg, False)
        time.sleep(0.4)
        if U.GetForegroundWindow() == h:
            return True
    return False


def capture(h, out):
    """Window content only (PrintWindow): never pixels of other windows or the desktop."""
    from PIL import Image
    l, t, r, b = window_rect(h)
    w, hh = r - l, b - t
    G = ctypes.windll.gdi32
    hdc = U.GetWindowDC(h)
    mdc = G.CreateCompatibleDC(hdc)
    bmp = G.CreateCompatibleBitmap(hdc, w, hh)
    G.SelectObject(mdc, bmp)
    U.PrintWindow(h, mdc, 2)                               # PW_RENDERFULLCONTENT
    buf = ctypes.create_string_buffer(w * hh * 4)

    class BMI(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("a", wintypes.LONG), ("b", wintypes.LONG),
                    ("c", wintypes.DWORD), ("d", wintypes.DWORD)]
    bi = BMI(ctypes.sizeof(BMI), w, -hh, 1, 32, 0, 0, 0, 0, 0, 0)
    G.GetDIBits(mdc, bmp, 0, hh, buf, ctypes.byref(bi), 0)
    Image.frombuffer("RGBA", (w, hh), buf, "raw", "BGRA", 0, 1).convert("RGB").save(out)
    G.DeleteObject(bmp)
    G.DeleteDC(mdc)
    U.ReleaseDC(h, hdc)


def client_origin(h):
    p = wintypes.POINT(0, 0)
    U.ClientToScreen(h, ctypes.byref(p))
    return p.x, p.y


def window_rect(h):
    r = wintypes.RECT()
    U.GetWindowRect(h, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="county")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    from bgv.__main__ import build_parser
    from bgv import launch
    title = "BG Flood Viewer (search check)"
    args = build_parser().parse_args(["--domain", a.domain, "--size", "1400x850"])
    L, canvas, loop = launch.start(args, title=title)
    res = {}

    def snapshot(app):
        m = app.get_layer("overlay").manager
        from bgv.ui import panels_search
        c = panels_search.controller(m)
        cin = {k: getattr(m, k, None) for k in ("show_help",)}
        return {"query": c.query, "storm": app.storm.name, "playing": bool(app.playing), "t": round(float(app.t), 3),
                "hidden": bool(m.st.hidden), "layers": {l.name: bool(l.enabled) for l in app.layers},
                "tour": bool(getattr(getattr(m, "cin", None), "tour_running", False)), **cin}

    def frames(app):
        """Per drawn frame: active item id + focused window name, logged when they change (who steals the focus)."""
        from imgui_bundle import imgui
        fl = res.setdefault("frames", [])
        orig = app.draw_frame

        def draw():
            orig()
            try:
                g = imgui.get_current_context()
                nw = g.nav_window.name if g.nav_window is not None else None
                row = [int(g.active_id), nw]
            except Exception as e:
                row = [repr(e)]
            if (not fl or fl[-1][1:] != row) and len(fl) < 200:
                fl.append([round(time.perf_counter(), 3)] + row)
        canvas.request_draw(draw)
    def driver(app):
        try:
            time.sleep(2.5)
            h = find_window(title)
            res["hwnd"], res["rect"], res["foreground"] = int(h or 0), window_rect(h), focus(h)
            if not h or not res["foreground"]:
                raise NotOurs("could not bring the viewer to the front: no input sent")
            m = app.get_layer("overlay").manager
            from bgv.ui import panels_search
            c = panels_search.controller(m)
            x, y, w = c.bar_rect
            pr = canvas.get_pixel_ratio()
            ox, oy = client_origin(h)
            app.playing = False
            time.sleep(0.3)
            res["before"] = snapshot(app)
            res["click"] = (ox + (x + 60) * pr, oy + (y + 18) * pr, pr, ox, oy, x, y)
            click(ox + (x + 60) * pr, oy + (y + 18) * pr)
            time.sleep(0.25)                      # no wait for a frame on purpose beyond this: keys follow the click fast
            type_text(TEXT)
            type_text(HOTKEYS)
            t_w = time.perf_counter()                 # imgui trickles queued key events (one per frame): at a low
            while time.perf_counter() - t_w < 30.0:   # loading frame rate the box fills in later, never out of order
                if len(snapshot(app)["query"]) >= len(TEXT + HOTKEYS):
                    break
                time.sleep(0.25)
            time.sleep(0.6)
            res["after"] = snapshot(app)
            if a.out:
                capture(h, a.out)
            if U.GetForegroundWindow() == h:
                _key(0x1B)
                _key(0x1B, True)
        except Exception as e:
            res["error"] = repr(e)
        finally:
            time.sleep(0.3)
            loop.call_soon(canvas.close)

    def trace(app):
        res["trace"] = tr = []

        def log(ev):
            if ev.get("event_type") == "key_down" and len(tr) < 80:
                io = app.imgui_renderer.backend.io
                m = app.get_layer("overlay").manager
                from bgv.ui import panels_search
                tr.append((round(time.perf_counter(), 3), ev.get("key"), bool(io.want_text_input), bool(io.want_capture_keyboard),
                           bool(m.text_focused()), bool(panels_search.controller(m).input_active),
                           bool(ev.get("stop_propagation"))))
        canvas.add_event_handler(log, "key_down", order=-98.5)

    L.on_loaded.append(trace)
    L.on_loaded.append(frames)
    L.on_loaded.append(lambda app: threading.Thread(target=driver, args=(app,), daemon=True).start())
    loop.call_later(240.0, canvas.close)
    loop.run()
    if L.app is not None:
        L.app.shutdown()
    b, f = res.get("before", {}), res.get("after", {})
    want = TEXT + HOTKEYS
    changed = {k: (b.get(k), f.get(k)) for k in b if k != "query" and b.get(k) != f.get(k)}
    res["pass"] = f.get("query") == want and not changed and "error" not in res
    res["changed"] = changed
    print(json.dumps({"query": f.get("query"), "want": want, "changed": changed, "pass": res["pass"],
                      "error": res.get("error")}))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
