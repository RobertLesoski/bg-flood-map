"""Subprocess driver for tests/test_interaction.py: runs the real App on an offscreen canvas (no window) against a data
root given on the command line, drives a few frames and real canvas events, and prints one `RESULT {json}` line.

    python tests/interaction_driver.py <data root> <scenario>        scenarios: flow | close

`flow`   default storm, all hours loaded, 40 frames; reports failed hours, errors, toasts, layer states.
`launcher` the window start-up path (bgv/launch.py) + scripted input on an offscreen canvas.
`close`  as `flow` plus a click on the view, then canvas.close() + app.shutdown(); prints CLOSING right before the close,
         reports the threads still alive afterwards, and exits normally (the parent times the exit).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))


def launcher_scenario(root: str) -> int:
    """The start-up path of the real window (bgv/launch.py: loading screen -> App) on an offscreen canvas, then scripted
    mouse / keyboard events through `canvas.submit_event` like the glfw callbacks make them."""
    import types
    from rendercanvas.offscreen import OffscreenRenderCanvas, loop
    from bgv import config, launch
    config.set_data_root(root)
    args = types.SimpleNamespace(storm=None, msaa=4, quiet=True, size=None, no_vsync=True, hour=None, camera=None, at=None,
                                 exag=None, sun=None, ripple=None, haze=None)
    canvas = OffscreenRenderCanvas(size=(1280, 720), pixel_ratio=1.0)
    L = launch.Launcher(args, canvas, loop, time.perf_counter(), selftest=True)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 90 and L._state not in ("done", "failed"):
        canvas._process_events()
        canvas.draw()
        n += 1
        time.sleep(0.003)
    res = {"state": L._state, "error": repr(L.error) if L.error is not None else None, "frames_to_load": n,
           "times": {k: round(v, 2) for k, v in L.times.items()}}
    app = L.app
    if L._state == "done":
        def pump(k=1):
            for _ in range(k):
                canvas._process_events()
                canvas.draw()
                time.sleep(0.004)

        def send(kind, x=0.0, y=0.0, button=0, buttons=(), **kw):
            canvas.submit_event({"event_type": kind, "x": float(x), "y": float(y), "button": button, "buttons": tuple(buttons),
                                 "modifiers": (), **kw})
        pump(30)
        m = app.get_layer("overlay").manager
        cx, cy = (m.L.sx0 + m.L.sx1) / 2, (m.L.sy0 + m.L.sy1) / 2
        t_a, d_a = app.camera.target.copy(), float(app.camera.dist)
        send("pointer_move", cx, cy)
        pump(2)
        send("pointer_down", cx, cy, 1, (1,))
        pump(1)
        for i in range(1, 7):
            send("pointer_move", cx - 25 * i, cy + 6 * i, 0, (1,))
            pump(1)
        pump(20)
        send("pointer_up", cx - 150, cy + 36, 1, ())
        pump(60)
        t_b = app.camera.target.copy()
        send("wheel", cx, cy, dx=0.0, dy=-250.0)
        pump(60)
        d_b = float(app.camera.dist)
        send("key_down", key="w")
        pump(30)
        send("key_up", key="w")
        pump(60)
        res.update(panned=float(((t_b[:2] - t_a[:2]) ** 2).sum() ** 0.5), zoom=d_b / d_a,
                   walked=float(((app.camera.target[:2] - t_b[:2]) ** 2).sum() ** 0.5), errors=list(app.errors),
                   loading=bool(app.loading))
    print("RESULT " + json.dumps(res), flush=True)
    if app is not None:
        app.shutdown()
    return 0


def main(root: str, scenario: str) -> int:
    if scenario == "launcher":
        return launcher_scenario(root)
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import tools
    from bgv.app import App
    from bgv.gpu.context import GpuContext

    ctx = GpuContext.create(verbose=False)
    canvas = OffscreenRenderCanvas(size=(800, 500), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
    app = App(ctx, root, storm="25yr", msaa=4, quiet=True, settings={}, colour_format=fmt)
    app.attach_canvas(canvas, ui=True)
    canvas.request_draw(app.draw_frame)
    t0 = time.perf_counter()
    while True:                                                   # all hours (do not wait for the stats hub: it may fail)
        app.flow.pump()
        r, n = app.flow.progress()
        if r >= n and app.flow._cur is None:
            break
        if time.perf_counter() - t0 > 90:
            raise TimeoutError("flow did not finish")
        time.sleep(0.002)
    for _ in range(40):
        canvas._process_events()
        canvas.draw()
        time.sleep(0.01)
    res = {
        "failed": {str(k): v for k, v in app.flow.failed.items()},
        "progress": list(app.flow.progress()),
        "errors": list(app.errors),
        "toasts": [it[0] for it in app.get_layer("overlay").manager.toasts.items],
        "layers": {l.name: bool(l.enabled) for l in app.layers},
        "scene_state": getattr(getattr(app, "_scene_data", None), "state", None),
        "frames": app.frame_index,
        "vram_mb": ctx.vram_mb(),
        "root": str(root),
    }
    if scenario == "close":
        W, H = canvas.get_physical_size()
        for kind, extra in (("pointer_move", {}), ("pointer_down", {"button": 1}), ("pointer_up", {"button": 1})):
            canvas.submit_event({"event_type": kind, "x": W / 2, "y": H / 2, "button": extra.get("button", 0),
                                 "buttons": (1,) if kind != "pointer_up" else (), "modifiers": ()})
            canvas._process_events()
            canvas.draw()
        print("RESULT " + json.dumps(res), flush=True)
        print("CLOSING", flush=True)
        t1 = time.perf_counter()
        canvas.close()
        app.shutdown()
        deadline = t1 + 3.0

        def ours():
            return [t.name for t in threading.enumerate()
                    if t is not threading.main_thread() and t.name.startswith(("bgv-", "basemap-"))]
        while time.perf_counter() < deadline and ours():
            time.sleep(0.02)
        nd = [t.name for t in threading.enumerate() if t is not threading.main_thread() and not t.daemon]
        print("THREADS " + json.dumps({"non_daemon": nd, "ours": ours(), "shutdown_s": time.perf_counter() - t1,
                                       "all": [t.name for t in threading.enumerate()]}), flush=True)
        return 0
    app.shutdown()
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
