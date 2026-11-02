"""Headless UI benchmark: frame cost of the whole interface on top of the scene (offscreen, never opens a window).

    python -m bgv.ui.uibench [--size 1920x1080] [--frames 300] [--json] [--storm 25yr]

For two views (overview and a close street view with a selected building) it renders `frames` frames twice: scene only
and scene + the complete imgui interface (sidebar, timeline chart, at-risk list, card, legends, moving mouse probe), each
frame synced to the GPU.  Reports mean / p95 / p99 / max ms and the UI's own CPU time (imgui build + draw-data submit).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def _stats(a):
    a = np.asarray(a, np.float64)
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "n": int(a.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--storm", default="25yr")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    import wgpu
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from .. import tools
    from ..gpu.context import GpuContext
    from . import bridge

    w, h = tools.parse_size(args.size)
    ctx = GpuContext.create(verbose=False)
    canvas = OffscreenRenderCanvas(size=(w, h), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        storm = args.storm
        msaa = 4
        quiet = True
        exag = None
        sun = None
        hour = 12.0
        camera = None
        at = None
        ripple = None
        haze = None
    app = tools.make_app(A, ctx, fmt)
    app.attach_canvas(canvas, ui=True)
    tools.wait_for_flow(app, all_hours=True)
    bridge.ensure_hub(app)
    hub = app._stats_hub
    hub.wait_ready(180.0)
    hub.wait_timeline(240.0)
    canvas.set_physical_size(w, h)
    canvas.request_draw(app.draw_frame)
    ov = app.get_layer("overlay")
    st = ov.state
    st.seen_help = True
    io = app.imgui_renderer.backend.io
    tex = ctx.device.create_texture(label="uibench", size=(w, h, 1), format=app.colour_format,
                                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    view = tex.create_view()
    fl = bridge.flood(app)
    bid = int(fl["buildings"].order[0])
    b = bridge.buildings(app)
    X, Y = float(b.cx[bid]), float(b.cy[bid])
    views = {"overview": (0.0, 0.0, 14000.0, 0.0, 55.0), "street": (X, Y, 260.0, 20.0, 45.0)}
    out = {"adapter": ctx.adapter_name, "size": [w, h], "frames": args.frames, "views": {}}
    dt = 1.0 / 60.0
    for vname, cam in views.items():
        res = {}
        for ui_on in (False, True):
            app.camera.set_view(*cam)
            bridge.selection(app)["building"] = bid if (vname == "street" and ui_on) else None
            app.t = 6.0
            walls, uis, scenes = [], [], []
            n = args.frames
            for i in range(n + 40):
                f = max(i - 40, 0) / n
                app.t = 6.0 + 16.0 * f
                app.step(dt)
                px = w * (0.35 + 0.3 * f)                   # mouse sweeps across the 3D view (probe + hover picking)
                py = h * (0.35 + 0.3 * np.sin(f * 12.0) * 0.5 + 0.15)
                io.add_mouse_pos_event(px, py)
                t0 = time.perf_counter()
                cpu = app.render_scene(view, w, h, dt)
                t1 = time.perf_counter()
                if ui_on:
                    app.imgui_renderer.render()
                t2 = time.perf_counter()
                ctx.sync()
                t3 = time.perf_counter()
                if i >= 40:
                    walls.append((t3 - t0) * 1000.0)
                    scenes.append(cpu)
                    uis.append((t2 - t1) * 1000.0)
            mean = float(np.mean(walls))
            res["ui_on" if ui_on else "ui_off"] = {"frame_ms": _stats(walls), "fps_mean": 1000.0 / mean,
                                                    "scene_cpu_ms": _stats(scenes), "ui_cpu_ms": _stats(uis) if ui_on else None}
        res["ui_cost_ms_mean"] = res["ui_on"]["frame_ms"]["mean"] - res["ui_off"]["frame_ms"]["mean"]
        out["views"][vname] = res
    out["panel_errors"] = ov.manager.panel_errors
    out["errors"] = list(app.errors)
    if args.json:
        print(json.dumps(out, indent=1))
    else:
        for vname, res in out["views"].items():
            for k in ("ui_off", "ui_on"):
                s = res[k]["frame_ms"]
                print(f"[uibench] {vname:9s} {k:6s} {res[k]['fps_mean']:6.1f} fps  mean {s['mean']:.2f}  p95 {s['p95']:.2f}  "
                      f"p99 {s['p99']:.2f}  max {s['max']:.2f} ms" + (f"   ui cpu mean {res[k]['ui_cpu_ms']['mean']:.2f} ms"
                                                                    if res[k]['ui_cpu_ms'] else ""))
            print(f"[uibench] {vname:9s} UI cost {res['ui_cost_ms_mean']:+.2f} ms mean")
        if out["panel_errors"] or out["errors"]:
            print("[uibench] errors:", out["panel_errors"], out["errors"])
    app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
