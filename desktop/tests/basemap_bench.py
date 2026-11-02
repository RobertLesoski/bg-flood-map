"""Headless benchmark with the basemap layer ON (synthetic composite draped on the terrain) or OFF - same scripted camera
path as `python -m bgv --benchmark`; prints the JSON result.

    python tests/basemap_bench.py --on  [--seconds 8] [--px 4096]
    python tests/basemap_bench.py --off [--seconds 8]
    python tests/basemap_bench.py --on --fixed "1205,1486,450,20,40"     fixed-camera frame time instead of the orbit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import basemap_synth as S                                # noqa: E402

SCRATCH = Path(r"C:\Users\Minec\AppData\Local\Temp\claude\C--Users-Minec-Documents-BG-3D-Map\6d20f384-9e95-4606-82dd-08b76f90f515\scratchpad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", action="store_true")
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--px", type=int, default=4096)
    ap.add_argument("--opacity", type=float, default=0.85)
    ap.add_argument("--fixed", default=None, help='"x,y,dist,yaw,pitch": time N frames at this camera')
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--ab", action="store_true", help="interleaved ON/OFF GPU pass timing in ONE process at 3 fixed cameras")
    a = ap.parse_args()
    on = a.on and not a.off
    if a.fixed:
        return fixed(a, on)
    if a.ab:
        return ab(a)
    from bgv import tools
    from bgv.__main__ import build_parser
    args = build_parser().parse_args(["--benchmark", "--json", "--seconds", str(a.seconds), "--size", a.size])
    orig = tools.make_app

    def patched(args_, ctx, fmt="rgba8unorm-srgb", settings=None):
        st = {"layers": {"basemap": {"enabled": on, "state": {"opacity": a.opacity}}}}
        app = orig(args_, ctx, fmt, st)
        if on:
            comp = S.cached_synth(tools.config.data_root(), app.terrain.frame, a.px, SCRATCH)
            S.install(app, comp)
        return app
    tools.make_app = patched
    return tools.run_benchmark(args)


def ab(a):
    """Interleaved A/B in one process (cancels load drift from other programs): GPU pass ms with the map ON vs OFF."""
    import numpy as np
    import wgpu
    from bgv import config
    from bgv.tools import wait_for_flow
    w, h = (int(v) for v in a.size.split("x"))
    app, canvas, ctx = S.make_app({"layers": {"basemap": {"enabled": True, "state": {"opacity": a.opacity}}}}, (w, h))
    lay = S.install(app, S.cached_synth(config.data_root(), app.terrain.frame, a.px, SCRATCH))
    app.t = 12.0
    wait_for_flow(app, all_hours=False)
    tex = ctx.device.create_texture(size=(w, h, 1), format=app.colour_format,
                                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    view = tex.create_view()
    out = {"size": [w, h], "px": a.px, "views": {}}
    views = {"overview": (0, 0, 16000, 0, 55), "mid": (800, 900, 2200, 35, 42), "street": (1205, 1486, 450, 20, 40)}
    for name, cam in views.items():
        app.camera.set_view(*cam)
        acc = {True: [], False: []}
        wall = {True: [], False: []}
        for rnd in range(8):
            for state in ((rnd % 2 == 0), (rnd % 2 == 1)):
                lay.enabled = state
                for i in range(70):
                    app.step(1 / 60)
                    t0 = time.perf_counter()
                    app.render_scene(view, w, h, 1 / 60)
                    ctx.sync()
                    t1 = time.perf_counter()
                    g = app.timer.read_blocking()
                    if i >= 10:
                        wall[state].append((t1 - t0) * 1000)
                        if g is not None:
                            acc[state].append(g)
        out["views"][name] = {("on" if k else "off"): {"gpu_ms_median": float(np.median(v)) if v else None,
                                                       "gpu_ms_p99": float(np.percentile(v, 99)) if v else None,
                                                       "wall_ms_median": float(np.median(wall[k])),
                                                       "wall_ms_p99": float(np.percentile(wall[k], 99))}
                              for k, v in acc.items()}
    out["errors"] = list(app.errors)
    print(json.dumps(out, indent=1))
    app.shutdown()


def fixed(a, on):
    import numpy as np
    import wgpu
    from bgv import config
    w, h = (int(v) for v in a.size.split("x"))
    st = {"layers": {"basemap": {"enabled": on, "state": {"opacity": a.opacity}}}}
    app, canvas, ctx = S.make_app(st, (w, h))
    if on:
        S.install(app, S.cached_synth(config.data_root(), app.terrain.frame, a.px, SCRATCH))
    from bgv.tools import wait_for_flow
    app.t = 12.0
    wait_for_flow(app, all_hours=True)
    tex = ctx.device.create_texture(size=(w, h, 1), format=app.colour_format,
                                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    view = tex.create_view()
    app.camera.set_view(*[float(v) for v in a.fixed.split(",")])
    for _ in range(30):
        app.step(1 / 60)
        app.render_scene(view, w, h, 1 / 60)
    ctx.sync()
    walls, gpus = [], []
    for _ in range(240):
        app.step(1 / 60)
        t0 = time.perf_counter()
        app.render_scene(view, w, h, 1 / 60)
        ctx.sync()
        walls.append((time.perf_counter() - t0) * 1000)
        g = app.timer.read_blocking()
        if g is not None:
            gpus.append(g)
    res = {"basemap": on, "camera": a.fixed, "wall_ms_mean": float(np.mean(walls)), "wall_ms_p99": float(np.percentile(walls, 99)),
           "gpu_ms_mean": float(np.mean(gpus)) if gpus else None, "gpu_ms_p99": float(np.percentile(gpus, 99)) if gpus else None,
           "errors": list(app.errors)}
    print(json.dumps(res))
    app.shutdown()


if __name__ == "__main__":
    sys.exit(main() or 0)
