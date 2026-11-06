"""Frame time of the full viewer (scene + interface + extensions) offscreen, county, 1920x1080, with one feature of the flood-tools
pack switched on, along the benchmark camera path.  Used for docs/PERF.md ("Feature pack").

    python docs/features_perf.py <config>        configs: off | route | heat | exposure | duration | hc | all
Prints one line: config, frames, synced fps (wall mean), mean / p95 / p99 ms.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROUTE = "-86.4436,36.9685,-86.40,36.99"


def main(cfg: str) -> int:
    env = {"route": {"BGV_FEATURES_ROUTE": ROUTE}, "heat": {"BGV_FEATURES_HEAT": "1"}, "exposure": {"BGV_FEATURES_TAB": "1"},
           "duration": {"BGV_FEATURES_TAB": "2"}, "hc": {"BGV_FEATURES_HC": "1"},
           "all": {"BGV_FEATURES_ROUTE": ROUTE, "BGV_FEATURES_HEAT": "1", "BGV_FEATURES_TAB": "1", "BGV_FEATURES_HC": "1"}}.get(cfg, {})
    os.environ.update(env)
    import numpy as np
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import tools
    from bgv.gpu.context import GpuContext
    w, h = 1920, 1080
    ctx = GpuContext.create(verbose=False)
    canvas = OffscreenRenderCanvas(size=(w, h), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        storm, msaa, quiet, exag, sun, hour, camera, at, ripple, haze, domain = "100yr", 4, True, None, None, 12.0, None, None, None, None, "county"
    app = tools.make_app(A, ctx, fmt)
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=True)
    app.get_layer("overlay").state.seen_help = True
    canvas.set_physical_size(w, h)
    canvas.request_draw(app.draw_frame)
    m = app.get_layer("overlay").manager
    m.LITE_ON_MS = float("inf")                   # measure the full interface every frame (no adaptive half-rate UI)
    base = app.domain.overview_dist(app.terrain.frame)
    for _ in range(60):                           # warm-up: features set up, heat layer computed / uploaded
        canvas.draw()
    lay = app.get_layer("hoursflooded")
    t0 = time.time()
    while lay is not None and lay.enabled and lay._loaded_for != app.storm.name and time.time() - t0 < 300:
        canvas.draw()
    n = 600
    walls = []
    for i in range(n):
        tools.bench_camera(app, i / n, base)
        t1 = time.perf_counter()
        canvas.draw()
        ctx.sync()
        walls.append((time.perf_counter() - t1) * 1000.0)
    a = np.array(walls)
    errs = list(app.errors)[:3] if hasattr(app, "errors") else []
    print(f"{cfg}: frames {n}  fps {1000.0 / a.mean():.1f}  mean {a.mean():.2f}  p95 {np.percentile(a, 95):.2f}  "
          f"p99 {np.percentile(a, 99):.2f} ms  heat {'drawn' if lay is not None and lay.draws else 'off'}  errors {errs}")
    pm = sorted(m.panel_ms.items(), key=lambda kv: -kv[1])[:4]
    print("  panel ms: " + ", ".join(f"{k} {v:.2f}" for k, v in pm))
    app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "off"))
